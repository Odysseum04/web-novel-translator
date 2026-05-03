#!/usr/bin/env python3
"""
web-novel-translator.py
=======================
A production-ready CLI tool for translating EPUB books using the OpenAI API.

Supports:
  - translate   : Translate an EPUB from one language to another.
  - show-chapters : Inspect chapter metadata without modifying anything.

Usage:
    python web-novel-translator.py translate --input novel.epub --output out.epub \
        --from-lang Chinese --to-lang English
    python web-novel-translator.py show-chapters --input novel.epub

Requirements (install via pip):
    openai tiktoken ebooklib beautifulsoup4 tqdm pyyaml lxml
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import ebooklib
import tiktoken
import yaml
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from ebooklib import epub
from openai import APIError, AuthenticationError, OpenAI, RateLimitError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level logger — callers configure via setup_logging()
# ---------------------------------------------------------------------------
# Suppress BeautifulSoup XML-parsed-as-HTML warning (EPUB XHTML is valid XML)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

#: Sentinel value that means "translate all chapters to the end".
ALL_CHAPTERS = float("inf")

#: Hard limit: never send more than this fraction of a model's context as input.
_CONTEXT_SAFETY_MARGIN: float = 0.85

DEFAULT_CONFIG: dict = {
    "openai": {
        "api_key": None,          # Override via OPENAI_API_KEY env var
        "model": "gpt-4o-mini",
        "temperature": 0.4,
        "max_tokens_per_chunk": 12_000,
        "request_timeout": 600,
    },
    "translation": {
        "retries": 3,
        "retry_base_delay": 10,   # seconds; multiplied by attempt number
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    },
}


@dataclass
class AppConfig:
    """Strongly-typed configuration container parsed from a YAML file + defaults."""

    # OpenAI
    api_key: Optional[str]
    model: str
    temperature: float
    max_tokens_per_chunk: int
    request_timeout: int

    # Translation behaviour
    retries: int
    retry_base_delay: int

    # Logging
    log_level: str
    log_format: str

    @classmethod
    def from_dict(cls, raw: dict) -> "AppConfig":
        oai = raw.get("openai", {})
        tr = raw.get("translation", {})
        lg = raw.get("logging", {})
        d_oai = DEFAULT_CONFIG["openai"]
        d_tr = DEFAULT_CONFIG["translation"]
        d_lg = DEFAULT_CONFIG["logging"]
        return cls(
            api_key=oai.get("api_key") or d_oai["api_key"],
            model=oai.get("model", d_oai["model"]),
            temperature=float(oai.get("temperature", d_oai["temperature"])),
            max_tokens_per_chunk=int(oai.get("max_tokens_per_chunk", d_oai["max_tokens_per_chunk"])),
            request_timeout=int(oai.get("request_timeout", d_oai["request_timeout"])),
            retries=int(tr.get("retries", d_tr["retries"])),
            retry_base_delay=int(tr.get("retry_base_delay", d_tr["retry_base_delay"])),
            log_level=lg.get("level", d_lg["level"]).upper(),
            log_format=lg.get("format", d_lg["format"]),
        )


def load_config(config_path: str) -> AppConfig:
    """
    Load a YAML config file and deep-merge it with ``DEFAULT_CONFIG``.

    Falls back to defaults silently if the file is absent; logs a warning on
    parse errors.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A fully populated :class:`AppConfig` instance.
    """
    base = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)

    if not path.exists():
        logger.debug("Config file '%s' not found — using defaults.", config_path)
        return AppConfig.from_dict(base)

    try:
        with path.open(encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        logger.warning("Could not parse config file '%s': %s — using defaults.", config_path, exc)
        return AppConfig.from_dict(base)

    # Deep-merge: user values win, but we keep default keys that aren't specified
    for section, defaults in base.items():
        if section in user and isinstance(defaults, dict) and isinstance(user[section], dict):
            base[section] = {**defaults, **user[section]}
        elif section in user:
            base[section] = user[section]

    return AppConfig.from_dict(base)


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(cfg: AppConfig, *, force_debug: bool = False) -> None:
    """
    Configure the root logger and silence noisy third-party libraries.

    Args:
        cfg:         Application configuration object.
        force_debug: If ``True``, override the configured level with DEBUG.
    """
    level_str = "DEBUG" if force_debug else cfg.log_level
    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format=cfg.log_format,
        stream=sys.stdout,
        force=True,
    )
    # Quiet down HTTP internals — their debug output is extremely verbose
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("Logging initialised at level: %s", level_str)


# ===========================================================================
# Token utilities
# ===========================================================================

_TIKTOKEN_CACHE: dict[str, tiktoken.Encoding] = {}


# Models that use the newer o200k_base tokeniser (GPT-4o family and beyond).
# tiktoken may not yet know about very new model names, so we apply a heuristic.
_O200K_PATTERNS = ("gpt-4o", "o1", "o3", "o4", "gpt-5")


def _infer_encoding_name(model: str) -> str:
    """
    Heuristically select the best available tiktoken encoding for *model*.

    Priority:
      1. tiktoken's own registry (exact match).
      2. o200k_base for GPT-4o / GPT-5 / o-series models.
      3. cl100k_base as a universal fallback.

    Args:
        model: OpenAI model name string (e.g. ``"gpt-5.4-mini"``).

    Returns:
        A tiktoken encoding name string.
    """
    try:
        tiktoken.encoding_for_model(model)
        return model  # tiktoken knows it — use directly
    except KeyError:
        pass
    model_lower = model.lower()
    if any(pattern in model_lower for pattern in _O200K_PATTERNS):
        return "o200k_base"
    return "cl100k_base"


def _get_encoding(model: str) -> tiktoken.Encoding:
    """
    Return a cached ``tiktoken`` encoding for *model*, falling back
    gracefully for models not yet in tiktoken's registry.

    Args:
        model: OpenAI model name.

    Returns:
        A :class:`tiktoken.Encoding` instance.
    """
    if model not in _TIKTOKEN_CACHE:
        try:
            _TIKTOKEN_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding_name = _infer_encoding_name(model)
            logger.debug(
                "tiktoken does not have an exact entry for model '%s'; "
                "using '%s' encoding for token counting.",
                model,
                encoding_name,
            )
            _TIKTOKEN_CACHE[model] = tiktoken.get_encoding(encoding_name)
    return _TIKTOKEN_CACHE[model]


def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens in *text* for the given *model*.

    Args:
        text:  The string to count tokens for.
        model: OpenAI model name used to select the correct tokeniser.

    Returns:
        Number of tokens as an integer.
    """
    return len(_get_encoding(model).encode(text))


# ===========================================================================
# HTML splitting
# ===========================================================================

def _iter_top_level_blocks(body: Tag) -> Iterator[list[str]]:
    """
    Yield groups of consecutive sibling strings from *body* whose combined
    HTML does not exceed ``_CONTEXT_SAFETY_MARGIN`` of tokens.  Grouping is
    done by iterating direct children of the ``<body>`` tag.

    Yields:
        Lists of stringified HTML nodes that together form one chunk.
    """
    # We accumulate child HTML strings and yield a batch when full
    batch: list[str] = []
    for child in body.children:
        batch.append(str(child))
    # Caller controls token budgets; just hand back raw list per child
    yield batch


def split_html_by_top_level_tags(
    html_content: str,
    max_tokens: int,
    model: str,
) -> list[str]:
    """
    Split *html_content* into token-budget-respecting chunks by grouping
    **top-level block elements** (``<p>``, ``<div>``, etc.) from the
    ``<body>``.  This approach avoids mid-tag breaks that break XHTML
    validity.

    If even a single top-level element exceeds *max_tokens*, that element is
    emitted as its own chunk with a warning (the LLM will either handle it or
    the API will return a context-length error that the caller catches).

    Args:
        html_content: Full XHTML/HTML string of one chapter item.
        max_tokens:   Maximum tokens per returned chunk.
        model:        Model name for token counting.

    Returns:
        A list of HTML strings, each within the token budget where possible.
    """
    total = count_tokens(html_content, model)
    if total <= max_tokens:
        return [html_content]

    logger.warning(
        "Chapter HTML (%d tokens) exceeds max_tokens_per_chunk (%d). "
        "Splitting by top-level block elements.",
        total,
        max_tokens,
    )

    soup = BeautifulSoup(html_content, "lxml")
    body = soup.body
    if body is None:
        # Fallback: no body tag — treat whole content as one chunk and warn
        logger.error(
            "Cannot find <body> for intelligent splitting; sending oversized chunk. "
            "Increase max_tokens_per_chunk or this chapter will likely fail."
        )
        return [html_content]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens: int = 0

    for child in body.children:
        child_str = str(child)
        child_tokens = count_tokens(child_str, model)

        if current_tokens + child_tokens > max_tokens and current_parts:
            # Flush current batch
            chunks.append("".join(current_parts))
            current_parts = []
            current_tokens = 0

        if child_tokens > max_tokens:
            logger.warning(
                "A single top-level HTML element is %d tokens (> max %d). "
                "Sending as its own chunk — the API may reject it.",
                child_tokens,
                max_tokens,
            )

        current_parts.append(child_str)
        current_tokens += child_tokens

    if current_parts:
        chunks.append("".join(current_parts))

    logger.debug("Intelligent HTML split produced %d chunk(s).", len(chunks))
    return chunks


# ===========================================================================
# OpenAI translation
# ===========================================================================

def build_system_prompt(from_lang: str, to_lang: str) -> str:
    """
    Build the system prompt instructing the model to translate HTML while
    preserving all tags and attributes.

    Args:
        from_lang: Source language name or code (e.g. ``"Chinese"``).
        to_lang:   Target language name or code (e.g. ``"English"``).

    Returns:
        A string suitable for the ``system`` role in a chat completion.
    """
    lines = [
        f"You are an expert {from_lang}-to-{to_lang} literary translator "
        f"specialised in web novels and light novels.",
        "",
        "RULES — follow exactly:",
        "1. Translate ONLY the human-readable text nodes inside the HTML.",
        "2. PRESERVE every HTML tag, attribute, and whitespace structure EXACTLY.",
        "3. Do NOT translate attribute values (class, id, href, src, alt, etc.).",
        "4. Output ONLY the translated HTML — no explanations, no markdown fences.",
        "5. Maintain the author's tone, register, and narrative voice.",
        f"6. The entire output must be in {to_lang}.",
    ]
    return "\n".join(lines)


def _call_openai_with_retry(
    client: OpenAI,
    cfg: AppConfig,
    system: str,
    user_content: str,
) -> Optional[str]:
    """
    Call the OpenAI Chat Completions API with exponential-back-off retries.

    Handles :exc:`RateLimitError`, generic :exc:`APIError`, and
    :exc:`AuthenticationError` explicitly.  Fatal errors (auth, bad model,
    context overflow) short-circuit immediately without retrying.

    Args:
        client:       Authenticated :class:`openai.OpenAI` instance.
        cfg:          Application config for model params and retry settings.
        system:       System prompt string.
        user_content: The HTML block to translate.

    Returns:
        Translated text string, or ``None`` if all retries are exhausted or a
        fatal error occurs.
    """
    for attempt in range(1, cfg.retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                timeout=cfg.request_timeout,
            )
            result = response.choices[0].message.content
            if not result:
                logger.warning("API returned an empty response (attempt %d/%d).", attempt, cfg.retries)
                return ""
            return result

        except AuthenticationError as exc:
            logger.critical("OpenAI authentication failed: %s — aborting.", exc)
            return None  # No point retrying

        except RateLimitError as exc:
            delay = cfg.retry_base_delay * attempt
            logger.warning(
                "Rate-limited (attempt %d/%d). Retrying in %ds. %s",
                attempt, cfg.retries, delay, exc,
            )
            time.sleep(delay)

        except APIError as exc:
            err_lower = str(exc).lower()
            if any(k in err_lower for k in ("does not exist", "invalid_model", "invalid model")):
                logger.critical(
                    "Model '%s' does not exist or you lack access: %s — aborting.",
                    cfg.model, exc,
                )
                return None
            if any(k in err_lower for k in ("context_length_exceeded", "maximum context length")):
                logger.error(
                    "Context length exceeded for model '%s'. "
                    "Reduce max_tokens_per_chunk or use a model with larger context: %s",
                    cfg.model, exc,
                )
                return None  # Retrying won't help
            delay = cfg.retry_base_delay * attempt
            logger.error(
                "API error (attempt %d/%d). Retrying in %ds. %s",
                attempt, cfg.retries, delay, exc,
            )
            time.sleep(delay)

        except Exception as exc:  # noqa: BLE001
            delay = cfg.retry_base_delay * attempt
            logger.error(
                "Unexpected error (attempt %d/%d). Retrying in %ds. %s",
                attempt, cfg.retries, delay, exc,
                exc_info=True,
            )
            time.sleep(delay)

    logger.error("Exhausted %d retries for this chunk.", cfg.retries)
    return None


def translate_html_chunk(
    client: OpenAI,
    cfg: AppConfig,
    html_chunk: str,
    from_lang: str,
    to_lang: str,
) -> str:
    """
    Translate a single HTML chunk via the OpenAI API.

    On failure, falls back to the original *html_chunk* so the EPUB stays
    structurally valid even if translation fails.

    Args:
        client:     Authenticated OpenAI client.
        cfg:        Application configuration.
        html_chunk: Raw HTML string to translate.
        from_lang:  Source language.
        to_lang:    Target language.

    Returns:
        Translated HTML string (or the original chunk on failure).
    """
    system = build_system_prompt(from_lang, to_lang)
    result = _call_openai_with_retry(client, cfg, system, html_chunk)
    if result is None:
        logger.error(
            "Translation failed; keeping original HTML for this chunk "
            "(first 80 chars: '%s').",
            html_chunk[:80],
        )
        return html_chunk
    return result


def translate_chapter_html(
    client: OpenAI,
    cfg: AppConfig,
    full_html: str,
    from_lang: str,
    to_lang: str,
    chapter_label: str = "",
) -> str:
    """
    Orchestrate splitting, translating, and reassembling a chapter's HTML.

    Extracts the ``<body>`` children for translation to avoid sending
    ``<head>`` boilerplate to the model, then reconstructs the full document.

    Args:
        client:        Authenticated OpenAI client.
        cfg:           Application configuration.
        full_html:     Complete XHTML content of the chapter item.
        from_lang:     Source language.
        to_lang:       Target language.
        chapter_label: Human-readable label used in log messages.

    Returns:
        Full XHTML string with translated body content.
    """
    soup = BeautifulSoup(full_html, "lxml")
    body = soup.body

    if body is None:
        # No <body> — translate the whole thing as a single chunk
        logger.debug("%s: No <body> found; translating entire item.", chapter_label)
        return translate_html_chunk(client, cfg, full_html, from_lang, to_lang)

    body_inner_html = "".join(str(c) for c in body.children)
    chunks = split_html_by_top_level_tags(body_inner_html, cfg.max_tokens_per_chunk, cfg.model)

    if len(chunks) > 1:
        pbar = tqdm(
            chunks,
            desc=f"  ↪ {chapter_label} blocks",
            unit="blk",
            leave=False,
        )
    else:
        pbar = chunks  # type: ignore[assignment]

    translated_parts: list[str] = []
    for chunk in pbar:
        translated_parts.append(
            translate_html_chunk(client, cfg, chunk, from_lang, to_lang)
        )

    translated_body_html = "".join(translated_parts)

    # Graft translated children back into the original <body>
    translated_soup = BeautifulSoup(translated_body_html, "lxml")
    body.clear()
    # lxml wraps content in html/body — use translated_soup.body when available
    source = translated_soup.body or translated_soup
    for node in list(source.children):
        body.append(node)

    return str(soup)


# ===========================================================================
# EPUB processing
# ===========================================================================

@dataclass
class ChapterInfo:
    """Metadata snapshot for a single EPUB document item."""
    number: int
    total: int
    item_name: str
    file_name: str
    item_id: str
    media_type: str
    size_bytes: int
    preview: str


def _safe_decode(content: bytes) -> str:
    """Decode bytes as UTF-8, replacing any invalid bytes."""
    return content.decode("utf-8", errors="replace")


def collect_document_items(book: epub.EpubBook) -> list[epub.EpubItem]:
    """Return all ITEM_DOCUMENT items from *book*."""
    return [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]


def translate_epub(
    client: OpenAI,
    cfg: AppConfig,
    input_path: str,
    output_path: str,
    from_lang: str,
    to_lang: str,
    from_chapter: int = 1,
    to_chapter: float = ALL_CHAPTERS,
) -> None:
    """
    Translate an EPUB file chapter by chapter and write the result to *output_path*.

    Chapters outside the ``[from_chapter, to_chapter]`` range are copied
    untouched.  On any per-chapter error the original content is preserved and
    execution continues.

    Args:
        client:       Authenticated OpenAI client.
        cfg:          Application configuration.
        input_path:   Path to the source ``.epub`` file.
        output_path:  Destination path for the translated ``.epub`` file.
        from_lang:    Source language name.
        to_lang:      Target language name.
        from_chapter: First chapter number to translate (1-based, inclusive).
        to_chapter:   Last chapter number to translate (1-based, inclusive).
                      Use :data:`ALL_CHAPTERS` for "translate everything".
    """
    # --- Load EPUB ---
    try:
        book = epub.read_epub(input_path)
    except FileNotFoundError:
        logger.critical("Input EPUB not found: '%s'", input_path)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Failed to open EPUB '%s': %s", input_path, exc, exc_info=True)
        sys.exit(1)

    doc_items = collect_document_items(book)
    total = len(doc_items)
    logger.info("EPUB contains %d document item(s).", total)

    in_scope = [
        (idx + 1, item)
        for idx, item in enumerate(doc_items)
        if from_chapter <= idx + 1 <= to_chapter
    ]

    if not in_scope:
        logger.warning(
            "No chapters fall in the range [%s, %s]. Writing EPUB without changes.",
            from_chapter,
            int(to_chapter) if to_chapter != ALL_CHAPTERS else "end",
        )
        _write_epub(book, output_path)
        return

    to_display = int(to_chapter) if to_chapter != ALL_CHAPTERS else total
    logger.info(
        "Translating %d chapter(s) (chapters %d–%d of %d) from %s → %s.",
        len(in_scope), from_chapter, min(to_display, total), total,
        from_lang, to_lang,
    )

    success_count = 0
    with tqdm(in_scope, desc="Translating", unit="ch") as pbar:
        for chapter_num, item in pbar:
            label = f"Ch {chapter_num}/{total} ({item.get_name()[:30]})"
            pbar.set_postfix_str(item.get_name()[:30])
            logger.info("▶ %s", label)

            try:
                original_html = _safe_decode(item.get_content())
                translated_html = translate_chapter_html(
                    client, cfg, original_html, from_lang, to_lang, label
                )
                item.set_content(translated_html.encode("utf-8"))
                success_count += 1
                logger.debug("✓ %s — done.", label)
            except Exception as exc:
                logger.error(
                    "Error processing %s — keeping original content. %s",
                    label, exc, exc_info=True,
                )

    logger.info("Translated %d/%d chapter(s) successfully.", success_count, len(in_scope))
    _write_epub(book, output_path)


def _write_epub(book: epub.EpubBook, output_path: str) -> None:
    """
    Serialise *book* to *output_path*, exiting on failure.

    Args:
        book:        The :class:`epub.EpubBook` to write.
        output_path: Destination file path.
    """
    try:
        epub.write_epub(output_path, book, {})
        logger.info("EPUB written to: %s", output_path)
    except Exception as exc:
        logger.critical("Failed to write EPUB to '%s': %s", output_path, exc, exc_info=True)
        sys.exit(1)


# ===========================================================================
# Show-chapters command
# ===========================================================================

def show_chapters_info(input_path: str) -> None:
    """
    Print detailed metadata for every document item in an EPUB to the log.

    Includes book title, language, table of contents, and a text preview of
    each chapter.

    Args:
        input_path: Path to the EPUB file to inspect.
    """
    try:
        book = epub.read_epub(input_path)
    except FileNotFoundError:
        logger.critical("EPUB not found: '%s'", input_path)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Cannot open EPUB '%s': %s", input_path, exc, exc_info=True)
        sys.exit(1)

    def _meta(key: str) -> str:
        items = book.get_metadata("DC", key)
        if items:
            val = items[0]
            return val[0] if isinstance(val, tuple) else str(val)
        return "(not found)"

    logger.info("══ EPUB Metadata ══════════════════════════")
    logger.info("  Title   : %s", _meta("title"))
    logger.info("  Language: %s", _meta("language"))
    logger.info("  Creator : %s", _meta("creator"))

    logger.info("══ Table of Contents ══════════════════════")
    if book.toc:
        for entry in book.toc:
            link = entry if isinstance(entry, epub.Link) else (
                entry[0] if isinstance(entry, tuple) and isinstance(entry[0], epub.Link) else None
            )
            if link:
                logger.info("  • %s  →  %s", link.title, link.href)
    else:
        logger.info("  (no ToC metadata available)")

    doc_items = collect_document_items(book)
    logger.info("══ Document Items (%d) ════════════════════", len(doc_items))

    for idx, item in enumerate(doc_items, start=1):
        raw = item.get_content()
        size = len(raw)
        preview = "(could not generate preview)"
        try:
            html_text = _safe_decode(raw)
            text = BeautifulSoup(html_text, "lxml").get_text(separator=" ", strip=True)
            text = re.sub(r"\\s+", " ", text).strip()
            preview = text[:300] + ("…" if len(text) > 300 else "")
        except Exception as exc:
            logger.debug("Preview error for item %d: %s", idx, exc)

        logger.info(
            "\n  [%02d/%02d]  %s\n"
            "           id=%s  type=%s  size=%d bytes\n"
            "           preview: %s",
            idx, len(doc_items),
            item.file_name,
            item.id, item.media_type, size,
            preview,
        )


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="web-novel-translator",
        description="Translate EPUB web novels using the OpenAI API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s translate --input novel.epub --output out.epub "
            "--from-lang Chinese --to-lang English\n"
            "  %(prog)s show-chapters --input novel.epub\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force DEBUG-level logging for this run.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- translate sub-command ----
    tr = sub.add_parser("translate", help="Translate an EPUB book.")
    tr.add_argument("--input",     required=True, help="Source EPUB file path.")
    tr.add_argument("--output",    required=True, help="Destination EPUB file path.")
    tr.add_argument("--from-lang", required=True, dest="from_lang",
                    help="Source language (e.g. Chinese, Japanese, Korean).")
    tr.add_argument("--to-lang",   required=True, dest="to_lang",
                    help="Target language (e.g. English, French, German).")
    tr.add_argument("--from-chapter", type=int, default=1, dest="from_chapter",
                    help="First chapter to translate (1-based, inclusive). Default: 1.")
    tr.add_argument("--to-chapter", type=int, default=None, dest="to_chapter",
                    help="Last chapter to translate (1-based, inclusive). Default: all.")

    # ---- show-chapters sub-command ----
    sh = sub.add_parser("show-chapters", help="Display chapter metadata.")
    sh.add_argument("--input", required=True, help="EPUB file to inspect.")

    return parser


def main() -> None:
    """Parse CLI arguments, load configuration, and dispatch to the appropriate handler."""
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, force_debug=args.debug)

    if args.debug:
        logger.debug("DEBUG mode enabled via CLI flag.")

    if args.command == "show-chapters":
        show_chapters_info(args.input)
        return

    # ---- translate ----
    if not cfg.model:
        logger.critical("No OpenAI model specified in configuration. Aborting.")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or cfg.api_key
    if not api_key:
        logger.critical(
            "OpenAI API key not found. Set the OPENAI_API_KEY environment variable "
            "or add 'api_key' under the 'openai' section of your config file."
        )
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key)
    except Exception as exc:
        logger.critical("Failed to create OpenAI client: %s", exc, exc_info=True)
        sys.exit(1)

    to_chapter: float = args.to_chapter if args.to_chapter is not None else ALL_CHAPTERS

    logger.info(
        "Starting translation: '%s' → '%s'  [%s → %s]  chapters %s–%s  model=%s",
        args.input, args.output,
        args.from_lang, args.to_lang,
        args.from_chapter,
        args.to_chapter or "end",
        cfg.model,
    )

    translate_epub(
        client=client,
        cfg=cfg,
        input_path=args.input,
        output_path=args.output,
        from_lang=args.from_lang,
        to_lang=args.to_lang,
        from_chapter=args.from_chapter,
        to_chapter=to_chapter,
    )


if __name__ == "__main__":
    main()
