# web-novel-translator

A production-ready CLI tool for translating EPUB web novels and light novels using the OpenAI API.
Preserves all HTML structure, chapter formatting, and EPUB metadata — only the text is translated.

---

## Requirements

- Python **3.10 or newer** (3.12 recommended)
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/web-novel-translator.git
cd web-novel-translator
```

Or simply place `web-novel-translator.py`, `config.yaml`, and `requirements.txt` in the same folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|---|---|
| Windows (PowerShell) | `.venv\\Scripts\\Activate.ps1` |
| Windows (CMD) | `.venv\\Scripts\\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

The safest method is an environment variable — it never touches any file on disk.

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Windows (permanent, via System Settings):**
```
System Properties → Environment Variables → New → OPENAI_API_KEY
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively, add it directly to `config.yaml` (not recommended for shared machines):
```yaml
openai:
  api_key: "sk-..."
```

---

## Configuration

Edit `config.yaml` to change the model, token limits, or retry behaviour.
All fields are optional — omitted keys fall back to built-in defaults.

```yaml
openai:
  model: "gpt-5.4-mini"        # OpenAI model to use
  temperature: 0.4              # 0.0 = literal, 1.0 = creative
  max_tokens_per_chunk: 32000   # Tokens sent per API call
  request_timeout: 300          # Seconds before a request times out

translation:
  retries: 3                    # Retry attempts on API errors
  retry_base_delay: 10          # Base delay in seconds (× attempt number)

logging:
  level: "INFO"                 # DEBUG | INFO | WARNING | ERROR
```

To use a custom config file path, pass `--config`:
```bash
python web-novel-translator.py --config my-settings.yaml translate ...
```

---

## Usage

### Translate an EPUB

```bash
python web-novel-translator.py translate \\
    --input  novel.epub \\
    --output translated.epub \\
    --from-lang Chinese \\
    --to-lang English
```

### Translate a specific chapter range

```bash
python web-novel-translator.py translate \\
    --input  novel.epub \\
    --output translated.epub \\
    --from-lang Japanese \\
    --to-lang French \\
    --from-chapter 5 \\
    --to-chapter 10
```

### Inspect chapters without translating

```bash
python web-novel-translator.py show-chapters --input novel.epub
```

Prints each chapter's filename, ID, size, media type, and a 300-character text preview.

### Enable verbose debug logging

```bash
python web-novel-translator.py --debug translate --input novel.epub --output out.epub \\
    --from-lang Korean --to-lang English
```

---

## Command Reference

```
usage: web-novel-translator [-h] [--config FILE] [--debug] {translate,show-chapters} ...

Options:
  --config FILE   Path to YAML config file (default: config.yaml)
  --debug         Force DEBUG-level logging for this run

Subcommands:
  translate
    --input       Source .epub file path               (required)
    --output      Destination .epub file path          (required)
    --from-lang   Source language (e.g. Chinese)       (required)
    --to-lang     Target language (e.g. English)       (required)
    --from-chapter  First chapter to translate (default: 1)
    --to-chapter    Last chapter to translate  (default: all)

  show-chapters
    --input       .epub file to inspect                (required)
```

---

## Tips

- **First run?** Use `show-chapters` to verify chapter numbering before committing to a full translation.
- **Large books:** Start with `--from-chapter 1 --to-chapter 3` to test quality and cost before translating everything.
- **Cost estimate:** Each chapter is roughly 1 000–5 000 tokens for a typical web novel chapter. Check [OpenAI pricing](https://openai.com/api/pricing/) for your model.
- **Interrupted run:** Re-run with `--from-chapter N` where `N` is the first untranslated chapter to resume from where you left off.

---

## Project Structure

```
web-novel-translator/
├── web-novel-translator.py   # Main script
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## License

MIT — free to use, modify, and distribute.
