# kokoro-pdf-tts

Turn any PDF — article, memo, report — into a high-quality audio file you actually want to listen to. One command, one output, no API keys.

```
uv run pdf_tts.py "article.pdf"  →  article.wav
```

## How it works

1. **Extract** — PyMuPDF reads the PDF and classifies spans by font size relative to the body. Titles, section headers, bylines, and body text are each handled to preserve the document's natural rhythm. Short artifacts and footnote-sized spans are discarded.
2. **Synthesize** — [Kokoro](https://github.com/hexgrad/kokoro) generates audio locally at 5–6x real-time. No cloud API, no per-character charges.
3. **Validate** — Whisper transcribes the output WAV and runs a word-level diff against the source text. The similarity score prints after every run so you can catch regressions.

## Performance

| Document                           | Audio length | Generation time | Speed          |
|------------------------------------|--------------|-----------------|----------------|
| WSJ article (5 pages)              | 7.7 min      | 81 s            | 5.7x real-time |
| Financial analysis memo (18 pages) | 59.8 min     | 10.8 min        | 5.5x real-time |

Target: articles under 5,000 words in under 60 seconds; long memos under 15 minutes.

## Setup

**Prerequisites**

```bash
brew install espeak-ng   # required by Kokoro for phonemization
```

**Install**

```bash
git clone https://github.com/thecarlhall/kokoro-pdf-tts
cd kokoro-pdf-tts
uv sync
uv pip install pip       # one-time workaround: spaCy calls pip at runtime on first use
```

The first run downloads the spaCy `en_core_web_sm` model (~13 MB) and the Whisper `base.en` model. Subsequent runs skip both.

## Usage

```bash
# Default voice (af_heart — warm American female)
uv run pdf_tts.py "article.pdf"

# Choose a voice
uv run pdf_tts.py "memo.pdf" am_michael

# Adjust speed
uv run pdf_tts.py "article.pdf" --speed 1.2

# Preview extracted text without generating audio
uv run pdf_tts.py "article.pdf" --dry-run

# Specify output path
uv run pdf_tts.py "article.pdf" --output listen.wav
```

## Voices

| Voice                                         | Description                      |
|-----------------------------------------------|----------------------------------|
| `af_heart`                                    | American female — warm (default) |
| `af_bella`, `af_nicole`, `af_sarah`, `af_sky` | American female variants         |
| `am_adam`, `am_michael`                       | American male                    |
| `bf_emma`, `bf_isabella`                      | British female                   |
| `bm_george`, `bm_lewis`                       | British male                     |

## Dependencies

- [Kokoro](https://github.com/hexgrad/kokoro) — local neural TTS
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [OpenAI Whisper](https://github.com/openai/whisper) — audio validation
- [uv](https://docs.astral.sh/uv/) — package manager / runner

## Notes

- **PDF quality matters.** Extraction is the highest-leverage point in the pipeline. If audio sounds wrong, run `--dry-run` first and inspect the extracted text.
- **Validation quirks.** Whisper normalizes differently from source text (e.g. "4" ↔ "four", "S&P" ↔ "s and p"). These show up as diff noise and don't indicate audio quality problems.
- **Column layouts.** Extraction is tuned for articles and memos. Multi-column academic PDFs may need `stop_at` / `skip_prefixes` tuning in the source.
