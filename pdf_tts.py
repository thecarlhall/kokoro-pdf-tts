import sys
import wave
import os
import re
import struct
import subprocess
import time
from collections import Counter

import fitz  # pymupdf
import numpy as np


SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


# --- PDF text extraction (shared with orpheus pipeline) ---

def _detect_body_size(doc):
    size_chars = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        size_chars[round(span["size"])] += len(text)
    return size_chars.most_common(1)[0][0] if size_chars else 12


def _classify(size, body_size):
    ratio = size / body_size
    if ratio > 1.8:
        return "title"
    if ratio > 1.3:
        return "section"
    if ratio > 1.1:
        return "deck"
    if ratio >= 0.85:
        return "body"
    return "skip"


def _is_artifact(text):
    return bool(re.match(r'^[A-Za-z]{1,8}$', text))


def _unclosed_quote(line):
    if line[-1:] not in "!?":
        return False
    straight = line.count('"')
    curly_open = line.count('\u201c')
    curly_close = line.count('\u201d')
    return straight % 2 == 1 or (curly_open - curly_close) == 1


def extract_text_from_pdf(pdf_path, stop_at=None, skip_exact=(), skip_prefixes=()):
    doc = fitz.open(pdf_path)
    body_size = _detect_body_size(doc)
    span_floor = body_size * 0.84
    print(f"  body_size={body_size}pt  span_floor={span_floor:.1f}pt  pages={len(doc)}")

    output_lines = []
    prev_role = None
    title_parts = []
    done = False

    for page in doc:
        if done:
            break
        for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if done:
                break
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                spans = [s for s in line["spans"]
                         if s["text"].strip() and s["size"] >= span_floor]
                if not spans:
                    continue

                text = "".join(s["text"] for s in spans).strip()
                if not text or _is_artifact(text):
                    continue
                if text in skip_exact:
                    continue
                if skip_prefixes and any(text.startswith(p) for p in skip_prefixes):
                    continue
                if stop_at and text.startswith(stop_at):
                    done = True
                    break

                dom = max(spans, key=lambda s: s["size"])
                size = round(dom["size"], 1)
                role = _classify(size, body_size)

                if role == "skip":
                    continue
                if role == "title":
                    title_parts.append(text)
                    continue

                if title_parts and role != "title":
                    output_lines.append(" ".join(title_parts))
                    title_parts = []

                if role in ("section", "deck"):
                    output_lines.append("")
                    output_lines.append(text)
                    output_lines.append("")
                elif role == "body":
                    if re.match(r'^By [A-Z]', text) and len(text) < 60:
                        output_lines.append("")
                        output_lines.append(text if text[-1] in ".!?" else text + ".")
                        output_lines.append("")
                    elif text.endswith("?") and len(text) < 80 and "\n" not in text:
                        output_lines.append("")
                        output_lines.append(text)
                        output_lines.append("")
                    elif (output_lines and prev_role == "body" and output_lines[-1]):
                        last = output_lines[-1]
                        if last[-1] not in ".!?" or _unclosed_quote(last):
                            output_lines[-1] = last + " " + text
                            continue
                        else:
                            output_lines.append(text)
                    else:
                        output_lines.append(text)

                prev_role = role

    if title_parts:
        output_lines.append(" ".join(title_parts))

    return re.sub(r'\n{3,}', '\n\n', "\n".join(output_lines)).strip()


# --- Claude text cleaning ---

CLEAN_PROMPT = """\
You are cleaning text that was extracted from a PDF for text-to-speech conversion.
The extraction process sometimes leaves artifacts that would sound wrong or jarring when read aloud.

Clean the text by:
- Removing standalone page numbers (e.g. a line that is just "42" or "Page 12")
- Removing inline footnote/endnote reference numbers embedded mid-sentence (e.g. "the study found1 that" → "the study found that")
- Removing figure and table labels that appear as orphaned fragments (e.g. "Figure 3." or "Table 1:" on their own)
- Removing header/footer repetition artifacts (repeated titles, running heads, URLs, DOIs)
- Removing isolated single characters or meaningless short tokens left by layout parsing
- Fixing any words that were split across lines and joined with a hyphen mid-word (e.g. "impor-tant" → "important")

Do NOT:
- Change any actual sentence content, wording, or meaning
- Remove numbers that are part of sentences (e.g. "over 3 million people")
- Add, rewrite, or summarise anything

Return only the cleaned text with no commentary or explanation.

TEXT TO CLEAN:
"""


def clean_text_with_claude(text, model="claude-haiku-4-5-20251001"):
    print("Cleaning extracted text with Claude...")
    result = subprocess.run(
        ["claude", "-p", "--model", model, CLEAN_PROMPT + text],
        capture_output=True, text=True, check=True,
    )
    cleaned = result.stdout.strip()
    print(f"  Claude cleaned: {len(text)} → {len(cleaned)} chars "
          f"(removed {len(text) - len(cleaned)})")
    return cleaned


# --- Validation ---

def _normalise_for_diff(text):
    text = text.lower()
    text = re.sub(r'[—–\-]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def validate_audio(wav_path, expected_text):
    import difflib
    import whisper

    print("\nLoading Whisper model (base.en)...")
    model = whisper.load_model("base.en")
    print(f"Transcribing {wav_path}...")
    result = model.transcribe(wav_path, language="en", fp16=False)
    transcribed = result["text"]

    exp_words = _normalise_for_diff(expected_text).split()
    got_words = _normalise_for_diff(transcribed).split()
    matcher = difflib.SequenceMatcher(None, exp_words, got_words, autojunk=False)
    opcodes = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes()
               if tag != 'equal']

    print(f"\n=== DIFF ({len(opcodes)} differences) ===")
    for tag, i1, i2, j1, j2 in opcodes[:40]:
        exp_snip = " ".join(exp_words[i1:i2])
        got_snip = " ".join(got_words[j1:j2])
        print(f"  [{tag}] expected: {exp_snip!r:50s}  got: {got_snip!r}")

    print(f"\nSimilarity: {matcher.ratio():.1%}")
    return matcher.ratio()


# --- TTS ---

def text_to_audio(text, voice=DEFAULT_VOICE, output_file="output.mp3", speed=1.0, fmt="mp3"):
    from kokoro import KPipeline

    print(f"Loading Kokoro (voice={voice})...")
    pipe = KPipeline(lang_code="a")

    audio_chunks = []
    sentence_count = 0
    t0 = time.monotonic()

    for result in pipe(text, voice=voice, speed=speed, split_pattern=r'\n+'):
        if result.audio is not None:
            audio_chunks.append(result.audio)
            sentence_count += 1
            if sentence_count % 10 == 0:
                elapsed = time.monotonic() - t0
                audio_secs = sum(len(a) for a in audio_chunks) / SAMPLE_RATE
                print(f"  {sentence_count} sentences | {audio_secs:.0f}s audio | {elapsed:.0f}s elapsed")

    if not audio_chunks:
        print("WARNING: no audio generated")
        return

    combined = np.concatenate(audio_chunks)
    pcm = (combined * 32767).astype(np.int16)

    if fmt == "mp3":
        cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
               "-i", "pipe:0", output_file]
        subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, check=True)
    else:
        with wave.open(output_file, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm.tobytes())

    total_elapsed = time.monotonic() - t0
    audio_dur = len(combined) / SAMPLE_RATE
    print(f"Done: {output_file}  ({audio_dur:.1f}s audio in {total_elapsed:.1f}s — {audio_dur/total_elapsed:.1f}x real-time)")


def file_to_speech(file_path, voice=DEFAULT_VOICE, output=None, speed=1.0, dry_run=False,
                  stop_at=None, skip_exact=(), skip_prefixes=(), fmt="mp3"):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    stem = re.sub(r'[^\w\-. ]', '_', stem).strip()

    if output is None:
        output = f"{stem}.{fmt}"

    raw_file = f"{stem}.txt"
    clean_file = f"{stem}-clean.txt"
    if os.path.exists(clean_file):
        print(f"Using existing cleaned text file: {clean_file}")
        with open(clean_file) as f:
            text = f.read()
    else:
        print(f"Extracting text from: {file_path}")
        raw_text = extract_text_from_pdf(file_path, stop_at=stop_at,
                                         skip_exact=skip_exact, skip_prefixes=skip_prefixes)
        with open(raw_file, "w") as f:
            f.write(raw_text)
        print(f"Raw text saved to: {raw_file}  ({len(raw_text)} chars)")
        text = clean_text_with_claude(raw_text)
        with open(clean_file, "w") as f:
            f.write(text)
        print(f"Cleaned text saved to: {clean_file}  ({len(text)} chars)")

    if dry_run:
        print(text)
        return

    text_to_audio(text, voice=voice, output_file=output, speed=speed, fmt=fmt)
    validate_audio(output, text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a file (pdf or txt) to a speech audio file via Kokoro TTS.")
    parser.add_argument("file", help="Path to the file")
    parser.add_argument("voice", nargs="?", default=DEFAULT_VOICE,
                        help=f"Voice to use (default: {DEFAULT_VOICE}). Options: {', '.join(VOICES)}")
    parser.add_argument("--output", "-o", help="Output audio path (default: <pdf stem>.<format>)")
    parser.add_argument("--format", "-f", choices=["wav", "mp3"], default="mp3",
                        help="Output format (default: mp3)")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                        help="Speech speed multiplier (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract text only — no TTS, no audio")
    args = parser.parse_args()

    file_to_speech(
        args.file,
        voice=args.voice,
        output=args.output,
        speed=args.speed,
        dry_run=args.dry_run,
        fmt=args.format,
        stop_at="Corrections & Amplifications",
        skip_prefixes=("Appeared in the",),
    )
