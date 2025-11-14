# scripts/02_chunk_text.py

import re
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TXT_DIR = BASE / "data_derived" / "text"
CHUNK_DIR = BASE / "data_derived" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Roughly 400–600 tokens; tweak as desired
MAX_CHARS = 1600
OVERLAP_CHARS = 300


def read_clean_text(path: Path) -> str:
    """
    Read the .txt, drop header metadata lines (# SOURCE_MD5, # FILE),
    and return the body as a single string.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # Drop metadata header lines starting with "# "
    body_lines = [ln for ln in lines if not ln.startswith("# ")]

    # Drop obvious OCR garbage lines (few letters, many symbols)
    body_lines = drop_garbage_lines(body_lines)

    return "\n".join(body_lines).strip()


def drop_garbage_lines(lines):
    """
    Remove lines that are likely OCR noise.
    Heuristic: if fewer than 30% of non-space characters are letters, drop it.
    """
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            cleaned.append(ln)
            continue

        total = len(stripped)
        letters = sum(ch.isalpha() for ch in stripped)

        # If mostly non-letters, treat as junk
        if total > 0 and (letters / total) < 0.3:
            continue

        cleaned.append(ln)
    return cleaned


def split_into_sections(text: str):
    """
    Heuristic for section titles:
    - Line is relatively short (<= 80 chars)
    - Mostly caps/numbers/punctuation
    - Does NOT end with a period
    - Has at least a few letters (to avoid "- 9-" page numbers)
    """
    lines = text.splitlines()
    sections = []
    cur_title = None
    cur_buf = []

    def commit():
        if cur_buf:
            sections.append({
                "title": cur_title,
                "text": "\n".join(cur_buf).strip()
            })

    for ln in lines:
        stripped = ln.strip()

        if not stripped:
            # always keep blank lines inside current section
            cur_buf.append(ln)
            continue

        # Candidate title: caps/punct only
        if (len(stripped) <= 80
            and not stripped.endswith(".")
            and re.match(r"^[A-Z0-9 ,.'&\-\(\)À-ÖØ-öø-ÿ]+$", stripped)):

            # Remove non-letters to check if there's any real text
            alpha = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", stripped)
            if len(alpha) < 3:
                # probably a page number like "- 9-" → not a real title
                cur_buf.append(ln)
                continue

            # New section title
            commit()
            cur_title = stripped.title()
            cur_buf = []
        else:
            cur_buf.append(ln)

    commit()

    # Filter out empty sections
    return [s for s in sections if s["text"]]


def sentence_chunks(text: str, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    """
    Simple sentence-ish chunking: split on punctuation boundaries and
    then pack into chunks up to max_chars with optional overlap.
    """
    # Very simple sentence split; you can swap in nltk later if you like.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    buf = ""

    for s in sentences:
        if not s:
            continue
        candidate = (buf + " " + s).strip() if buf else s

        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)

            # Start new buffer; optional overlap with previous chunk tail
            if overlap and len(s) < overlap and chunks:
                buf = chunks[-1][-overlap:] + " " + s
            else:
                buf = s

    if buf:
        chunks.append(buf)

    return chunks


def infer_book_meta(path: Path):
    """
    Very simple metadata inference from filename.
    You can replace this with a YAML manifest later.
    """
    book_title = path.stem
    book_id = re.sub(r'\W+', '_', book_title.lower()).strip('_')
    return {
        "book_id": book_id,
        "book_title": book_title,
        "year": None,
        "author": None,
    }


def main():
    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files in data_derived/text")
        return

    for txt in txt_files:
        print(f"\n==> Chunking {txt.name}")
        meta = infer_book_meta(txt)
        text = read_clean_text(txt)

        sections = split_into_sections(text)
        if not sections:
            # Fallback: whole doc as one section
            sections = [{"title": None, "text": text}]

        records = []
        for sec_idx, sec in enumerate(sections):
            sec_title = sec["title"]
            for chunk_idx, chunk in enumerate(sentence_chunks(sec["text"])):
                records.append({
                    "id": f"{meta['book_id']}:s{sec_idx}:c{chunk_idx}",
                    "book_id": meta["book_id"],
                    "book_title": meta["book_title"],
                    "year": meta["year"],
                    "author": meta["author"],
                    "section_index": sec_idx,
                    "section_title": sec_title,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                })

        out_path = CHUNK_DIR / (txt.stem + ".jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"   -> wrote {out_path.relative_to(BASE)} ({len(records)} chunks)")


if __name__ == "__main__":
    main()
