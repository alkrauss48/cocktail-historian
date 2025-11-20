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


def is_valuable_chunk(text: str) -> bool:
    """
    Determine if a chunk contains valuable content or is just noise.
    Returns False for index entries, ads, publisher info, etc.
    """
    stripped = text.strip()

    # Too short to be valuable
    if len(stripped) < 20:
        return False

    # Very short chunks (< 50 chars) need extra scrutiny
    if len(stripped) < 50:
        # Check for publisher patterns
        publisher_patterns = [
            r"publication[s]?",
            r"pub['\"]?[hl]",  # catches "PubHcationa", "Pub'l", etc.
            r"press\s*(ltd|inc)?",
            r"company",
            r"by\s+mail"
        ]
        lower_text = stripped.lower()
        if any(re.search(pattern, lower_text) for pattern in publisher_patterns):
            return False

        # Check for unusual consonant clusters (OCR artifacts)
        # Remove vowels and spaces, see what's left
        consonants_only = re.sub(r'[aeiouy\s\d]', '', stripped.lower())
        if len(consonants_only) > 10 and len(consonants_only) > len(stripped) * 0.6:
            return False

        # Check for random capitalization patterns (e.g., "S:AHMONLNAY")
        # More than 50% caps in a short chunk that isn't a proper title
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            # If >60% uppercase but has weird characters (colons, random punctuation), likely garbage
            if upper_ratio > 0.6 and re.search(r'[:\'"]{1,}[A-Z]', stripped):
                return False

    # Check if it's mostly OCR garbage (lots of random characters, few real words)
    # Count alphanumeric characters
    alnum = sum(c.isalnum() for c in stripped)
    if len(stripped) > 0 and (alnum / len(stripped)) < 0.4:
        return False

    # Check if it looks like an index/table of contents
    # Pattern: lots of dots with numbers at the end (e.g., "Brandy Cocktail . . . . . 11")
    if re.search(r'\.{3,}\s*\d+', stripped):
        return False

    # Check for excessive ellipsis or dot leaders
    dot_count = stripped.count('.')
    if len(stripped) > 0 and (dot_count / len(stripped)) > 0.15:
        return False

    # Check if it's mostly a list of items (index-like)
    # Pattern: multiple short lines, many with trailing numbers or no punctuation
    lines = stripped.split('\n')
    if len(lines) >= 5:
        short_lines = sum(1 for ln in lines if len(ln.strip()) < 40)
        lines_with_trailing_numbers = sum(1 for ln in lines if re.search(r'\d+\s*$', ln.strip()))

        # If most lines are short and many have trailing numbers, it's likely an index
        if (short_lines / len(lines)) > 0.7 and (lines_with_trailing_numbers / len(lines)) > 0.3:
            return False

        # If it's just a list of short items without much prose, skip it
        if (short_lines / len(lines)) > 0.8:
            return False

    # Check for advertisement keywords
    ad_keywords = [
        'direct from trapper',
        'silver black foxes',
        'all skins are taken',
        'price, gold cloth',
        'cents per copy',
        'published by',
        'printed by',
        'for sale by'
    ]
    lower_text = stripped.lower()
    if any(keyword in lower_text for keyword in ad_keywords):
        return False

    # Check if it's mostly uppercase (likely a title page or header spam)
    alpha_chars = [c for c in stripped if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        # If >70% uppercase and shorter than 150 chars, probably not valuable content
        if upper_ratio > 0.7 and len(stripped) < 150:
            return False

    # Check for copyright/legal text patterns
    copyright_patterns = [
        r'copyright.*\d{4}',
        r'all rights reserved',
        r'library of congress',
        r'catalog.*entry',
        r'press.*ltd',
        r'printing.*company'
    ]
    if any(re.search(pattern, lower_text) for pattern in copyright_patterns):
        return False

    # Check if chunk has very few complete sentences (likely garbage or list)
    # Look for sentence-ending punctuation followed by space or newline
    sentence_endings = len(re.findall(r'[.!?][\s\n]', stripped))
    # If chunk is over 100 chars but has fewer than 2 sentences, it's probably not prose
    if len(stripped) > 100 and sentence_endings < 2:
        # Exception: if it looks like a recipe (has measurements and ingredients)
        recipe_indicators = ['jigger', 'dash', 'spoonful', 'glass', 'shake', 'stir', 'serve', 'strain']
        if not any(indicator in lower_text for indicator in recipe_indicators):
            return False

    # If it has reasonable length and none of the above patterns, it's probably valuable
    return True


def sentence_chunks(text: str, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    """
    Simple sentence-ish chunking: split on punctuation boundaries and
    then pack into chunks up to max_chars with optional overlap.
    Only returns chunks that pass the value check.
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
            if buf and is_valuable_chunk(buf):
                chunks.append(buf)

            # Start new buffer; optional overlap with previous chunk tail
            if overlap and len(s) < overlap and chunks:
                buf = chunks[-1][-overlap:] + " " + s
            else:
                buf = s

    if buf and is_valuable_chunk(buf):
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
