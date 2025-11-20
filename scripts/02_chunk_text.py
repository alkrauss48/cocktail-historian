# scripts/02_chunk_text.py

import re
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TXT_DIR = BASE / "data_derived" / "text"
CHUNK_DIR = BASE / "data_derived" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Chunking parameters - roughly 400–600 tokens
MAX_CHARS = 1600
OVERLAP_CHARS = 300

# Chunk validation thresholds
MIN_CHUNK_LENGTH = 20
SHORT_CHUNK_THRESHOLD = 50
MIN_ALPHANUMERIC_RATIO = 0.4
MAX_DOT_RATIO = 0.15
MIN_SENTENCES_FOR_PROSE = 2
PROSE_LENGTH_THRESHOLD = 100

# OCR garbage detection thresholds
MAX_CONSONANT_RATIO = 0.6
MIN_CONSONANT_CLUSTER = 10
CAPS_RATIO_THRESHOLD = 0.6
TITLE_CAPS_RATIO = 0.7
TITLE_LENGTH_MAX = 150

# List/index detection thresholds
MIN_LINES_FOR_LIST = 5
SHORT_LINE_RATIO_THRESHOLD = 0.7
TRAILING_NUMBER_RATIO_THRESHOLD = 0.3
VERY_SHORT_LINE_RATIO = 0.8
SHORT_LINE_LENGTH = 40


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


def has_publisher_info(text: str) -> bool:
    """Check if text contains publisher/printing information."""
    publisher_patterns = [
        r"publication[s]?",
        r"pub['\"]?[hl]",  # catches "PubHcationa", "Pub'l", etc.
        r"press\s*(ltd|inc)?",
        r"company",
        r"by\s+mail"
    ]
    lower_text = text.lower()
    return any(re.search(pattern, lower_text) for pattern in publisher_patterns)


def has_ocr_consonant_clusters(text: str) -> bool:
    """Check for unusual consonant clusters that indicate OCR garbage."""
    consonants_only = re.sub(r'[aeiouy\s\d]', '', text.lower())
    return (len(consonants_only) > MIN_CONSONANT_CLUSTER and
            len(consonants_only) > len(text) * MAX_CONSONANT_RATIO)


def has_weird_capitalization(text: str) -> bool:
    """Check for random capitalization patterns like 'S:AHMONLNAY'."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    # High caps ratio + weird punctuation = garbage
    return upper_ratio > CAPS_RATIO_THRESHOLD and re.search(r'[:\'"]{1,}[A-Z]', text)


def is_index_entry(text: str) -> bool:
    """Check if text looks like a table of contents or index entry."""
    # Dotted lines with page numbers
    if re.search(r'\.{3,}\s*\d+', text):
        return True

    # Excessive dots (dot leaders)
    dot_count = text.count('.')
    if len(text) > 0 and (dot_count / len(text)) > MAX_DOT_RATIO:
        return True

    return False


def is_list_of_items(text: str) -> bool:
    """Check if text is a list of short items (like an index)."""
    lines = text.split('\n')
    if len(lines) < MIN_LINES_FOR_LIST:
        return False

    short_lines = sum(1 for ln in lines if len(ln.strip()) < SHORT_LINE_LENGTH)
    lines_with_numbers = sum(1 for ln in lines if re.search(r'\d+\s*$', ln.strip()))

    # Mostly short lines with trailing numbers = index
    if ((short_lines / len(lines)) > SHORT_LINE_RATIO_THRESHOLD and
        (lines_with_numbers / len(lines)) > TRAILING_NUMBER_RATIO_THRESHOLD):
        return True

    # Almost all short lines = probably a list, not prose
    if (short_lines / len(lines)) > VERY_SHORT_LINE_RATIO:
        return True

    return False


def has_advertisement_content(text: str) -> bool:
    """Check for advertisement keywords."""
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
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in ad_keywords)


def has_copyright_info(text: str) -> bool:
    """Check for copyright and legal text patterns."""
    copyright_patterns = [
        r'copyright.*\d{4}',
        r'all rights reserved',
        r'library of congress',
        r'catalog.*entry',
        r'press.*ltd',
        r'printing.*company'
    ]
    lower_text = text.lower()
    return any(re.search(pattern, lower_text) for pattern in copyright_patterns)


def is_title_page_spam(text: str) -> bool:
    """Check if text is mostly uppercase title page content."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    return upper_ratio > TITLE_CAPS_RATIO and len(text) < TITLE_LENGTH_MAX


def lacks_prose_structure(text: str) -> bool:
    """Check if text lacks proper sentence structure."""
    # Count complete sentences
    sentence_endings = len(re.findall(r'[.!?][\s\n]', text))

    # Long text without sentences = probably not prose
    if len(text) > PROSE_LENGTH_THRESHOLD and sentence_endings < MIN_SENTENCES_FOR_PROSE:
        # Exception: cocktail recipes don't always have full sentences
        recipe_indicators = ['jigger', 'dash', 'spoonful', 'glass', 'shake', 'stir', 'serve', 'strain']
        if not any(indicator in text.lower() for indicator in recipe_indicators):
            return True

    return False


def is_valuable_chunk(text: str) -> bool:
    """
    Determine if a chunk contains valuable content or is just noise.

    Returns False for:
    - Very short chunks (< MIN_CHUNK_LENGTH)
    - Publisher/printing information
    - OCR garbage and artifacts
    - Index/table of contents entries
    - Advertisements
    - Copyright/legal text
    - Title page spam
    - Text lacking prose structure

    Returns True for cocktail recipes, instructions, and historical content.
    """
    stripped = text.strip()

    # Basic length check
    if len(stripped) < MIN_CHUNK_LENGTH:
        return False

    # Check alphanumeric ratio (catch severe OCR corruption)
    alnum = sum(c.isalnum() for c in stripped)
    if len(stripped) > 0 and (alnum / len(stripped)) < MIN_ALPHANUMERIC_RATIO:
        return False

    # Short chunks need extra scrutiny
    if len(stripped) < SHORT_CHUNK_THRESHOLD:
        if (has_publisher_info(stripped) or
            has_ocr_consonant_clusters(stripped) or
            has_weird_capitalization(stripped)):
            return False

    # Check for various types of non-valuable content
    if (is_index_entry(stripped) or
        is_list_of_items(stripped) or
        has_advertisement_content(stripped) or
        has_copyright_info(stripped) or
        is_title_page_spam(stripped) or
        lacks_prose_structure(stripped)):
        return False

    # If it passes all filters, it's valuable
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
