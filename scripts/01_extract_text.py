# scripts/01_extract_text.py

"""
Step 1 of the RAG preprocessing pipeline: Extract and normalize text from PDFs.

This script:
- Processes all PDFs in data_raw/original/
- Extracts text using PyMuPDF
- Detects if OCR is needed (based on empty page ratio)
- Runs OCRmyPDF for PDFs that need OCR processing
- Normalizes text (removes ligatures, control characters, page numbers, extra whitespace)
- Saves cleaned text to data_derived/text/ with metadata headers (MD5 hash, filename)

The script is idempotent: it skips already-processed PDFs unless the source PDF has changed
(detected via MD5 hash comparison).

Dependencies:
    pip install pymupdf ocrmypdf
"""

import os, sys, re, subprocess, hashlib
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

BASE = Path(__file__).resolve().parents[1]
RAW_ORIG = BASE / "data_raw" / "original"
RAW_OCR  = BASE / "data_raw" / "ocr"
OUT_TXT  = BASE / "data_derived" / "text"

RAW_ORIG.mkdir(parents=True, exist_ok=True)
RAW_OCR.mkdir(parents=True, exist_ok=True)
OUT_TXT.mkdir(parents=True, exist_ok=True)

# Constants
LIGATURES = {
    "\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl","ſ":"s"
}
MIN_PAGE_TEXT_LENGTH = 30  # Minimum characters to consider a page non-empty
MIN_EMPTY_PAGES_FOR_OCR = 2  # Minimum empty pages to trigger OCR
EMPTY_PAGE_RATIO_THRESHOLD = 0.15  # Ratio of empty pages to trigger OCR

def normalize(text: str) -> str:
    # Replace typographic ligatures with their component letters (e.g., "ﬁ" -> "fi")
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    # Remove control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    # Remove soft hyphens
    text = text.replace('\u00AD', '')
    # Normalize Unicode whitespace to regular space
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)

    # Strip standalone page numbers (e.g., lines containing only "123")
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Remove lines that look like page numbers with dashes (e.g., "- 9-")
    text = re.sub(r"\n\s*[-—]\s*\d{1,4}\s*[-—]\s*\n", "\n", text)

    # Remove common copyright/library catalog patterns
    text = re.sub(r'COPYRIGHT\s+DEPOS[IU]T?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'TWO\s+COP[IL]-U:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[R|r]ir\.CtlVEC-', '', text)
    text = re.sub(r'OpyqiAMT\s+PMTBV', '', text)
    text = re.sub(r'PLA\d+\s+\^t\s+VXc\s+No\.', '', text)
    text = re.sub(r"'¥Ml\s+LliRARV\s+\d+", '', text)

    # Line-by-line filtering to remove OCR garbage
    lines = text.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Calculate various quality metrics for the line
        total = len(stripped)
        alnum = sum(c.isalnum() for c in stripped)
        alpha = sum(c.isalpha() for c in stripped)
        spaces = stripped.count(' ')

        # Skip very short lines (< 3 chars) unless they look like valid abbreviations
        if total < 3:
            continue

        # Skip lines that are mostly symbols/punctuation (less than 40% alphanumeric)
        if total > 0 and (alnum / total) < 0.40:
            continue

        # Skip lines with no vowels (likely OCR garbage) - but allow very short lines
        if alpha > 5:
            vowels = sum(1 for c in stripped.lower() if c in 'aeiouy')
            if vowels == 0:
                continue
            # Also skip if <5% vowels (likely garbage like "AMONLNAYM")
            if (vowels / alpha) < 0.05:
                continue

        # Skip lines that are all caps AND have no spaces (likely OCR garbage)
        # Exception: keep if it's a plausible title (has spaces)
        if alpha > 0 and stripped.isupper() and spaces == 0 and total > 15:
            continue

        # Skip lines with unusual character patterns (OCR artifacts)
        # e.g., lines with lots of apostrophes or quotes at odd positions
        if stripped.count("'") > 3 or stripped.count('"') > 2:
            # Check if they're in sensible positions (contractions, quotes)
            if not re.search(r"(^['\"]|['\"]$|\w'\w)", stripped):
                continue

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # More aggressive OCR spacing fixes - multiple passes
    # Fix single letter spaces (e.g., "t he" -> "the", "t o" -> "to")
    for _ in range(5):  # Multiple passes to catch consecutive errors
        # Fix "t he" -> "the"
        text = re.sub(r'\b([a-z]) ([a-z]{1,})\b', r'\1\2', text, flags=re.IGNORECASE)
        # Fix "drin k" -> "drink"
        text = re.sub(r'\b([a-z]{2,}) ([a-z])\b', r'\1\2', text, flags=re.IGNORECASE)
        # Fix "a nd" -> "and", "i s" -> "is" (very common OCR errors)
        text = re.sub(r'\b(a|i) (nd|s|t|n|ll|re)\b', r'\1\2', text, flags=re.IGNORECASE)

    # Collapse multiple spaces/tabs into single space
    text = re.sub(r"[ \t]+", " ", text)
    # Limit multiple consecutive newlines to maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace
    return text.strip()

def extract_text_from_pdf(pdf_path: Path):
    """Extract text from PDF and return text, empty page count, and total pages."""
    doc = fitz.open(pdf_path)
    try:
        page_texts = []
        empty_pages = 0
        for p in doc:
            txt = p.get_text("text") or ""
            if len(txt.strip()) < MIN_PAGE_TEXT_LENGTH:
                empty_pages += 1
            page_texts.append(txt)
        return "\n\n".join(page_texts), empty_pages, len(doc)
    finally:
        doc.close()

def needs_ocr(empty_pages: int, total_pages: int) -> bool:
    """Determine if PDF needs OCR based on empty page count and ratio."""
    if total_pages == 0:
        return False
    ratio = empty_pages / total_pages
    return (empty_pages >= MIN_EMPTY_PAGES_FOR_OCR) or (ratio > EMPTY_PAGE_RATIO_THRESHOLD)

def ocr_pdf(in_path: Path, out_path: Path):
    cmd = [
        "ocrmypdf",
        "--deskew",
        "--clean",
        "--optimize", "1",
        "--skip-text",    # only OCR image-only pages
        "--output-type", "pdf",   # avoid PDF/A, just make a normal PDF
        str(in_path), str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   !! OCRmyPDF failed for {in_path.name}: {e}")
        return False


def md5(path: Path) -> str:
    """Calculate MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_stored_md5(txt_path: Path) -> Optional[str]:
    """Extract stored MD5 hash from text file header, or None if not found."""
    if not txt_path.exists():
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line.startswith("# SOURCE_MD5: "):
                return first_line.split("SOURCE_MD5: ", 1)[1].strip()
    except Exception:
        pass
    return None

def main():
    pdfs = sorted(p for p in RAW_ORIG.glob("*.pdf"))
    if not pdfs:
        print("No PDFs in data_raw/original/")
        sys.exit(0)

    for orig_pdf in pdfs:
        print(f"\n==> {orig_pdf.name}")
        txt_path = OUT_TXT / (orig_pdf.stem + ".txt")
        ocr_pdf_path = RAW_OCR / orig_pdf.name

        # Check if already processed and source hasn't changed
        stored_md5 = get_stored_md5(txt_path)
        if stored_md5:
            current_md5 = md5(orig_pdf)
            if stored_md5 == current_md5:
                print("   -> text already extracted, skipping")
                continue
            else:
                print(f"   -> source PDF changed (was {stored_md5[:8]}..., now {current_md5[:8]}...), re-extracting")

        # Determine which PDF to use (original or OCR'd)
        use_pdf = orig_pdf
        text = None

        # Check if OCR'd version exists and prefer it
        if ocr_pdf_path.exists():
            print("   -> found existing OCR'd PDF, using that")
            use_pdf = ocr_pdf_path
        else:
            # Extract from original to check if OCR is needed
            text, empty, total = extract_text_from_pdf(orig_pdf)
            print(f"   pages: {total}, near-empty: {empty}")

            if needs_ocr(empty, total):
                print("   -> running OCR and saving to data_raw/ocr/")
                if ocr_pdf(orig_pdf, ocr_pdf_path):
                    use_pdf = ocr_pdf_path
                    # Re-extract from OCR'd version
                    text, empty2, total2 = extract_text_from_pdf(ocr_pdf_path)
                    print(f"   after OCR pages: {total2}, near-empty: {empty2}")
                else:
                    print("   -> OCR failed, using original PDF")
                    use_pdf = orig_pdf

        # Extract text if we haven't already
        if text is None:
            text, _, _ = extract_text_from_pdf(use_pdf)

        # Normalize and save
        text = normalize(text)
        # Always store the original PDF's MD5 for change detection, not the OCR'd version
        source_md5 = md5(orig_pdf)
        header = f"# SOURCE_MD5: {source_md5}\n# FILE: {use_pdf.name}\n\n"
        try:
            txt_path.write_text(header + text, encoding="utf-8")
            print(f"   -> wrote {txt_path.relative_to(BASE)}")
        except Exception as e:
            print(f"   !! Failed to write {txt_path.name}: {e}")
            continue

if __name__ == "__main__":
    main()
