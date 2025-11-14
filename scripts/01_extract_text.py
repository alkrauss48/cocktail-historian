# scripts/01_extract_text.py
import os, sys, re, subprocess, hashlib
from pathlib import Path
import fitz  # PyMuPDF

BASE = Path(__file__).resolve().parents[1]
RAW_ORIG = BASE / "data_raw" / "original"
RAW_OCR  = BASE / "data_raw" / "ocr"
OUT_TXT  = BASE / "data_derived" / "text"

RAW_ORIG.mkdir(parents=True, exist_ok=True)
RAW_OCR.mkdir(parents=True, exist_ok=True)
OUT_TXT.mkdir(parents=True, exist_ok=True)

LIGATURES = {
    "\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl","ſ":"s"
}

def normalize(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)   # strip lone page numbers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    page_texts = []
    empty_pages = 0
    for p in doc:
        txt = p.get_text("text") or ""
        if len(txt.strip()) < 30:
            empty_pages += 1
        page_texts.append(txt)
    return "\n\n".join(page_texts), empty_pages, len(doc)

def needs_ocr(empty_pages: int, total_pages: int) -> bool:
    if total_pages == 0:
        return False
    ratio = empty_pages / total_pages
    return (empty_pages >= 2) or (ratio > 0.15)

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
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    pdfs = sorted(p for p in RAW_ORIG.glob("*.pdf"))
    if not pdfs:
        print("No PDFs in data_raw/original/")
        sys.exit(0)

    for orig_pdf in pdfs:
        print(f"\n==> {orig_pdf.name}")
        txt_path = OUT_TXT / (orig_pdf.stem + ".txt")
        if txt_path.exists():
            print("   -> text already extracted, skipping")
            continue

        ocr_pdf_path = RAW_OCR / orig_pdf.name

        # Prefer OCR’d version if it exists
        if ocr_pdf_path.exists():
            print("   -> found existing OCR’d PDF, using that")
            use_pdf = ocr_pdf_path
        else:
            # Try original
            text, empty, total = extract_text_from_pdf(orig_pdf)
            print(f"   pages: {total}, near-empty: {empty}")

            if needs_ocr(empty, total):
                print("   -> running OCR and saving to data_raw/ocr/")
                ocr_pdf(orig_pdf, ocr_pdf_path)
                use_pdf = ocr_pdf_path
                text, empty2, total2 = extract_text_from_pdf(ocr_pdf_path)
                print(f"   after OCR pages: {total2}, near-empty: {empty2}")
            else:
                use_pdf = orig_pdf

        # If we haven’t extracted text yet for this loop, do it now
        if 'text' not in locals() or not text.strip() or Path(use_pdf) != orig_pdf:
            text, _, _ = extract_text_from_pdf(use_pdf)

        text = normalize(text)
        header = f"# SOURCE_MD5: {md5(use_pdf)}\n# FILE: {use_pdf.name}\n\n"
        txt_path.write_text(header + text, encoding="utf-8")
        print(f"   -> wrote {txt_path.relative_to(BASE)}")

if __name__ == "__main__":
    main()
