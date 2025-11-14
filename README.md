# Cocktail Historian

A RAG (Retrieval-Augmented Generation) pipeline for processing and indexing historical cocktail books from the late 1800s and early 1900s. This project extracts, cleans, and chunks text from vintage bartending manuals to enable semantic search and question-answering over historical cocktail recipes and techniques.

## Overview

This pipeline processes scanned PDFs of historical cocktail books, handling OCR when necessary, cleaning the extracted text, and chunking it into structured JSONL files suitable for vector embeddings and RAG applications.

## Project Structure

```
cocktail-historian/
├── data_raw/
│   ├── original/          # Original PDF files
│   └── ocr/               # OCR-processed PDFs (generated when needed)
├── data_derived/
│   ├── text/              # Extracted and cleaned text files
│   └── chunks/            # Chunked JSONL files for RAG
├── scripts/
│   ├── 01_extract_text.py # Extract text from PDFs
│   └── 02_chunk_text.py   # Chunk text into structured JSONL
└── requirements.txt       # Python dependencies
```

## Prerequisites

- Python 3.x
- [OCRmyPDF](https://ocrmypdf.readthedocs.io/) (for OCR processing when needed)
  - Install via: `pip install ocrmypdf` or `brew install ocrmypdf` (macOS)

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The pipeline consists of sequential scripts that should be run in order:

### Step 1: Extract Text from PDFs

```bash
python scripts/01_extract_text.py
```

This script:
- Processes all PDFs in `data_raw/original/`
- Extracts text using PyMuPDF
- Detects if OCR is needed (based on empty page ratio)
- Runs OCRmyPDF for PDFs that need OCR processing
- Normalizes text (removes ligatures, page numbers, extra whitespace)
- Saves cleaned text to `data_derived/text/` with metadata headers

**Output**: Text files in `data_derived/text/` (one `.txt` file per PDF)

### Step 2: Chunk Text into Structured JSONL

```bash
python scripts/02_chunk_text.py
```

This script:
- Reads cleaned text files from `data_derived/text/`
- Filters out OCR garbage lines
- Identifies sections using heuristics (short, all-caps lines)
- Chunks sections into ~400-600 token pieces (max 2200 chars, 300 char overlap)
- Creates structured JSONL files with metadata

**Output**: JSONL files in `data_derived/chunks/` (one `.jsonl` file per book)

Each chunk in the JSONL contains:
- `id`: Unique chunk identifier
- `book_id`: Normalized book identifier
- `book_title`: Full book title
- `year`: Publication year (if extractable)
- `author`: Author name (if extractable)
- `section_index`: Index of the section
- `section_title`: Title of the section (if detected)
- `chunk_index`: Index of the chunk within the section
- `text`: The actual chunk text

## Data Sources

The project currently includes cocktail books from the late 1800s and early 1900s, including works by:
- Jerry Thomas (1862)
- Harry Johnson (multiple editions)
- Hon. Wm. Boothby (1908)
- And many others spanning 1827-1939

## Future Scripts

Additional scripts will be added to complete the RAG pipeline, including:
- Vector embedding generation
- Index creation and management
- Query interface

## Notes

- The scripts are idempotent: re-running them will skip already-processed files
- OCR processing can be slow for large PDFs; processed OCR files are cached in `data_raw/ocr/`
- Text normalization handles common OCR issues like ligatures and page number artifacts
- Chunking uses sentence boundaries and section detection to maintain context

