# Cocktail Historian

A RAG (Retrieval-Augmented Generation) pipeline for processing and indexing historical cocktail books from the late 1800s and early 1900s. This project extracts, cleans, chunks, and embeds text from vintage bartending manuals to enable semantic search and question-answering over historical cocktail recipes and techniques.

## Overview

This pipeline processes scanned PDFs of historical cocktail books, handling OCR when necessary, cleaning the extracted text, chunking it into structured JSONL files, and embedding the chunks into a Qdrant vector database for semantic search and RAG applications.

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
│   ├── 02_chunk_text.py   # Chunk text into structured JSONL
│   └── 03_embed_qdrant.py  # Embed chunks and upsert into Qdrant
└── requirements.txt       # Python dependencies
```

## Prerequisites

- Python 3.x
- [OCRmyPDF](https://ocrmypdf.readthedocs.io/) (for OCR processing when needed)
  - Install via: `pip install ocrmypdf` or `brew install ocrmypdf` (macOS)
- Docker and Docker Compose (for running Qdrant vector database)
  - Install via: [Docker Desktop](https://www.docker.com/products/docker-desktop/) or your system's package manager

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Qdrant vector database (required for Step 3):
   ```bash
   docker-compose up -d
   ```
   This starts Qdrant on `http://localhost:6333` (REST API) and `http://localhost:6334` (gRPC API).

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
- Chunks sections into ~400-500 token pieces (max 1600 chars, 300 char overlap)
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

### Step 3: Embed Chunks and Upsert into Qdrant

```bash
python scripts/03_embed_qdrant.py
```

This script:
- Reads all JSONL chunk files from `data_derived/chunks/`
- Loads a local sentence-transformers embedding model (default: `BAAI/bge-base-en-v1.5`, 768 dimensions)
- Embeds each chunk's text using the local model (no API calls required)
- Creates a Qdrant collection if it doesn't exist (cosine distance, configurable vector size)
- Upserts embeddings and metadata into Qdrant in batches

**Output**: Vector embeddings stored in Qdrant collection `cocktail_chunks` (default)

**Configuration options**:
- `--chunks-dir`: Directory containing JSONL files (default: `data_derived/chunks`)
- `--collection-name`: Qdrant collection name (default: `cocktail_chunks`)
- `--batch-size`: Batch size for embeddings and upserts (default: 64)
- `--embedding-model`: Local embedding model name (default: `BAAI/bge-base-en-v1.5`)
- `--embedding-dim`: Embedding dimension (default: 768 for bge-base-en-v1.5)

**Example usage**:
```bash
# Use a smaller, faster model
python scripts/03_embed_qdrant.py --embedding-model BAAI/bge-small-en-v1.5 --embedding-dim 384

# Custom collection name and batch size
python scripts/03_embed_qdrant.py --collection-name my_cocktails --batch-size 32
```

**Note**: The script is idempotent—re-running it will safely re-upsert chunks (updates existing points with the same ID). Make sure Qdrant is running (`docker-compose up -d`) before executing this script.

## Data Sources

The project currently includes cocktail books from the late 1800s and early 1900s, including works by:
- Jerry Thomas (1862)
- Harry Johnson (multiple editions)
- Hon. Wm. Boothby (1908)
- And many others spanning 1827-1939

## Future Scripts

Additional scripts will be added to complete the RAG pipeline, including:
- Query interface for semantic search over the embedded chunks
- RAG application for question-answering over historical cocktail recipes

## Notes

- The scripts are idempotent: re-running them will skip already-processed files or safely update existing data
- OCR processing can be slow for large PDFs; processed OCR files are cached in `data_raw/ocr/`
- Text normalization handles common OCR issues like ligatures and page number artifacts
- Chunking uses sentence boundaries and section detection to maintain context
- Embedding uses local models (no API keys required); models are downloaded from Hugging Face on first use
- Qdrant data persists in Docker volumes; stop/start the container without losing your indexed data

