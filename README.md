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
│   ├── 03_embed_qdrant.py  # Embed chunks and upsert into Qdrant
│   ├── 04_search_qdrant.py # Interactive semantic search interface
│   └── 05_rag_answer.py   # RAG question-answering with LLM
└── requirements.txt       # Python dependencies
```

## Prerequisites

- Python 3.x
- [OCRmyPDF](https://ocrmypdf.readthedocs.io/) (for OCR processing when needed)
  - Install via: `pip install ocrmypdf` or `brew install ocrmypdf` (macOS)
- Docker and Docker Compose (for running Qdrant vector database)
  - Install via: [Docker Desktop](https://www.docker.com/products/docker-desktop/) or your system's package manager
- OpenAI API key (for Step 5 RAG question-answering)
  - Set via environment variable `OPENAI_API_KEY` or `.env` file

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
- Normalizes text (removes ligatures, control characters, Unicode whitespace, page numbers, extra whitespace)
- Saves cleaned text to `data_derived/text/` with metadata headers (MD5 hash, filename)
- Automatically re-extracts if source PDF changes (detected via MD5 comparison)

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

### Step 4: Interactive Semantic Search

```bash
python scripts/04_search_qdrant.py
```

This script:
- Loads the same local embedding model used in Step 3 (default: `BAAI/bge-base-en-v1.5`)
- Connects to Qdrant at `localhost:6333`
- Provides an interactive REPL (Read-Eval-Print Loop) for querying
- Embeds each query using the local model
- Performs semantic search over the indexed chunks
- Displays top-k results with similarity scores, metadata, and text previews

**Output**: Interactive search results displayed in the terminal

**Configuration options**:
- `--collection-name`: Qdrant collection name (default: `cocktail_chunks`)
- `--embedding-model`: Local embedding model name (default: `BAAI/bge-base-en-v1.5`)
- `--top-k`: Number of results to return per query (default: 5)

**Example usage**:
```bash
# Search with more results
python scripts/04_search_qdrant.py --top-k 10

# Use a different embedding model (must match the model used in Step 3)
python scripts/04_search_qdrant.py --embedding-model BAAI/bge-small-en-v1.5

# Search a different collection
python scripts/04_search_qdrant.py --collection-name my_cocktails
```

**Usage tips**:
- Type your query and press Enter to search
- Press Enter on an empty line or Ctrl+C to exit
- Results show similarity scores (higher is better), book titles, section titles, chunk indices, and text previews
- The script uses the same embedding model as Step 3 to ensure query embeddings match the indexed chunk embeddings

**Note**: Make sure Qdrant is running (`docker-compose up -d`) and that you've completed Steps 1-3 before using this script.

### Step 5: RAG Question-Answering

```bash
python scripts/05_rag_answer.py
```

This script:
- Loads the same local embedding model used in Step 3 (default: `BAAI/bge-base-en-v1.5`)
- Connects to Qdrant at `localhost:6333`
- Provides an interactive REPL (Read-Eval-Print Loop) for asking questions
- Embeds each question using the local model
- Retrieves top-k relevant chunks from Qdrant
- Uses an OpenAI chat model (default: `gpt-5-mini`) to synthesize answers using ONLY the retrieved context
- Optionally streams responses token-by-token for real-time feedback
- Displays source citations and token usage/cost information (in non-stream mode)

**Output**: Interactive Q&A results with synthesized answers and source citations

**Configuration options**:
- `--collection-name`: Qdrant collection name (default: `cocktail_chunks`)
- `--embedding-model`: Local embedding model name (default: `BAAI/bge-base-en-v1.5`)
- `--top-k`: Number of chunks to retrieve per query (default: 6)
- `--model`: OpenAI chat model name (default: `gpt-5-mini`)
- `--stream`: Stream LLM responses token-by-token (no token usage reported in stream mode)

**Example usage**:
```bash
# Use streaming mode for real-time responses
python scripts/05_rag_answer.py --stream

# Retrieve more context chunks
python scripts/05_rag_answer.py --top-k 10

# Use a different OpenAI model
python scripts/05_rag_answer.py --model gpt-4o-mini

# Use a different embedding model (must match the model used in Step 3)
python scripts/05_rag_answer.py --embedding-model BAAI/bge-small-en-v1.5

# Combine options: streaming with more chunks
python scripts/05_rag_answer.py --stream --top-k 8
```

**Usage tips**:
- Type your question and press Enter to get an answer
- Press Enter on an empty line or Ctrl+C to exit
- Answers are synthesized from retrieved chunks and include source citations as `[Source 1]`, `[Source 2]`, etc.
- In non-stream mode, the script displays token usage and estimated cost per query
- Streaming mode provides immediate feedback but doesn't show token usage statistics
- The script uses the same embedding model as Step 3 to ensure query embeddings match the indexed chunk embeddings

**Note**: Make sure Qdrant is running (`docker-compose up -d`), that you've completed Steps 1-3, and that `OPENAI_API_KEY` is set (via environment variable or `.env` file) before using this script.

## Data Sources

The project currently includes cocktail books from the late 1800s and early 1900s, including works by:
- Jerry Thomas (1862)
- Harry Johnson (multiple editions)
- Hon. Wm. Boothby (1908)
- And many others spanning 1827-1939

## Future Scripts

Additional scripts may be added to enhance the RAG pipeline, including:
- Advanced filtering and metadata-based search capabilities
- Batch question-answering from files
- Evaluation metrics and benchmarking tools

## Notes

- The scripts are idempotent: re-running them will skip already-processed files or safely update existing data. Step 1 will automatically re-extract if a source PDF's MD5 hash changes (indicating the PDF was updated).
- OCR processing can be slow for large PDFs; processed OCR files are cached in `data_raw/ocr/`
- Text normalization handles common OCR issues like ligatures and page number artifacts
- Chunking uses sentence boundaries and section detection to maintain context
- Embedding uses local models (no API keys required); models are downloaded from Hugging Face on first use
- Qdrant data persists in Docker volumes; stop/start the container without losing your indexed data


## Sources

You may use whatever PDFs you want; just put them in `/data_raw/original`. I chose to use the following 28 influential cocktail books ranging from the mid 1800s to the mid 1900s, all of which are freely and legally available at https://archive.org/.

A zip file of the following sources can be found at: https://personal-k8s-main-space.nyc3.cdn.digitaloceanspaces.com/cocktail-historian/pdfs/cocktail-historian-books.zip

- 1827 – Oxford Night Caps by Richard Cook
- 1862 – The Bar-Tender’s Guide or How to Mix Drinks by Jerry Thomas
- 1872 – Cooling Cups & Dainty Drinks by William Terrington
- 1882 – Harry Johnson’s Bartenders’ Manual by Harry Johnson
- 1888 – Harry Johnson’s Bartenders’ Manual by Harry Johnson
- 1892 – The Flowing Bowl by The Only William
- 1895 – The Mixicologist by C. F. Lawlor
- 1896 – Bariana by Louis Fouquet
- 1900 – Modern American Drinks by J. Kappeler
- 1900 – Harry Johnson’s Bartenders’ Manual by Harry Johnson
- 1903 – Daly's Bartenders' Encyclopedia
- 1904 – Stuart's Fancy Drinks and How To Mix Them
- 1906 – Louis’ Mixed Drinks by Muckensturm
- 1908 – The World's Drinks and How to Mix Them by Hon. Wm. Boothby
- 1912 – Hoffman House Bartender’s Guide by Charles Mahoney
- 1914 – Drinks by Jacques Straub
- 1917 – Recipes for Mixed Drinks by Hugo R. Ensslin (2nd ed.)
- 1924 – Manual del Cantinero by León Pujol & Oscar Muñiz
- 1928 – When It’s Cocktail Time in Cuba by Basil Woon
- 1931 – Old Waldorf Bar Days by Albert Stevens Crockett
- 1934 – Cocktails Bar La Florida by Constante Ribalaigua Vert
- 1935 – Old Waldorf-Astoria Bar Book
- 1936 – 1000 Misture by Elvezio Grassi
- 1936 – The Artistry of Mixing Drinks by Frank Meier
- 1937 – U.K.B.G. Approved Cocktails (United Kingdom Bartenders’ Guild)
- 1937 – Café Royal Cocktail Book by William J. Tarling
- 1938 – Famous Orleans Drinks and How to Mix ’Em by Stanley Clisby Arthur
- 1939 – The Gentleman’s Companion, Volume II: An Exotic Drinking Book by Charles H. Baker Jr.
