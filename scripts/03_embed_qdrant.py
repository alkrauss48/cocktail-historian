# scripts/03_embed_qdrant.py

"""
Step 3 of the RAG preprocessing pipeline: Embed chunks locally and upsert into Qdrant.

This script:
- Reads chunked JSONL files from data_derived/chunks/
- Embeds each chunk's text using a local sentence-transformers model
- Upserts embeddings and metadata into a Qdrant collection

Dependencies:
    pip install qdrant-client sentence-transformers tqdm

No API keys are required. All embeddings are computed locally (CPU or GPU).
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:
    print("Error: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)

# Local embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)


# ----------------------
# Configuration constants
# ----------------------

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_DIR = BASE / "data_derived" / "chunks"
DEFAULT_COLLECTION_NAME = "cocktail_chunks"

# Good default local embedding model
# You can change this to:
#   "BAAI/bge-small-en-v1.5"  (384 dims, faster)
#   "all-MiniLM-L6-v2"        (384 dims, very light)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE = 64
DEFAULT_EMBEDDING_DIM = 768  # bge-base-en-v1.5 outputs 768-dim vectors

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------
# Data loading
# ----------------------

def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all chunks from JSONL files in the chunks directory.

    Each line in each *.jsonl file should be a JSON object with at least:
        - id   (string)
        - text (string)
    Other fields (book_id, section_title, etc.) are treated as metadata.

    Args:
        chunks_dir: Directory containing *.jsonl files

    Returns:
        List of chunk dictionaries.
    """
    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {chunks_dir}")
        return []

    chunks: List[Dict[str, Any]] = []
    skipped = 0

    for jsonl_file in jsonl_files:
        logger.info(f"Loading chunks from {jsonl_file.name}")
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON in {jsonl_file.name}:{line_num}: {e}"
                    )
                    skipped += 1
                    continue

                if not chunk.get("id") or not chunk.get("text"):
                    logger.debug(
                        f"Skipping chunk in {jsonl_file.name}:{line_num} "
                        f"(missing id or text)"
                    )
                    skipped += 1
                    continue

                chunks.append(chunk)

    logger.info(
        f"Loaded {len(chunks)} chunks from {len(jsonl_files)} files "
        f"(skipped {skipped} invalid/missing records)"
    )
    return chunks


# ----------------------
# Local embeddings
# ----------------------

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load a local sentence-transformers embedding model.

    Args:
        model_name: Hugging Face / sentence-transformers model name

    Returns:
        Loaded SentenceTransformer model instance.
    """
    logger.info(f"Loading local embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded successfully")
    return model


def embed_batch_local(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    """
    Embed a batch of texts locally using a sentence-transformers model.

    Args:
        model: Loaded SentenceTransformer model
        texts: List of texts to embed
        batch_size: Batch size for model.encode()

    Returns:
        List of embedding vectors (list of floats)
    """
    # sentence-transformers returns numpy array; convert to plain Python lists
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


# ----------------------
# Qdrant helpers
# ----------------------

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Ensure the Qdrant collection exists, creating it if necessary.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vector_size: Dimension of the embedding vectors
    """
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
            # Optionally verify vector size
            collection_info = client.get_collection(collection_name)
            existing_size = collection_info.config.params.vectors.size
            if existing_size != vector_size:
                logger.warning(
                    f"Collection '{collection_name}' exists with vector size "
                    f"{existing_size}, but current config expects {vector_size}. "
                    f"Consider recreating the collection if this is incorrect."
                )
            return

        logger.info(
            f"Creating collection '{collection_name}' with vector size {vector_size}"
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Collection '{collection_name}' created successfully")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise


def upsert_batch(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
) -> bool:
    """
    Upsert a batch of points into Qdrant.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        points: List of PointStruct objects to upsert

    Returns:
        True if successful, False otherwise
    """
    try:
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        return True
    except Exception as e:
        logger.error(f"Error upserting batch: {e}")
        return False


def convert_chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Convert a chunk ID string to a deterministic integer for Qdrant point ID.

    Uses MD5 hash of the chunk ID to generate a stable 31-bit positive integer.

    Args:
        chunk_id: String chunk ID

    Returns:
        Integer point ID
    """
    hash_bytes = hashlib.md5(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big") & 0x7FFFFFFF


# ----------------------
# Main
# ----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks locally and upsert into Qdrant"
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help=f"Directory containing JSONL chunk files (default: {DEFAULT_CHUNKS_DIR})",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embeddings and upserts (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Local embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})",
    )

    args = parser.parse_args()

    # Validate chunks directory
    if not args.chunks_dir.exists():
        logger.error(f"Chunks directory does not exist: {args.chunks_dir}")
        sys.exit(1)

    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qdrant_client.get_collections()  # test connection
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        logger.error(
            f"Make sure Qdrant is running at {QDRANT_HOST}:{QDRANT_PORT}. "
            "For Docker, start it with: docker-compose up -d"
        )
        sys.exit(1)

    # Load local embedding model
    model = load_embedding_model(args.embedding_model)

    # Ensure collection exists
    ensure_collection(
        client=qdrant_client,
        collection_name=args.collection_name,
        vector_size=args.embedding_dim,
    )

    # Load chunks
    chunks = load_chunks(args.chunks_dir)
    if not chunks:
        logger.warning("No chunks to process. Exiting.")
        return

    total_chunks = len(chunks)
    total_upserted = 0
    total_failed = 0

    logger.info(
        f"Processing {total_chunks} chunks in batches of {args.batch_size} "
        f"using local model {args.embedding_model}"
    )

    # Process in batches with progress bar
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for batch_start in range(0, total_chunks, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]

            # Extract texts for embedding
            batch_texts = [chunk["text"] for chunk in batch_chunks]

            # Embed batch locally
            try:
                embeddings = embed_batch_local(
                    model=model,
                    texts=batch_texts,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                logger.error(f"Error embedding batch {batch_start}-{batch_end}: {e}")
                total_failed += len(batch_chunks)
                pbar.update(len(batch_chunks))
                continue

            # Prepare Qdrant points
            points: List[PointStruct] = []
            for chunk, embedding in zip(batch_chunks, embeddings):
                point_id = convert_chunk_id_to_point_id(chunk["id"])

                payload = {
                    "id": chunk["id"],
                    "book_id": chunk.get("book_id"),
                    "book_title": chunk.get("book_title"),
                    "year": chunk.get("year"),
                    "author": chunk.get("author"),
                    "section_index": chunk.get("section_index"),
                    "section_title": chunk.get("section_title"),
                    "chunk_index": chunk.get("chunk_index"),
                    "chunk_text": chunk.get("text"),
                }
                # Strip None values
                payload = {k: v for k, v in payload.items() if v is not None}

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upsert batch into Qdrant
            success = upsert_batch(
                client=qdrant_client,
                collection_name=args.collection_name,
                points=points,
            )

            if success:
                total_upserted += len(points)
            else:
                total_failed += len(points)

            pbar.update(len(batch_chunks))

    # Summary
    logger.info("=" * 60)
    logger.info("Embedding + Qdrant upsert complete!")
    logger.info(f"Total chunks processed: {total_chunks}")
    logger.info(f"Successfully upserted: {total_upserted}")
    logger.info(f"Failed/skipped: {total_failed}")
    logger.info(f"Collection: {args.collection_name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
