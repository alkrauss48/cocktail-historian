# scripts/04_search_qdrant.py

"""
Step 4 of the RAG pipeline: Query Qdrant and inspect retrieved chunks.

This script:
- Loads the same local embedding model used in 03_embed_qdrant.py
- Connects to Qdrant
- Lets you type queries in a simple REPL
- Embeds the query and searches Qdrant
- Prints top-k results with metadata and text snippets

Dependencies:
    pip install qdrant-client sentence-transformers
"""

import argparse
import sys
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest
from sentence_transformers import SentenceTransformer

# Defaults (keep consistent with 03_embed_qdrant.py)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DEFAULT_COLLECTION_NAME = "cocktail_chunks"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_TOP_K = 8


def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"[info] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("[info] Model loaded")
    return model


def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
    return vec.tolist()


def connect_qdrant(host: str, port: int) -> QdrantClient:
    print(f"[info] Connecting to Qdrant at {host}:{port}")
    client = QdrantClient(host=host, port=port)
    # Test connection
    client.get_collections()
    print("[info] Connected to Qdrant")
    return client


def search_once(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    top_k: int,
):
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return results


def format_result(result, idx: int):
    payload = result.payload or {}
    score = result.score

    book_title = payload.get("book_title", "<unknown book>")
    section_title = payload.get("section_title", "<no section title>")
    chunk_text = payload.get("chunk_text", "")
    chunk_index = payload.get("chunk_index")

    # Create preview with full chunk text
    text_preview = ""
    if chunk_text:
        text_preview = chunk_text.replace("\n", " ").replace("\t", " ")

    return f"""
Result #{idx + 1}
  Score        : {score:.4f}
  Book         : {book_title}
  Section      : {section_title}
  Chunk index  : {chunk_index}

  Text preview:
    {text_preview}
"""

def main():
    parser = argparse.ArgumentParser(
        description="Interactive search over Qdrant cocktail collection"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Local embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to show per query (default: {DEFAULT_TOP_K})",
    )

    args = parser.parse_args()

    # Connect
    try:
        client = connect_qdrant(QDRANT_HOST, QDRANT_PORT)
    except Exception as e:
        print(f"[error] Could not connect to Qdrant: {e}")
        sys.exit(1)

    # Load model
    try:
        model = load_embedding_model(args.embedding_model)
    except Exception as e:
        print(f"[error] Could not load embedding model '{args.embedding_model}': {e}")
        sys.exit(1)

    print("\n[ready] Type a query and press Enter to search.")
    print("        Empty line or Ctrl+C to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[info] Exiting.")
            break

        if not query:
            print("[info] Empty query, exiting.")
            break

        qvec = embed_query(model, query)
        try:
            results = search_once(
                client=client,
                collection=args.collection_name,
                query_vector=qvec,
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"[error] Search failed: {e}")
            continue

        if not results:
            print("No results.")
            continue

        print(f"\nTop {len(results)} results:\n")
        for i, r in enumerate(results):
            print(format_result(r, i))
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
