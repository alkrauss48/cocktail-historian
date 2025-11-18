# scripts/05_rag_answer.py

"""
Step 5 of the RAG pipeline: RAG-style answers over your cocktail corpus.

This script:
- Loads the same local embedding model used in 03_embed_qdrant.py
- Connects to Qdrant
- Lets you type questions in a CLI REPL
- Embeds the question, retrieves top-k chunks, and
- Calls an OpenAI chat model to synthesize an answer using ONLY that context.
- Prints estimated tokens and cost per query (GPT-5 Mini pricing)

Dependencies:
    pip install qdrant-client sentence-transformers openai tqdm python-dotenv

Environment:
    OPENAI_API_KEY must be set (via .env file or environment variable).
"""

import argparse
import os
import sys
from textwrap import indent
from typing import List, Dict, Any

from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from openai import OpenAI

# --- Config consistent with your pipeline ---

BASE = Path(__file__).resolve().parents[1]

# Load environment variables from .env file
load_dotenv(BASE / ".env")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DEFAULT_COLLECTION_NAME = "cocktail_chunks"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Updated defaults per your request
DEFAULT_TOP_K = 8
DEFAULT_CHAT_MODEL = "gpt-5-mini"

MAX_CHUNK_CHARS = 1200


# Pricing for GPT-5 Mini
PRICE_INPUT = 0.250 / 1_000_000   # $0.00000025 per input token
PRICE_OUTPUT = 2.000 / 1_000_000  # $0.00000200 per output token


# --- Helpers: embeddings & Qdrant ---


def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"[info] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("[info] Embedding model loaded.")
    return model


def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    return model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0].tolist()


def connect_qdrant(host: str, port: int) -> QdrantClient:
    print(f"[info] Connecting to Qdrant at {host}:{port}")
    client = QdrantClient(host=host, port=port)
    client.get_collections()  # Test connection
    print("[info] Connected to Qdrant.")
    return client


def search_qdrant(client: QdrantClient, collection: str, query_vector: List[float], top_k: int):
    return client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )


# --- Prompt building ---


def build_context_from_results(results) -> str:
    blocks = []

    for idx, r in enumerate(results):
        payload = r.payload or {}
        book = payload.get("book_title", "<unknown book>")
        section = payload.get("section_title", "<no section title>")
        chunk_text = payload.get("chunk_text", "")
        chunk_idx = payload.get("chunk_index")

        cleaned = chunk_text.replace("\n", " ").replace("\t", " ")
        snippet = cleaned[:MAX_CHUNK_CHARS]
        if len(cleaned) > MAX_CHUNK_CHARS:
            snippet += "..."

        blocks.append(
            f"[Source {idx + 1}] Book: {book}\n"
            f"Section: {section} (chunk {chunk_idx})\n"
            f"Text: {snippet}\n"
        )

    return "\n".join(blocks)


def build_system_prompt() -> str:
    return (
        "You are an expert on historical cocktails, bartending, and drink literature. "
        "Answer the question using ONLY the provided sources. "
        "Cite sources as [Source 1], [Source 2], etc. "
        "If something is not in the sources, say so. "
        "Be clear, concise, and historically accurate."
    )


def build_user_prompt(user_query: str, context_block: str) -> str:
    return (
        "Below are excerpts from historical cocktail books:\n\n"
        f"{context_block}\n\n"
        f"User question: {user_query}\n\n"
        "Use only the information above and cite the relevant source numbers."
    )


# --- LLM call ---


def answer_with_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content, resp.usage


# --- Cost calculation ---


def calculate_cost(usage) -> float:
    """usage = {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}"""
    input_cost = usage.prompt_tokens * PRICE_INPUT
    output_cost = usage.completion_tokens * PRICE_OUTPUT
    return input_cost + output_cost


# --- Main CLI loop ---


def run_cli(collection_name, embedding_model_name, openai_model_name, top_k):

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY not set. Export it first.")
        sys.exit(1)

    qdrant = connect_qdrant(QDRANT_HOST, QDRANT_PORT)
    embed_model = load_embedding_model(embedding_model_name)
    oa = OpenAI(api_key=api_key)

    print("\n[ready] Ask questions about historical cocktails.")
    print("        (Ctrl+C or empty line to exit)\n")

    system_prompt = build_system_prompt()

    while True:
        try:
            question = input("Question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[info] Exiting.")
            break

        if not question:
            print("[info] Goodbye.")
            break

        # --- Step 1: Embed query ---
        try:
            qvec = embed_query(embed_model, question)
        except Exception as e:
            print(f"[error] Embedding failed: {e}")
            continue

        # --- Step 2: Search Qdrant ---
        try:
            results = search_qdrant(qdrant, collection_name, qvec, top_k)
        except Exception as e:
            print(f"[error] Qdrant search failed: {e}")
            continue

        if not results:
            print("No results found.")
            continue

        # --- Step 3: Build context block ---
        context_block = build_context_from_results(results)

        # --- Step 4: Build LLM prompt ---
        user_prompt = build_user_prompt(question, context_block)

        # --- Step 5: LLM response ---
        print("\n[info] Querying GPT-5 Mini...\n")

        try:
            answer, usage = answer_with_llm(
                client=oa,
                model=openai_model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"[error] LLM failure: {e}")
            continue

        # --- Output answer ---
        print("=== Answer ===")
        print(answer)
        print("================\n")

        # --- Print source list ---
        print("=== Top sources ===")
        for idx, r in enumerate(results):
            payload = r.payload or {}
            print(f"[Source {idx + 1}] {payload.get('book_title')} — {payload.get('section_title')}")
        print("====================\n")

        # --- Token + cost reporting ---
        print("=== Token Usage ===")
        print(f"Input tokens:      {usage.prompt_tokens}")
        print(f"Output tokens:     {usage.completion_tokens}")
        print(f"Total tokens:      {usage.total_tokens}")
        cost = calculate_cost(usage)
        print(f"Estimated cost:    ${cost:.6f} per query (GPT-5 Mini)")
        print("====================\n")


def main():
    parser = argparse.ArgumentParser(description="RAG QA over your cocktail corpus.")
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
        help="Local embedding model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: 8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CHAT_MODEL,
        help="OpenAI chat model (default: gpt-5-mini)",
    )

    args = parser.parse_args()

    run_cli(
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        openai_model_name=args.model,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
