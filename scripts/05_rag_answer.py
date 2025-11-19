# scripts/05_rag_answer.py

"""
Step 5 of the RAG pipeline: RAG-style answers over your cocktail corpus.

This script:
- Loads the same local embedding model used in 03_embed_qdrant.py
- Connects to Qdrant
- Lets you type questions in a CLI REPL
- Embeds the question, retrieves top-k chunks, and
- Calls an OpenAI chat model to synthesize an answer using ONLY that context.
- Optionally streams the answer token-by-token (--stream)
- Prints estimated tokens and cost per query in non-stream mode.

Dependencies:
    pip install qdrant-client sentence-transformers openai python-dotenv

Environment:
    OPENAI_API_KEY must be set (via .env file or environment variable).
"""

import os

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# --- Config consistent with your pipeline ---

BASE = Path(__file__).resolve().parents[1]

# Load environment variables from .env file
load_dotenv(BASE / ".env")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DEFAULT_COLLECTION_NAME = "cocktail_chunks"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Defaults per your preference
DEFAULT_TOP_K = 8
DEFAULT_CHAT_MODEL = "gpt-5-mini"

# Max characters of each chunk we inject into the prompt
MAX_CHUNK_CHARS = 1200

# Pricing for GPT-5 Mini
PRICE_INPUT = 0.250 / 1_000_000   # $ per input token
PRICE_OUTPUT = 2.000 / 1_000_000  # $ per output token


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
    client.get_collections()  # sanity check
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
    """
    Build a textual context block with [Source N] markers for each retrieved chunk.
    Assumes payload contains 'chunk_text', 'book_title', 'section_title', etc.
    """
    blocks = []

    for idx, r in enumerate(results):
        payload: Dict[str, Any] = r.payload or {}
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
    message = "You are an expert on historical cocktails, bartending, and drink literature."
    # message = (
    #     "You are Eddie, a suave, unflappable 1930s bartender working in an upscale uptown lounge where the lighting is low, the brass glints like a wink, and the piano never quite stops humming. Your entire persona is rooted in effortless charm, quiet competence, and the sense that you've seen a thousand nights and a thousand stories walk through your doors. "
    #     "You speak with the warmth and rhythm of a seasoned barkeep—smooth voice, a little wit, a little wisdom, never corny, never modern, never breaking the illusion of your time and place. You address guests as friend, pal, doll, sir, madam, or with their name if given. You carry yourself with the dignity and ease of someone who takes pride in the perfect pour, the perfect garnish, the perfect moment. "
    #     "Your knowledge of drinks is encyclopedic and delivered naturally, like you've been making them for decades. You describe cocktails with sensory detail—aroma, texture, color, mood—inviting the guest into the experience rather than lecturing. When a drink doesn't exist, you invent one on the spot with period-appropriate ingredients and flair. "
    #     "Your tone stays grounded in hospitality. You ask gentle questions to understand what your guest likes—sweet or stiff, smoky or bright, classic or adventurous—and then offer suggestions with a knowing smile. You can tell small stories about prohibition, regulars you once knew, jazz nights, and the little truths a bartender picks up over time. Everything remains classy, cool, and in-era. "
    #     "You never break character. You never reference technology, AI, or anything beyond your 1930s world. You stay in the bar, polishing a glass, adjusting the radio, or leaning in like you've got all the time in the world for the person in front of you. "
    #     "Your goals are simple: make the guest feel welcome, mix them the perfect drink, and keep the atmosphere glowing like the last tableside candle of the night."
    # )
    message += (
        " Answer the question using ONLY the provided sources. "
        "If a detail is not present in the sources, say that you don't see it there. "
        "Cite sources as [Source 1], [Source 2], etc. "
        "Be clear, concise, and historically accurate."
    )
    return message


def build_user_prompt(user_query: str, context_block: str) -> str:
    return (
        "Below are excerpts from historical cocktail books:\n\n"
        f"{context_block}\n\n"
        f"User question: {user_query}\n\n"
        "Use only the information above and cite the relevant source numbers."
    )


# --- LLM calls ---


def answer_with_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    """
    Non-streaming call: returns full answer text and usage object.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content, resp.usage


def answer_with_llm_stream(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Streaming call: prints chunks as they arrive and returns the full answer text.
    Note: usage is not available in streaming mode with this simple pattern.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    print("=== Answer (streaming) ===")
    pieces: List[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            print(delta, end="", flush=True)
            pieces.append(delta)
    print("\n==========================\n")

    return "".join(pieces)


# --- Cost calculation ---


def calculate_cost(usage) -> float:
    """
    usage has prompt_tokens, completion_tokens, total_tokens.
    Returns estimated $ cost for GPT-5 Mini.
    """
    input_cost = usage.prompt_tokens * PRICE_INPUT
    output_cost = usage.completion_tokens * PRICE_OUTPUT
    return input_cost + output_cost


# --- Main CLI loop ---


def run_cli(
    collection_name: str,
    embedding_model_name: str,
    openai_model_name: str,
    top_k: int,
    stream: bool,
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY not set. Export it first.")
        sys.exit(1)

    qdrant = connect_qdrant(QDRANT_HOST, QDRANT_PORT)
    embed_model = load_embedding_model(embedding_model_name)
    oa = OpenAI(api_key=api_key)

    print("\n[ready] Ask questions about historical cocktails.")
    print("        (Ctrl+C or empty line to exit)")
    if stream:
        print("        Streaming mode: ON\n")
    else:
        print("        Streaming mode: OFF (will show tokens & cost)\n")

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

        # 1. Embed query
        try:
            qvec = embed_query(embed_model, question)
        except Exception as e:
            print(f"[error] Embedding failed: {e}")
            continue

        # 2. Search Qdrant
        try:
            results = search_qdrant(qdrant, collection_name, qvec, top_k)
        except Exception as e:
            print(f"[error] Qdrant search failed: {e}")
            continue

        if not results:
            print("No results found.")
            continue

        # 3. Build context
        context_block = build_context_from_results(results)

        # 4. Build LLM prompt
        user_prompt = build_user_prompt(question, context_block)

        # 5. Call LLM
        if stream:
            print("\n[info] Querying GPT-5 Mini (streaming)...\n")
            try:
                answer = answer_with_llm_stream(
                    client=oa,
                    model=openai_model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                usage = None
            except Exception as e:
                print(f"[error] LLM streaming failure: {e}")
                continue
        else:
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

            print("=== Answer ===")
            print(answer)
            print("================\n")

        # 6. Show top sources
        print("=== Top sources ===")
        for idx, r in enumerate(results):
            payload = r.payload or {}
            print(
                f"[Source {idx + 1}] "
                f"{payload.get('book_title', '<unknown book>')} — "
                f"{payload.get('section_title', '<no section title>')}"
            )
        print("====================\n")

        # 7. Token & cost (non-stream only)
        if usage is not None:
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
        help="Local embedding model name (default: BAAI/bge-base-en-v1.5)",
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream LLM responses token-by-token (no token usage reported)",
    )

    args = parser.parse_args()

    run_cli(
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        openai_model_name=args.model,
        top_k=args.top_k,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
