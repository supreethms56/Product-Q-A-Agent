import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

# ==============================================================================
# CONFIGURATION & SETTINGS (Single place as required)
# ==============================================================================
SETTINGS = {
    "LM_STUDIO_BASE_URL": os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    "LM_STUDIO_API_KEY": os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"),
    "CHAT_MODEL": os.getenv("CHAT_MODEL", "lfm2.5-1.2b-instruct"),
    "CATALOGUE_PATH": "product_catalogue.md",
    "CACHE_PATH": ".embeddings_cache.json",
    "TOP_K": 3,
    "HYBRID_ALPHA": 0.65,  # Weight for dense vector similarity vs BM25 keyword score
    "MIN_RELEVANCE_SCORE": 0.35,  # Threshold below which query is treated as out-of-scope
}

TEST_QUERIES = [
    "What is the Insulation Trend Memory feature and which product has it?",
    "What is the difference between the SOLVANTA 3K and the SOLVANTA 5K PRO?",
    "Which products use Dual-Sense MPPT (DSM), and what advantage does DSM give?",
    "We are installing panels on a large commercial rooftop with long cable runs. What do you recommend?",
    "We want permanent monitoring inside steel electrical cabinets with no drilling. What fits?",
    "Can I buy the GRIDWATCH PULSE today?",
    "Does the GRIDWATCH AI use clamp-on CT sensors?",
]

# Words damaged by the PDF-to-Markdown conversion (the "ff" ligature was lost),
# plus one typo. Found by reading the whole catalogue.
PDF_FIXES = {
    "eficiency": "efficiency",
    "ofering": "offering",
    "Of-Grid": "Off-Grid",
    "Ofline": "Offline",
    "diferences": "differences",
    "settings an batteries": "settings and batteries",
}

# Common words ignored by BM25, so they don't give unrelated chunks a keyword score.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "so", "that", "the", "their", "there", "this", "to",
    "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


# ==============================================================================
# 1. INGESTION & CHUNKING (R1, R2)
# ==============================================================================
def clean_markdown(raw_text: str) -> str:
    """Remove PDF-to-Markdown conversion noise while preserving structure."""
    # Remove HTML comments such as <!-- page:1 --> and <!-- image_id:... -->
    text = re.sub(r"<!--.*?-->", "", raw_text)
    # Remove markdown image tags such as ![img_001](../../assets/...)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Fix words damaged by the PDF conversion
    for wrong, right in PDF_FIXES.items():
        text = text.replace(wrong, right)
    # Normalize tabs and multiple spaces inside lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Use one bullet style ("• " and "•<tab>" both become "- ")
    lines = [re.sub(r"^•\s*", "- ", line) for line in lines]
    # Collapse 3+ blank lines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned


def parse_and_chunk_catalogue(filepath: str) -> list[dict[str, Any]]:
    """
    Parse product_catalogue.md into 14 product-level chunks.
    Strategy:
    - Split on '### ' headings (each represents one of the 14 products).
    - Track the active '## ' category heading AND any category-level intro/bullets
      (crucial for 'Permanent Plant Monitoring', whose shared specs sit above GRIDWATCH AI/PULSE).
    - Prepend category context to each product chunk so every chunk is self-contained.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Catalogue file not found: {filepath}")

    raw_text = path.read_text(encoding="utf-8")
    cleaned_text = clean_markdown(raw_text)

    chunks: list[dict[str, Any]] = []
    current_category = "General"
    category_shared_lines: list[str] = []
    current_product: str | None = None
    product_lines: list[str] = []

    def flush_product() -> None:
        nonlocal current_product, product_lines
        if not current_product:
            return
        body = "\n".join(line for line in product_lines if line).strip()
        shared_context = "\n".join(line for line in category_shared_lines if line).strip()

        full_text_parts = [f"Category: {current_category}"]
        if shared_context:
            full_text_parts.append(f"Category Overview & Shared Features:\n{shared_context}")
        full_text_parts.append(f"Product: {current_product}\n{body}")

        chunks.append(
            {
                "id": len(chunks) + 1,
                "product_name": current_product,
                "category": current_category,
                "content": "\n\n".join(full_text_parts),
            }
        )
        current_product = None
        product_lines = []

    for line in cleaned_text.splitlines():
        if line.startswith("# "):
            # Document title header (# PRODUCT CATALOGUE...)
            continue
        elif line.startswith("## "):
            flush_product()
            current_category = line.lstrip("#").strip()
            category_shared_lines = []
        elif line.startswith("### "):
            flush_product()
            current_product = line.lstrip("#").strip()
            product_lines = []
        else:
            # Handle occasional un-prefixed subheaders in the raw catalogue (e.g., "Handheld Solar Test Tools")
            if line == "Handheld Solar Test Tools":
                flush_product()
                current_category = line
                category_shared_lines = []
                continue

            if current_product is None:
                if line:
                    category_shared_lines.append(line)
            else:
                if line:
                    product_lines.append(line)

    flush_product()
    return chunks


# ==============================================================================
# 2. EMBEDDINGS & LOCAL CACHE (R3)
# ==============================================================================
def get_client() -> OpenAI:
    return OpenAI(
        base_url=SETTINGS["LM_STUDIO_BASE_URL"],
        api_key=SETTINGS["LM_STUDIO_API_KEY"],
    )


def compute_catalogue_hash(chunks: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"model": SETTINGS["EMBEDDING_MODEL"], "chunks": chunks},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=SETTINGS["EMBEDDING_MODEL"],
        input=texts,
    )
    return [item.embedding for item in response.data]


def load_or_create_embeddings(
    client: OpenAI, chunks: list[dict[str, Any]]
) -> np.ndarray:
    """Load cached embeddings if catalogue & model match; otherwise compute and save."""
    cache_path = Path(SETTINGS["CACHE_PATH"])
    expected_hash = compute_catalogue_hash(chunks)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("hash") == expected_hash and "embeddings" in cached:
                return np.array(cached["embeddings"], dtype=np.float32)
        except (json.JSONDecodeError, KeyError):
            pass

    # Nomic models perform best with 'search_document: ' prefix on corpus chunks
    texts_to_embed = [f"search_document: {c['content']}" for c in chunks]
    vectors = embed_texts(client, texts_to_embed)

    cache_path.write_text(
        json.dumps({"hash": expected_hash, "embeddings": vectors}),
        encoding="utf-8",
    )
    return np.array(vectors, dtype=np.float32)


# ==============================================================================
# 3. HYBRID RETRIEVAL: COSINE SIMILARITY + BM25 (R4, R11)
# ==============================================================================
def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def compute_bm25_scores(query: str, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Lightweight BM25 implementation for exact keyword/product-code matching."""
    docs_tokens = [tokenize(c["content"]) for c in chunks]
    n_docs = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / max(n_docs, 1)

    # Document frequency
    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    q_tokens = tokenize(query)
    raw_scores = []
    for doc_tokens in docs_tokens:
        doc_len = len(doc_tokens)
        tf_map: dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for term in q_tokens:
            if term not in tf_map:
                continue
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            tf = tf_map[term]
            denom = tf + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1.0)))
            score += idf * (tf * (k1 + 1.0)) / denom
        raw_scores.append(score)

    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score > 0:
        return [s / max_score for s in raw_scores]
    return [0.0] * n_docs


def retrieve_chunks(
    client: OpenAI,
    query: str,
    chunks: list[dict[str, Any]],
    chunk_embeddings: np.ndarray,
    top_k: int = SETTINGS["TOP_K"],
) -> list[dict[str, Any]]:
    """Rank chunks using hybrid dense cosine similarity + normalized BM25."""
    # Nomic models perform best with 'search_query: ' prefix on queries
    q_vec = np.array(embed_texts(client, [f"search_query: {query}"])[0], dtype=np.float32)

    # Cosine similarity
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    q_norm = np.linalg.norm(q_vec)
    cosine_scores = (chunk_embeddings @ q_vec) / np.maximum(chunk_norms * q_norm, 1e-10)

    # BM25 keyword score
    bm25_scores = compute_bm25_scores(query, chunks)

    alpha = SETTINGS["HYBRID_ALPHA"]
    results = []
    for idx, chunk in enumerate(chunks):
        cos_sim = float(cosine_scores[idx])
        bm25_sim = float(bm25_scores[idx])
        hybrid_score = alpha * cos_sim + (1.0 - alpha) * bm25_sim
        results.append(
            {
                **chunk,
                "cosine_score": round(cos_sim, 4),
                "bm25_score": round(bm25_sim, 4),
                "score": round(hybrid_score, 4),
            }
        )

    # Products named in the question go first, so the model reads the asked-about
    # product before any similar one (e.g. GRIDWATCH AI before GRIDWATCH PULSE).
    named = find_named_products(query, chunks)
    results.sort(key=lambda x: (x["product_name"] in named, x["score"]), reverse=True)
    return results[:top_k]


def find_named_products(query: str, chunks: list[dict[str, Any]]) -> set[str]:
    """Return product names that appear word-for-word in the question."""
    q = query.lower()
    return {
        c["product_name"]
        for c in chunks
        if re.search(r"\b" + re.escape(c["product_name"].lower()) + r"\b", q)
    }


# ==============================================================================
# 4. GROUNDED ANSWER GENERATION (R5, R6, R7)
# ==============================================================================
NOT_FOUND_MESSAGE = (
    "Sorry, I don't have that information. Our product catalogue doesn't cover it."
)

SYSTEM_PROMPT = f"""You are the Brightmere Energy product assistant. You help customers and installers
choose and understand Brightmere solar products. Speak naturally, like a knowledgeable product
specialist talking to a customer.

You will receive product information from the Brightmere catalogue (dated March 2026).
It is your ONLY source of facts.

HOW TO ANSWER:
- Answer directly and naturally, in a friendly, professional tone. Use 1 to 4 short sentences,
  or a short list when comparing products.
- Never mention "context", "provided information", "sections", "excerpts" or "tags".
  Do not start with phrases like "Based on the information" or "According to the context".
  Just answer.
- Do not copy the [product] tags into your answer.

ACCURACY RULES (always follow these):
1. Only state facts given in the product information. Never add facts, numbers, comparisons or
   opinions that are not there.
2. Each line is tagged with its product, like [GRIDWATCH AI]. A fact belongs ONLY to that product.
   Never move a feature from one product to another.
3. Lines tagged [Shared: <category>] apply to every product in that category.
4. For a yes/no question about a product, check that product's own lines. If they say it does not
   need or have something, the answer is no.
5. If the information gives a launch date or availability, share it and mention that it comes from
   our March 2026 catalogue, so the customer should check with their dealer for current availability.
6. If the product information has nothing that answers the question, reply exactly:
   "{NOT_FOUND_MESSAGE}"
"""


def build_tagged_context(item: dict[str, Any]) -> str:
    """Tag every line with the product (or shared category) it belongs to,
    so a small model cannot mix up features between products."""
    product_tag = f"[{item['product_name']}]"
    shared_tag = f"[Shared: {item['category']}]"
    tag = product_tag
    lines = [f"--- PRODUCT SECTION: {item['product_name']} ---"]
    for line in item["content"].splitlines():
        if not line.strip():
            continue
        if line.startswith("Category Overview & Shared Features"):
            tag = shared_tag
            continue
        if line.startswith("Product: "):
            tag = product_tag
            continue
        if line.startswith("Category: "):
            lines.append(line)
            continue
        lines.append(f"{tag} {line}")
    return "\n".join(lines)

# Phrases a small model sometimes adds that make answers sound robotic.
ROBOTIC_OPENERS = re.compile(
    r"^(based on|according to|from|using|as per)\s+(the\s+)?"
    r"(provided\s+|given\s+|above\s+)?"
    r"(context|information|product information|catalogue information|excerpts?|sections?)"
    r"(\s+provided|\s+given|\s+above)?\s*[,:]\s*",
    re.IGNORECASE,
)


def clean_answer(text: str) -> str:
    """Remove robotic openers and leftover [product] tags, then tidy the answer."""
    answer = text.strip()
    answer = ROBOTIC_OPENERS.sub("", answer)
    answer = re.sub(r"\[(Shared: )?[A-Z0-9][A-Za-z0-9 &\-]*\]\s*", "", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
    if answer:
        answer = answer[0].upper() + answer[1:]
    return answer


def generate_grounded_answer(
    client: OpenAI, query: str, retrieved: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Generate a grounded response and return (answer_text, cited_product_names)."""
    if not retrieved or retrieved[0]["cosine_score"] < SETTINGS["MIN_RELEVANCE_SCORE"]:
        return NOT_FOUND_MESSAGE, []

    context_str = "\n\n".join(build_tagged_context(item) for item in retrieved)

    user_message = (
        f"Product information:\n{context_str}\n\n"
        f"Customer question: {query}"
    )

    response = client.chat.completions.create(
        model=SETTINGS["CHAT_MODEL"],
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    answer = (response.choices[0].message.content or "").strip()

    # Determine which retrieved sections were actually referenced or relevant
    if NOT_FOUND_MESSAGE.lower() in answer.lower():
        sources = []
    else:
        mentioned = [
            item["product_name"]
            for item in retrieved
            if item["product_name"].lower() in answer.lower()
        ]
        sources = mentioned if mentioned else [item["product_name"] for item in retrieved]

    return answer, sources


# ==============================================================================
# 5. CLI MODES: --chunks, --eval, INTERACTIVE LOOP (R8, R9, R10, R11)
# ==============================================================================
def print_chunks(chunks: list[dict[str, Any]]) -> None:
    print(f"Total chunks built: {len(chunks)}\n" + "=" * 60)
    for c in chunks:
        print(f"[Chunk {c['id']}] {c['product_name']} (Category: {c['category']})")
        print("-" * 60)
        print(c["content"])
        print("=" * 60 + "\n")


def run_single_query(
    client: OpenAI,
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    debug: bool = False,
    show_retrieved: bool = False,
) -> None:
    retrieved = retrieve_chunks(client, query, chunks, embeddings)

    if debug:
        print("[DEBUG] Top retrieved chunks:")
        for r in retrieved:
            print(
                f"  - {r['product_name']}: hybrid={r['score']:.4f} "
                f"(cosine={r['cosine_score']:.4f}, bm25={r['bm25_score']:.4f})"
            )

    if show_retrieved:
        retrieved_names = ", ".join(r["product_name"] for r in retrieved)
        print(f"Retrieved sections: {retrieved_names}")

    answer, sources = generate_grounded_answer(client, query, retrieved)
    print(f"Agent: {answer}")
    print(f"Sources: {', '.join(sources) if sources else 'None'}")


def run_eval(
    client: OpenAI, chunks: list[dict[str, Any]], embeddings: np.ndarray, debug: bool = False
) -> None:
    print("Running evaluation on 7 test queries...\n" + "=" * 60)
    for idx, query in enumerate(TEST_QUERIES, start=1):
        print(f"\nQuery {idx}: {query}")
        run_single_query(
            client,
            query,
            chunks,
            embeddings,
            debug=debug,
            show_retrieved=True,
        )
        print("-" * 60)


def run_interactive(
    client: OpenAI, chunks: list[dict[str, Any]], embeddings: np.ndarray, debug: bool = False
) -> None:
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        run_single_query(client, user_input, chunks, embeddings, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brightmere Energy Product Q&A Agent")
    parser.add_argument("--chunks", action="store_true", help="Print all parsed chunks and exit")
    parser.add_argument("--eval", action="store_true", help="Run the 7 evaluation queries")
    parser.add_argument("--debug", action="store_true", help="Print similarity scores for retrieved chunks")
    args = parser.parse_args()

    # Print °, ’ and other symbols correctly on Windows consoles and in redirected output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chunks = parse_and_chunk_catalogue(SETTINGS["CATALOGUE_PATH"])
    print(f"Loaded {len(chunks)} product sections from {SETTINGS['CATALOGUE_PATH']}")

    if args.chunks:
        print_chunks(chunks)
        return

    client = get_client()
    try:
        embeddings = load_or_create_embeddings(client, chunks)
    except Exception as exc:
        print(
            f"Error connecting to LM Studio at {SETTINGS['LM_STUDIO_BASE_URL']}: {exc}\n"
            "Ensure LM Studio's local server is running with both models loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.eval:
        run_eval(client, chunks, embeddings, debug=args.debug)
    else:
        run_interactive(client, chunks, embeddings, debug=args.debug)


if __name__ == "__main__":
    main()