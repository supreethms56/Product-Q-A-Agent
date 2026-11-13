# Brightmere Energy Product Q&A Agent

A command-line Retrieval-Augmented Generation (RAG) agent that answers questions strictly from `product_catalogue.md` using local models hosted in **LM Studio**.

---

## Pipeline Architecture

```text
product_catalogue.md
        │
        ▼  1. Clean PDF artifacts (comments, image tags) & split by ### product headers
[ 14 Product Chunks ]  (with parent ## category & shared category bullets attached)
        │
        ▼  2. Embed once via nomic-embed-text-v1.5 & cache to .embeddings_cache.json
[ Vector Store + BM25 Index ]
        │
        ▼  3. Hybrid Search (0.65 Cosine Similarity + 0.35 BM25) → Top-3 Chunks
[ Retrieved Context ]
        │
        ▼  4. Grounded System Prompt (temperature=0.0, March 2026 doc date)
[ LFM2.5-1.2B-Instruct ]
        │
        ▼
  Grounded Answer + Cited Product Sources
```

---

## Setup Instructions

1. **Start LM Studio Local Server**
   - Download `nomic-embed-text-v1.5` (embedding model) and `LFM2.5-1.2B-Instruct` (chat model) in LM Studio.
   - Start the local server on `http://localhost:1234` with both models loaded.
   - Verify the model IDs in `SETTINGS` at the top of `qa_agent.py` match your LM Studio identifiers.

2. **Install Python Dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## How to Run

- **Interactive Q&A Loop (`exit` or `quit` to stop):**
  ```bash
  python qa_agent.py
  ```

- **Run the 7 Evaluation Queries (`eval_output.txt`):**
  ```bash
  python qa_agent.py --eval > eval_output.txt
  ```

- **Inspect Parsed Chunks:**
  ```bash
  python qa_agent.py --chunks
  ```

- **Enable Debug Similarity Scores:**
  ```bash
  python qa_agent.py --debug
  python qa_agent.py --eval --debug
  ```

---

## Chunking Strategy & Rationale

- **Boundary:** Product-level splitting on `### ` Markdown headings (`14` chunks for `14` products). Each product in the catalogue is a concise self-contained spec sheet (5–8 bullets), so splitting mid-product would separate a product's name from its features.
- **Category Inheritance:** Under `## Permanent Plant Monitoring`, five shared bullets (including *97% connectivity inside metal cabinets* and *magnetic mounting with no drilling*) appear **above** `### GRIDWATCH AI` and `### GRIDWATCH PULSE`. During parsing, category headers and pre-product shared bullets are inherited into every child product chunk in that section. Without this, Query 5 (*"inside steel electrical cabinets with no drilling"*) cannot be answered from the product chunks alone.
- **Pre-chunk Cleaning:** Stripped PDF-to-Markdown artifacts (`<!-- page:X -->`, `<!-- image_id:... -->`, `![img_xxx](...)`) so image filenames and HTML comments do not pollute token counts, BM25 term frequencies, or embedding vectors.
