"""Build a local FAISS knowledge base from text files.

Input : data/knowledge/*.txt, *.md   — drop your texts here
Output: data/knowledge_base/
          index.faiss   — FAISS vector index (cosine / Inner Product)
          chunks.json   — [{"id", "text", "source", "start_chunk"}]
          meta.json     — build statistics

Usage:
    py -3 scripts/build_knowledge_base.py [--chunk 800 --overlap 150 --top-k 4]

The embedding model is multilingual MiniLM (384-dim, works for Russian);
it is downloaded from HuggingFace on first run and cached locally.
"""

import argparse
import glob
import json
import os
import re
import time

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOW_DIR = os.path.join(ROOT, "data", "knowledge")
OUT_DIR = os.path.join(ROOT, "data", "knowledge_base")
INDEX_FILE = os.path.join(OUT_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.json")
META_FILE = os.path.join(OUT_DIR, "meta.json")

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

RE_PARA = re.compile(r"\n\s*\n")


def iter_texts() -> list[tuple[str, str]]:
    """Yield (source_name, text) for every .txt/.md in data/knowledge/."""
    files = sorted(glob.glob(os.path.join(KNOW_DIR, "*.txt")) +
                   glob.glob(os.path.join(KNOW_DIR, "*.md")))
    if not files:
        raise SystemExit(
            f"No .txt/.md files in {KNOW_DIR} — drop your texts there first.")
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield os.path.basename(path), f.read()


def chunk_text(text: str, source: str, max_chars: int, overlap: int) -> list[dict]:
    """Paragraph-aware chunking with tail overlap between chunks."""
    paras = [p.strip() for p in RE_PARA.split(text) if p.strip()]
    units: list[str] = []
    for para in paras:
        if len(para) <= max_chars:
            units.append(para)
            continue
        # split oversized paragraph at sentence boundaries
        piece = ""
        for sent in re.split(r"(?<=[.!?…])\s+", para):
            if piece and len(piece) + len(sent) + 1 > max_chars:
                units.append(piece)
                piece = sent
            else:
                piece = f"{piece} {sent}".strip() if piece else sent
        if piece:
            units.append(piece)

    chunks: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    for unit in units:
        if buf and buf_len + len(unit) + 2 > max_chars:
            text_c = "\n\n".join(buf).strip()
            chunks.append({"text": text_c, "source": source})
            tail = text_c[-overlap:] if overlap and len(text_c) > overlap else ""
            buf = [tail] if tail else []
            buf_len = len(tail)
        buf.append(unit)
        buf_len += len(unit) + 2
    if buf:
        chunks.append({"text": "\n\n".join(buf).strip(), "source": source})
    return [c for c in chunks if len(c["text"]) >= 40]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS knowledge base")
    ap.add_argument("--chunk", type=int, default=800, help="max chars per chunk")
    ap.add_argument("--overlap", type=int, default=150, help="tail overlap")
    ap.add_argument("--model", type=str, default=EMBED_MODEL)
    args = ap.parse_args()

    t0 = time.time()
    sources = list(iter_texts())
    print(f"found {len(sources)} file(s): " +
          ", ".join(s for s, _ in sources), flush=True)

    all_chunks: list[dict] = []
    for source, text in sources:
        parts = chunk_text(text, source, args.chunk, args.overlap)
        print(f"  {source}: {len(text):,} chars -> {len(parts)} chunks",
              flush=True)
        all_chunks.extend(parts)
    if not all_chunks:
        raise SystemExit("No chunks produced — check input files.")

    print(f"loading embedding model {args.model} ...", flush=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.model)
    print("encoding chunks ...", flush=True)
    emb = st.encode([c["text"] for c in all_chunks],
                    batch_size=64, show_progress_bar=True,
                    convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)  # cosine similarity via inner product

    index = faiss.IndexFlatIP(EMBED_DIM)
    assert emb.shape[1] == EMBED_DIM, f"dim mismatch: {emb.shape[1]}"
    index.add(emb)

    for i, c in enumerate(all_chunks):
        c["id"] = i
    os.makedirs(OUT_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    meta = {"chunks": len(all_chunks), "files": [s for s, _ in sources],
            "embed_model": args.model, "dim": EMBED_DIM,
            "chunk_chars": args.chunk, "overlap": args.overlap,
            "built_at": int(time.time()), "seconds": round(time.time() - t0, 1)}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"done: {len(all_chunks)} chunks -> {INDEX_FILE} "
          f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

