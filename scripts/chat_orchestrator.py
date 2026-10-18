"""Hybrid chat orchestrator: retrieval (FAISS) -> context -> local model.

Pipeline:
  1. embed the user query (same MiniLM model as the knowledge base)
  2. top-k search over data/knowledge_base/index.faiss
  3. assemble a context block from the found chunks
  4. hand the prompt to our local MiniGPT via POST /api/chat
     (stub: in the future this is where an LLM / better generator plugs in)

Usage:
    py -3 scripts/chat_orchestrator.py "что в текстах про ..."
    py -3 scripts/chat_orchestrator.py            # interactive REPL
    py -3 scripts/chat_orchestrator.py --no-model  # retrieval only (debug)
"""

import argparse
import json
import os
import sys
import urllib.request

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "knowledge_base")
INDEX_FILE = os.path.join(OUT_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.json")
META_FILE = os.path.join(OUT_DIR, "meta.json")

CHAT_URL = "http://127.0.0.1:8000/api/chat"
DEFAULT_TOP_K = 4
# our MiniGPT has a short context window, so only the best chunk goes to it
MODEL_CONTEXT_CHARS = 200


class KnowledgeBase:
    def __init__(self):
        if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
            raise SystemExit(
                "Knowledge base not built. Run:\n"
                "  1) put .txt/.md files into data/knowledge/\n"
                "  2) py -3 scripts/build_knowledge_base.py")
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.index = faiss.read_index(INDEX_FILE)
        meta = {}
        if os.path.exists(META_FILE):
            with open(META_FILE, encoding="utf-8") as f:
                meta = json.load(f)
        self.model_name = meta.get("embed_model")
        self._st = None
        print(f"knowledge base: {self.index.ntotal} chunks loaded")

    @property
    def st(self):
        # lazy-load the embedding model (only needed at query time)
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            name = self.model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(f"loading embedder {name} ...")
            self._st = SentenceTransformer(name)
        return self._st

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Return top-k chunks with cosine scores for the query."""
        q = self.st.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(top_k, self.index.ntotal))
        hits = []
        for score, cid in zip(scores[0], ids[0]):
            if cid < 0:
                continue
            c = self.chunks[int(cid)]
            hits.append({"score": float(score), "id": int(cid),
                         "source": c["source"], "text": c["text"]})
        return hits


def build_prompt(query: str, hits: list[dict]) -> str:
    """Assemble the prompt: best chunk as context + the question.

    NOTE: MiniGPT's window is short (block_size tokens), so we pass only a
    trimmed slice of the best hit. When a real LLM is plugged in later,
    raise MODEL_CONTEXT_CHARS and include all hits.
    """
    if not hits:
        return query
    best = hits[0]["text"][:MODEL_CONTEXT_CHARS].strip()
    return (f"Из текста: {best}\n\nВопрос: {query}\nОтвет:")


def ask_model(prompt: str) -> str:
    """Send the prompt to the local chat server (MiniGPT). Stub point for a future LLM."""
    body = json.dumps({"message": prompt}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        CHAT_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("reply", "(пустой ответ)")
    except Exception as e:  # noqa: BLE001
        return f"[модель недоступна: {e}] запустите: py -3 scripts/chat_server.py"


def answer(query: str, kb: KnowledgeBase, top_k: int, use_model: bool) -> str:
    hits = kb.search(query, top_k)
    lines = [f"Запрос: {query}",
             f"Найдено {len(hits)} фрагментов:"]
    for i, h in enumerate(hits, 1):
        snippet = h["text"][:160].replace("\n", " ")
        lines.append(f"  {i}. [{h['source']}] score={h['score']:.3f}  {snippet}…")
    prompt = build_prompt(query, hits)
    lines.append("\nПромпт для модели:")
    lines.append("  " + prompt.replace("\n", "\n  "))
    if use_model:
        lines.append("\nОтвет модели:")
        lines.append("  " + ask_model(prompt))
    else:
        lines.append("\n(--no-model: генерация пропущена)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid RAG orchestrator")
    ap.add_argument("query", nargs="?", help="query (omit for interactive mode)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--no-model", action="store_true", help="retrieval only")
    args = ap.parse_args()

    kb = KnowledgeBase()
    if args.query:
        print(answer(args.query, kb, args.top_k, not args.no_model))
        return
    print("Hybrid chat (REPL). 'exit' to quit.")
    while True:
        try:
            q = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "выход"):
            break
        print("\n" + answer(q, kb, args.top_k, not args.no_model))


if __name__ == "__main__":
    main()
