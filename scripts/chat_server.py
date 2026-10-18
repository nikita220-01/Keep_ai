"""Keep AI — chat server: talk to the trained MiniGPT through the web UI.

Loads the checkpoint (data/checkpoints/keepai.pt), serves the static UI
from ui/ and exposes POST /api/chat which generates a reply.

Run:
    py -3 scripts/chat_server.py [--port 8000]
Then open http://localhost:8000 and type into the chat input field.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, MiniGPT, decode, load_checkpoint  # noqa: E402
from train_bpe import CKPT_PATH as BPE_CKPT_PATH  # noqa: E402

# prefer the BPE model when its checkpoint exists
CKPT_PATH = BPE_CKPT_PATH if os.path.exists(BPE_CKPT_PATH) else os.path.join(
    ROOT, "data", "checkpoints", "keepai.pt")

UI_DIR = os.path.join(ROOT, "ui")
MIME = {".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".txt": "text/plain; charset=utf-8",
        ".xml": "application/xml; charset=utf-8",
        ".ico": "image/x-icon"}

LOCK = threading.Lock()
STATE = {"model": None, "cfg": None, "vocab": None, "device": "cpu",
         "error": None, "info": {}}

RE_STOP = re.compile(r"\n\s*(Вопрос|Человек|User):|Вопрос:|\n- ")

RE_GREET = re.compile(
    r"^(привет|здравствуй(те)?|хай|ку|добрый (день|вечер)|доброе утро|салют)[!?.,\s]*$",
    re.I)


def ensure_model() -> None:
    """Load the checkpoint once; auto-reload when training rewrites it."""
    try:
        mtime = os.path.getmtime(CKPT_PATH)
    except OSError:
        mtime = None
    with LOCK:
        loaded = STATE["model"] is not None or STATE["error"] is not None
        if loaded and STATE.get("mtime") == mtime:
            return
        try:
            if mtime is None:
                raise FileNotFoundError(CKPT_PATH)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            t0 = time.time()
            # load explicitly from CKPT_PATH (may be the BPE checkpoint)
            ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
            fields = set(Config.__dataclass_fields__)
            cfg = Config(**{k: v for k, v in ck["config"].items() if k in fields})
            model = MiniGPT(cfg).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            STATE.update(model=model, cfg=cfg, vocab=ck.get("vocab"),
                         device=device, error=None, mtime=mtime,
                         info={"step": ck["step"], "best_val": ck["best_val"],
                               "type": ck.get("type", "char"),
                               "load_seconds": round(time.time() - t0, 1)})
            # tokenizers: BPE checkpoints use sentencepiece
            if ck.get("type") == "bpe":
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=ck["sp_model"])
                STATE["encode"] = lambda s: sp.encode(s, out_type=int)
                STATE["decode"] = lambda ids: sp.decode([int(i) for i in ids])
            else:
                stoi = ck["vocab"]
                STATE["encode"] = lambda s: [stoi.get(ch, 0) for ch in s]
                STATE["decode"] = lambda ids: decode(ids, stoi)
            print(f"[model] loaded in {time.time() - t0:.1f}s "
                  f"(step {ck['step']}, val {ck['best_val']:.3f})", flush=True)
        except Exception as e:  # noqa: BLE001
            STATE["model"] = None
            STATE["error"] = str(e)
            STATE["mtime"] = mtime
            print(f"[model] load failed: {e}", flush=True)


def build_prompt(message: str) -> str:
    """Dialogue framing: the corpus (incl. dialogues) uses '- reply' lines."""
    q = message.strip()
    if len(q) > 120:
        q = q[:120].rsplit(" ", 1)[0] + "…"
    if RE_GREET.match(q):
        return "- Здравствуйте! Как у тебя дела?\n- "
    return f"- {q}\n- "


@torch.no_grad()
def generate_reply(message: str) -> str:
    ensure_model()
    if STATE["error"]:
        return "⚠️ Не удалось загрузить модель: " + STATE["error"]
    model, cfg, device = STATE["model"], STATE["cfg"], STATE["device"]
    enc, dec = STATE["encode"], STATE["decode"]
    prompt = build_prompt(message)
    ids = enc(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out: list[int] = []
    text = ""
    temperature, top_k = 0.5, 24
    for _ in range(300):
        ctx = x[:, -cfg.block_size:]
        logits, _ = model(ctx)
        logits = logits[0, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        out.append(nxt)
        text = dec(out)
        if len(out) >= 10 and (RE_STOP.search(text)
                               or text.endswith("\n\n") or nxt == 3):
            break
        x = torch.cat((x, torch.tensor([[nxt]], device=device)), dim=1)
    # clean up
    text = RE_STOP.split(text)[0] if RE_STOP.search(text) else text
    text = text.replace("\n\n", " ").strip().strip("«»\"'")
    while "\n" in text:
        text = text.replace("\n", " ")
    if len(text) > 700:
        text = text[:700].rsplit(" ", 1)[0] + "…"
    return text or "(модель промолчала — попробуйте переформулировать)"

# ── HTTP server ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "KeepAI/1.0"

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            ensure_model()
            return self._json(200, {
                "model_loaded": STATE["model"] is not None,
                "error": STATE["error"], **STATE["info"]})
        if path == "/":
            path = "/index.html"
        fs = os.path.realpath(os.path.join(UI_DIR, path.lstrip("/")))
        if not fs.startswith(os.path.realpath(UI_DIR) + os.sep) or not os.path.isfile(fs):
            return self._json(404, {"error": "not found"})
        with open(fs, "rb") as f:
            data = f.read()
        ctype = MIME.get(os.path.splitext(fs)[1].lower(),
                         "application/octet-stream")
        self._send(200, data, ctype)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = str(payload.get("message") or "").strip()
            if not message:
                raise ValueError("empty message")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": f"bad request: {e}"})
        t0 = time.time()
        reply = generate_reply(message)
        print(f"[chat] q={message[:60]!r} -> {len(reply)} chars "
              f"in {time.time() - t0:.1f}s", flush=True)
        self._json(200, {"reply": reply})


def main() -> None:
    ap = argparse.ArgumentParser(description="Keep AI chat server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()
    print("loading model checkpoint ...", flush=True)
    ensure_model()
    if STATE["error"]:
        print(f"WARNING: model unavailable: {STATE['error']}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Keep AI ready: http://{args.host}:{args.port} "
          f"(UI + POST /api/chat)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

