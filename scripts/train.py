"""Keep AI — mini Transformer trained from scratch (nanoGPT-style).

Model: n_layer=2, n_head=2, n_embd=64, char-level GPT trained on the
prepared Chekhov corpus (data/clean/tokens_uint32.bin, vocab = 193 chars).

Usage:
    py -3 scripts/train.py                    # train from scratch
    py -3 scripts/train.py --resume           # continue from checkpoint
    py -3 scripts/train.py --steps 500        # quick smoke run
    py -3 scripts/train.py --sample --prompt "Вдруг"   # generate text
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS_PATH = os.path.join(ROOT, "data", "clean", "tokens_uint32.bin")
VOCAB_PATH = os.path.join(ROOT, "data", "clean", "vocab.json")
CKPT_DIR = os.path.join(ROOT, "data", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "keepai.pt")
SAMPLES_DIR = os.path.join(ROOT, "data", "samples")


@dataclass
class Config:
    vocab_size: int = 193
    block_size: int = 128   # context length in characters
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=400, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        self.train()
        return idx

# ── Data ─────────────────────────────────────────────────────────────────────

def load_vocab() -> dict[str, int]:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)  # {token: id}, ids 0..3 = <unk> <s> </s> <pad>


def decode(ids, itos: dict[str, int]) -> str:
    rev = {i: t for t, i in itos.items()}
    out = []
    for i in ids:
        t = rev.get(int(i), "<unk>")
        if t in ("<unk>", "<pad>", "<s>", "</s>"):
            continue
        out.append(t)
    return "".join(out)


def load_tokens():
    """Load the uint32 token stream as a numpy array."""
    import numpy as np
    data = np.fromfile(TOKENS_PATH, dtype=np.uint32)
    n_val = int(len(data) * 0.01)          # 1% validation split, in order
    return data[:-n_val], data[-n_val:]


def get_batch(data, batch_size: int, block_size: int, device: str):
    import numpy as np
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
                     for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# ── LR schedule & checkpoints ────────────────────────────────────────────────

def lr_for(step: int, max_lr: float, min_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # cosine decay
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(model, cfg, step, best_val, itos):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val": best_val,
        "vocab": itos,
    }, CKPT_PATH)


def load_checkpoint(device: str):
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    fields = set(Config.__dataclass_fields__)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in fields})
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg, ck

# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    itos = load_vocab()
    cfg = Config(vocab_size=len(itos))

    model = MiniGPT(cfg).to(device)
    print(f"device={device}  params={model.num_params():,}  "
          f"vocab={cfg.vocab_size}  block={cfg.block_size}  "
          f"layers={cfg.n_layer}  heads={cfg.n_head}  embd={cfg.n_embd}")

    start_step, best_val = 0, float("inf")
    if args.resume and os.path.exists(CKPT_PATH):
        model, cfg, ck = load_checkpoint(device)
        start_step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from {CKPT_PATH} at step {start_step} (best_val={best_val:.3f})")

    train_data, val_data = load_tokens()
    print(f"tokens: train={len(train_data):,}  val={len(val_data):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  betas=(0.9, 0.99), weight_decay=0.01)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    t0 = time.time()
    running = []
    model.train()

    for step in range(start_step, args.steps):
        lr = lr_for(step, args.max_lr, args.min_lr, args.warmup, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running.append(loss.item())
        if (step + 1) % args.log_every == 0:
            mean = sum(running) / len(running)
            sps = (step + 1 - start_step) / max(1e-9, time.time() - t0)
            print(f"step {step + 1:6d}/{args.steps} | loss {mean:.3f} | "
                  f"lr {lr:.2e} | {sps:.1f} it/s", flush=True)
            running = []

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            vl = estimate_loss(model, val_data, args.batch_size, cfg.block_size, device)
            print(f"  eval | val loss {vl:.3f} | ppl {math.exp(min(vl, 20)):,.1f}", flush=True)
            if vl < best_val:
                best_val = vl
                save_checkpoint(model, cfg, step + 1, best_val, itos)
                print(f"  checkpoint saved ({CKPT_PATH})", flush=True)

        if (step + 1) % args.sample_every == 0 and (step + 1) != args.steps:
            text = sample(model, itos, device, args.sample_prompt, 300)
            print(f"  sample >> {text[:200]}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"done in {total_min:.1f} min | best val loss {best_val:.3f} | "
          f"ppl {math.exp(min(best_val, 20)):,.1f}")
    text = sample(model, itos, device, args.sample_prompt, 600)
    out = os.path.join(SAMPLES_DIR, "final.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"final sample written to {out}")
    print("--- sample ---")
    print(text)


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample(model, itos, device, prompt: str, max_new_tokens: int,
           temperature: float = 0.7, top_k: int = 40) -> str:
    ids = [itos.get(ch, 0) for ch in prompt]  # 0 = <unk>
    if not ids:
        ids = [itos.get("в", 0) or 0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(y[0].tolist(), itos)


def sample_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(CKPT_PATH):
        raise SystemExit(f"no checkpoint at {CKPT_PATH} — train first")
    model, _, ck = load_checkpoint(device)
    text = sample(model, ck["vocab"], device, args.sample_prompt,
                  args.max_new_tokens, args.temperature, args.top_k)
    print(text)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Keep AI mini-Transformer trainer")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--sample-prompt", type=str, default="Вдруг ")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample", action="store_true", help="generate from checkpoint")
    ap.add_argument("--prompt", type=str, default=None, help="prompt for --sample")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    if args.sample:
        if args.prompt:
            args.sample_prompt = args.prompt
        sample_main(args)
    else:
        train(args)


if __name__ == "__main__":
    main()


