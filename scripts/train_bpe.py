"""Keep AI — BPE-level Transformer trainer (step 2: real dialogue model).

Trains MiniGPT on the BPE-encoded mixed corpus (Chekhov + Russian
dialogues). Model defaults are sized for CPU: ~3M params, block 256.

Usage:
    py -3 scripts/train_bpe.py                  # train from scratch
    py -3 scripts/train_bpe.py --resume         # continue
    py -3 scripts/train_bpe.py --sample --prompt "Привет"
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, MiniGPT, lr_for  # noqa: E402

import sentencepiece as spm  # noqa: E402

TOKENS_PATH = os.path.join(ROOT, "data", "bpe", "tokens_uint32.bin")
SP_MODEL = os.path.join(ROOT, "data", "bpe", "sp.model")
CKPT_DIR = os.path.join(ROOT, "data", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "keepai_bpe.pt")
SAMPLES_DIR = os.path.join(ROOT, "data", "samples")

# ~3M params, fits CPU training
BPE_CONFIG = dict(block_size=256, n_layer=4, n_head=4, n_embd=192, dropout=0.1)


def load_sp() -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=SP_MODEL)


def load_tokens():
    data = np.fromfile(TOKENS_PATH, dtype=np.uint32)
    n_val = int(len(data) * 0.01)
    return data[:-n_val], data[-n_val:]


def get_batch(data, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
                     for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_iters=30):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(model, cfg, step, best_val):
    os.makedirs(CKPT_DIR, exist_ok=True)
    from dataclasses import asdict
    torch.save({"model": model.state_dict(),
                "config": asdict(cfg),
                "step": step, "best_val": best_val,
                "type": "bpe", "sp_model": SP_MODEL}, CKPT_PATH)


def load_checkpoint(device: str):
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    fields = set(Config.__dataclass_fields__)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in fields})
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg, ck


def sample(model, cfg, sp, device, prompt: str, max_new_tokens: int = 150,
           temperature: float = 0.6, top_k: int = 32) -> str:
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = x[:, -cfg.block_size:]
            logits, _ = model(ctx)
            logits = logits[0, -1, :] / max(temperature, 1e-6)
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            x = torch.cat((x, torch.tensor([[nxt]], device=device)), dim=1)
            if nxt == 3:  # eos
                break
    model.train()
    return sp.decode(x[0].tolist())


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = load_sp()
    cfg = Config(vocab_size=sp.get_piece_size(), **BPE_CONFIG)

    model = MiniGPT(cfg).to(device)
    print(f"device={device}  params={model.num_params():,}  "
          f"vocab={cfg.vocab_size} (BPE)  block={cfg.block_size}  "
          f"layers={cfg.n_layer}  heads={cfg.n_head}  embd={cfg.n_embd}",
          flush=True)

    start_step, best_val = 0, float("inf")
    if args.resume and os.path.exists(CKPT_PATH):
        model, cfg, ck = load_checkpoint(device)
        start_step, best_val = ck["step"], ck["best_val"]
        print(f"resumed at step {start_step} (best_val={best_val:.3f})", flush=True)

    train_data, val_data = load_tokens()
    print(f"BPE tokens: train={len(train_data):,}  val={len(val_data):,}",
          flush=True)

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
            eta_min = (args.steps - step - 1) / max(sps, 1e-9) / 60
            print(f"step {step + 1:6d}/{args.steps} | loss {mean:.3f} | "
                  f"lr {lr:.2e} | {sps:.2f} it/s | ETA {eta_min:.0f} min",
                  flush=True)
            running = []

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            vl = estimate_loss(model, val_data, args.batch_size, cfg.block_size, device)
            print(f"  eval | val loss {vl:.3f} | ppl {math.exp(min(vl, 20)):,.1f}",
                  flush=True)
            if vl < best_val:
                best_val = vl
                save_checkpoint(model, cfg, step + 1, best_val)
                print(f"  checkpoint saved ({CKPT_PATH})", flush=True)

        if (step + 1) % args.sample_every == 0 and (step + 1) != args.steps:
            text = sample(model, cfg, sp, device, args.sample_prompt, 120)
            print(f"  sample >> {text[:200]}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"done in {total_min:.1f} min | best val {best_val:.3f}", flush=True)
    text = sample(model, cfg, sp, device, args.sample_prompt, 300)
    out = os.path.join(SAMPLES_DIR, "final_bpe.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"final sample -> {out}", flush=True)
    print("--- sample ---", flush=True)
    print(text, flush=True)


def sample_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(CKPT_PATH):
        raise SystemExit(f"no checkpoint at {CKPT_PATH} — train first")
    model, cfg, _ = load_checkpoint(device)
    sp = load_sp()
    text = sample(model, cfg, sp, device, args.sample_prompt,
                  args.max_new_tokens, args.temperature, args.top_k)
    print(text)


def main():
    ap = argparse.ArgumentParser(description="Keep AI BPE trainer")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--sample-prompt", type=str, default="- Привет, как дела?\n- ")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=32)
    args = ap.parse_args()
    if args.sample:
        if args.prompt:
            args.sample_prompt = args.prompt
        sample_main(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

