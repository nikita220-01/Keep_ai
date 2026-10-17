"""Download a Russian text dataset (~10-50 MB) for training.

Source: Imperius/ru-classic on Hugging Face (CC0, classic Russian literature
from lib.ru, pre-cleaned). We download a single author file — Chekhov, the
master of the short story. Fallbacks: Turgenev (33 MB), Bunin (11 MB).

Usage:
    py -3 scripts/download_dataset.py
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/Imperius/ru-classic/resolve/main/per_author/"
# (filename, expected min size in bytes) — tried in order
CANDIDATES = [
    ("chekhov.txt", 10 * 1024 * 1024),
    ("turgenev.txt", 10 * 1024 * 1024),
    ("bunin.txt", 10 * 1024 * 1024),
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download(url: str, dest: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "keep-ai-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  downloaded {total / 1e6:.1f} MB ...", flush=True)
    return total


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, min_size in CANDIDATES:
        url = BASE + name
        dest = os.path.join(RAW_DIR, name)
        print(f"Trying {url}")
        try:
            size = download(url, dest)
        except Exception as exc:  # noqa: BLE001 — report and try next candidate
            print(f"  failed: {exc}")
            continue
        if size < min_size:
            print(f"  too small ({size} B), likely an error page — trying next")
            os.remove(dest)
            continue
        print(f"OK: {dest} ({size / 1e6:.1f} MB)")
        return
    print("All candidates failed", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
