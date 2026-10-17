"""Clean a raw Russian text file and prepare it for tokenization.

Pipeline:
  1. Normalize encoding/newlines, strip BOM/NBSP.
  2. Remove lib.ru glosses { ... }, URLs, emails, page artifacts.
  3. Unify typography: quotes «», em dash —, ellipsis …; ё -> е.
  4. Drop junk lines (punctuation-only, latin-only, too short).
  5. Collapse whitespace, split into paragraphs, drop duplicate paragraphs.
  6. Build a character-level vocabulary (specials + chars by frequency)
     and optionally encode the whole corpus to a uint32 binary stream.

Outputs (in data/clean/):
  corpus.txt          — clean UTF-8 text, paragraphs separated by blank line
  vocab.json          — {token: id}, ids 0..3 are <unk> <s> </s> <pad>
  stats.json          — cleaning statistics
  preview_tokens.txt  — example: text -> ids
  tokens_uint32.bin   — (only with --encode) full corpus as uint32 ids

Usage:
    py -3 scripts/prepare_data.py [--input data/raw/chekhov.txt]
                                  [--output data/clean] [--encode] [--keep-yo]
"""

import argparse
import json
import os
import re
import sys
import time
from array import array
from collections import Counter

SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]

RE_GLOSS = re.compile(r"\{[^}]*\}")            # lib.ru translation notes
RE_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
RE_NAV = re.compile(r"lib\.ru|az\.lib|гутенберг|классическ\w* библиотек\w*", re.I)
RE_PAGE = re.compile(r"^\s*(стр\.?\s*\d+|-\s*\d+\s*-|\d+)\s*$")
RE_JUNK = re.compile(r"^[.\-_=*~…—•·\s]+$")   # punctuation/separator-only
RE_SPACES = re.compile(r" {2,}")
RE_DASH = re.compile(r"\s*—\s*")


def unify_quotes(line: str) -> str:
    """Convert straight/typewriter quotes to Russian « ».

    Heuristic: a quote right after a letter/digit/closing punctuation is a
    closing quote, otherwise it is an opening one.
    """
    out = []
    prev = " "  # treat line start as "after a space"
    for ch in line:
        if ch == '"':
            if prev.isalnum() or prev in ".,!?;:»…":
                out.append("»")
            else:
                out.append("«")
        else:
            out.append(ch)
            prev = ch
    return "".join(out)


def clean_line(line: str, keep_yo: bool) -> str | None:
    """Return cleaned line, or None if the line is junk and should be dropped."""
    line = line.replace("\u00a0", " ").replace("\ufeff", "")
    line = RE_GLOSS.sub("", line)                     # {Кстати (франц.).}
    line = RE_URL.sub(" ", line)
    if RE_NAV.search(line) or RE_PAGE.match(line):
        return None
    line = line.replace("--", "—")
    line = re.sub(r"\.{3,}", "…", line)
    line = unify_quotes(line)
    line = RE_SPACES.sub(" ", line).strip()
    line = re.sub(r" +([,;:!?…])", r"\1", line)         # space before punct
    line = RE_DASH.sub(" — ", line).strip()
    if not keep_yo:
        line = line.replace("ё", "е").replace("Ё", "Е")
    if not line or len(line) < 2:
        return None
    if RE_JUNK.match(line):
        return None
    cyr = sum(1 for c in line if "а" <= c.lower() <= "я" or c in "ёЁ")
    if cyr == 0:                                        # latin-only line
        return None
    if cyr / len(line) < 0.3 and not line.startswith("—"):
        return None                                     # mostly non-Cyrillic
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="raw .txt file (default: first .txt in data/raw)")
    ap.add_argument("--output", default=os.path.join("data", "clean"))
    ap.add_argument("--encode", action="store_true",
                    help="write full corpus as uint32 token ids")
    ap.add_argument("--keep-yo", action="store_true",
                    help="do not fold ё -> е")
    args = ap.parse_args()

    if args.input is None:
        raw_dir = os.path.join("data", "raw")
        txts = sorted(f for f in os.listdir(raw_dir) if f.endswith(".txt"))
        if not txts:
            sys.exit(f"No .txt in {raw_dir}; run scripts/download_dataset.py first")
        args.input = os.path.join(raw_dir, txts[0])

    with open(args.input, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    stats: Counter = Counter()
    out_lines = []
    for line in raw.splitlines():
        stats["lines_in"] += 1
        if not line.strip():
            out_lines.append("")                      # keep paragraph breaks
            continue
        cleaned = clean_line(line, args.keep_yo)
        if cleaned is None:
            stats["lines_dropped"] += 1
            continue
        out_lines.append(cleaned)
    text = "\n".join(out_lines)

    # Paragraphs + exact-duplicate removal (long paragraphs only)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen: set[str] = set()
    kept = []
    for p in paragraphs:
        key = p.strip()
        if len(key) >= 150:
            if key in seen:
                stats["paragraphs_dup"] += 1
                continue
            seen.add(key)
        kept.append(key)
    text = "\n\n".join(kept) + "\n"
    stats["paragraphs_out"] = len(kept)
    stats["chars_out"] = len(text)
    stats["chars_in"] = len(raw)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    # Character-level vocabulary (prep for tokenization -> numbers)
    counts = Counter(text)
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for ch, _ in counts.most_common():
        if ch not in vocab:
            vocab[ch] = len(vocab)
    with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)

    # Preview: first 400 chars as ids
    sample = text[:400]
    ids = [vocab.get(c, 0) for c in sample]
    with open(os.path.join(args.output, "preview_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("text:\n" + sample + "\n\nids:\n" + " ".join(map(str, ids)) + "\n")

    if args.encode:
        t0 = time.time()
        ids_all = array("I", (vocab.get(c, 0) for c in text))  # uint32
        with open(os.path.join(args.output, "tokens_uint32.bin"), "wb") as f:
            ids_all.tofile(f)
        stats["encode_seconds"] = round(time.time() - t0, 1)

    stats["vocab_size"] = len(vocab)
    stats["word_tokens_est"] = len(text.split())
    with open(os.path.join(args.output, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Done.")
    for k in ("chars_in", "chars_out", "lines_in", "lines_dropped",
              "paragraphs_out", "paragraphs_dup", "word_tokens_est", "vocab_size"):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()

