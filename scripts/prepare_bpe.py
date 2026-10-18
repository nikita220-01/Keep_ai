"""Prepare mixed corpus (Chekhov prose + Russian dialogues) and train a
sentencepiece BPE tokenizer, then encode the corpus to a uint32 token file.

Outputs (data/bpe/):
  mixed.txt            — combined training text
  sp.model / sp.vocab  — BPE tokenizer (8000 tokens)
  tokens_uint32.bin    — encoded corpus (uint32 LE)
"""

import os

import numpy as np
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEKHOV = os.path.join(ROOT, "data", "clean", "corpus.txt")
DIALOGS = os.path.join(ROOT, "data", "raw", "ru.conversations.txt")
OUT_DIR = os.path.join(ROOT, "data", "bpe")
MIXED = os.path.join(OUT_DIR, "mixed.txt")
SP_PREFIX = os.path.join(OUT_DIR, "sp")
TOKENS = os.path.join(OUT_DIR, "tokens_uint32.bin")

VOCAB_SIZE = 8000


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CHEKHOV, encoding="utf-8") as f:
        chekhov = f.read()
    with open(DIALOGS, encoding="utf-8") as f:
        dialogs = f.read()
    # dialogues as "- реплика" lines (already in that format)
    with open(MIXED, "w", encoding="utf-8") as f:
        f.write(chekhov)
        f.write("\n")
        f.write(dialogs)
    print(f"mixed corpus: {len(chekhov):,} + {len(dialogs):,} chars",
          flush=True)

    print(f"training sentencepiece BPE (vocab={VOCAB_SIZE}) ...", flush=True)
    spm.SentencePieceTrainer.train(
        input=MIXED, model_prefix=SP_PREFIX, vocab_size=VOCAB_SIZE,
        model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        num_threads=os.cpu_count() or 4, hard_vocab_limit=False)

    sp = spm.SentencePieceProcessor(model_file=SP_PREFIX + ".model")
    print(f"vocab size = {sp.get_piece_size()}", flush=True)

    ids = sp.encode(chekhov, out_type=int)
    ids += sp.encode(dialogs, out_type=int)
    arr = np.array(ids, dtype=np.uint32)
    arr.tofile(TOKENS)
    print(f"encoded {len(arr):,} BPE tokens -> {TOKENS}", flush=True)


if __name__ == "__main__":
    main()
