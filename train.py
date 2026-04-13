import argparse
import random
import time

import datasets
from tqdm import tqdm

from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on WikiText-2")
    parser.add_argument("--vocab-size", type=int, default=512, help="Target vocabulary size (default: 512)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer.json",
        help="Path to write the learned tokenizer JSON (default: data/tokenizer.json)",
    )
    parser.add_argument("--validate", action="store_true", help="Run roundtrip validation after training")
    args = parser.parse_args()

    print("Loading WikiText-2 training split...")
    train_ds = datasets.load_dataset(
        path="Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        cache_dir="data",
        split=datasets.Split.TRAIN,
    )

    # Filter out blank lines (empty/whitespace-only rows); actual text content and its spaces are preserved.
    chunks = [
        row["text"]
        for row in tqdm(train_ds, desc="Preparing training chunks", unit="chunk")
        if row["text"].strip()
    ]
    print(f"Total chunks: {len(chunks)}")
    print(f"Total characters: {sum(len(c) for c in chunks):,}")

    tok = Tokenizer()
    num_merges = args.vocab_size - 256

    print(f"\nTraining with vocab_size={args.vocab_size} ({num_merges} merges)...")
    start = time.time()
    tok.train(chunks, vocab_size=args.vocab_size, show_progress=True)
    elapsed = time.time() - start

    print(f"Training completed in {elapsed:.2f}s")
    print(f"Vocab size: {len(tok.vocab)}")
    print(f"Merges learned: {len(tok.byte_pair_map)}")
    tok.save(args.output)
    print(f"Saved tokenizer JSON to {args.output}")

    # Show first 20 merges
    print("\nFirst 20 merges:")
    for (a, b), new_tok in list(tok.byte_pair_map.items())[:20]:
        a_str = bytes(tok.vocab[a]).decode("utf-8", errors="replace")
        b_str = bytes(tok.vocab[b]).decode("utf-8", errors="replace")
        merged_str = bytes(tok.vocab[new_tok]).decode("utf-8", errors="replace")
        print(f"  ({a}, {b}) -> {new_tok}  |  {a_str!r} + {b_str!r} = {merged_str!r}")

    if not args.validate:
        return

    # Roundtrip on training data
    print("\nValidating roundtrip on 100 training chunks...")
    for chunk in tqdm(chunks[:100], desc="Training roundtrip validation", unit="chunk"):
        decoded = tok.decode(tok.encode(chunk))
        assert decoded == chunk, f"Roundtrip failed for: {chunk[:80]!r}"
    print("Passed.")

    # Roundtrip on validation data
    print("Loading WikiText-2 validation split...")
    val_ds = datasets.load_dataset(
        path="Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        cache_dir="data",
        split=datasets.Split.VALIDATION,
    )
    val_chunks = [
        row["text"]
        for row in tqdm(val_ds, desc="Preparing validation chunks", unit="chunk")
        if row["text"].strip()
    ]

    print("Validating roundtrip on 100 validation chunks...")
    for chunk in tqdm(val_chunks[:100], desc="Validation roundtrip validation", unit="chunk"):
        decoded = tok.decode(tok.encode(chunk))
        assert decoded == chunk, f"Roundtrip failed for: {chunk[:80]!r}"
    print("Passed.")

    # Compression ratio
    sample = random.sample(val_chunks, min(200, len(val_chunks)))
    raw_bytes = sum(len(chunk.encode("utf-8")) for chunk in sample)
    total_tokens = sum(
        len(tok.encode(chunk))
        for chunk in tqdm(sample, desc="Measuring compression", unit="chunk")
    )

    print(f"\nCompression stats (on {len(sample)} validation samples):")
    print(f"  Raw UTF-8 bytes: {raw_bytes:,}")
    print(f"  Token count:     {total_tokens:,}")
    print(f"  Compression ratio: {raw_bytes / total_tokens:.2f}x")


if __name__ == "__main__":
    main()
