# TinyBPE

`TinyBPE` is a compact, educational Byte Pair Encoding (BPE) tokenizer project built to explore how modern LLM tokenizers work under the hood. The codebase starts from raw UTF-8 bytes, learns merge rules from text, and supports end-to-end training, encoding, decoding, and basic compression analysis.

## What This Project Does

- Trains a byte-level BPE tokenizer from text data.
- Encodes text into learned token IDs and decodes token IDs back into text.
- Uses an optimized merge pipeline based on a custom doubly linked list, pair-position cache, and an AVL tree for fast pair-frequency ranking during training.
- Includes a CLI training script that runs on WikiText-2 and can validate roundtrip correctness plus simple compression stats.

## Latest Updates

- Reworked training from a naive repeated-scan approach to an `O(N log N)` merge pipeline.
- Replaced the training-time ordered pair tracker with an explicit AVL tree implementation for efficient max-pair selection.
- Optimized `encode()` to replay learned merges using the same doubly linked list plus cache-map approach, without rescanning the full sequence on every merge.
- Added production-style `Tokenizer` APIs for `train()`, `encode()`, and `decode()`.
- Added a `train.py` script with WikiText-2 loading, merge inspection, roundtrip validation, and compression-ratio reporting.

## Project Structure

- `tokenizer.py`: Core BPE tokenizer implementation, including training, encoding, decoding, and merge bookkeeping.
- `avl_tree.py`: Explicit AVL tree used to maintain ordered `(frequency, pair)` entries during training.
- `linked_list.py`: Minimal doubly linked list used to support in-place token merges efficiently.
- `train.py`: Command-line entry point for training on WikiText-2 and evaluating the learned tokenizer.
- `setup_venv.sh`: Convenience script for creating a virtual environment with `uv`.
- `requirements.txt`: Python dependencies used by the project setup flow.

## Installation

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) for environment management

### Setup

```bash
git clone https://github.com/saarthakz/TinyBPE.git
cd TinyBPE
uv venv
source .venv/bin/activate
uv pip install datasets regex
```

If you prefer, you can also use the helper script:

```bash
bash setup_venv.sh
source .venv/bin/activate
```

## Usage

### Train the tokenizer

Train with the default vocabulary size (`512`):

```bash
python train.py
```

Train with a larger vocabulary:

```bash
python train.py --vocab-size 1024
```

Train and run validation:

```bash
python train.py --vocab-size 1024 --validate
```

The training script will:

- download and cache WikiText-2,
- train the tokenizer,
- print the first learned merges,
- optionally verify encode/decode roundtrips on train and validation samples,
- report a simple byte-to-token compression ratio on validation data.

## Example API

```python
from tokenizer import Tokenizer

tok = Tokenizer()
tok.train(["hello world", "hello there"], vocab_size=300)

encoded = tok.encode("hello world")
decoded = tok.decode(encoded)

print(encoded)
print(decoded)
```

## Implementation Notes

### Byte-level vocabulary

The base vocabulary is the 256 raw byte values. New tokens are created starting at ID `256` by merging the most frequent adjacent pairs.

### Fast merge bookkeeping

Instead of rescanning the full token stream after every merge, the tokenizer:

- stores tokens in a doubly linked list,
- caches every active adjacent pair and the nodes where it appears,
- maintains pair frequencies in an AVL tree keyed by `(frequency, pair)` so inserts, deletes, and max extraction stay logarithmic.

This keeps the project educational while making the training path much closer to a practical tokenizer implementation.

### Optimized encoding and decoding

- `train()` learns merge rules and reconstructs the merged-token vocabulary.
- `encode()` builds the same doubly linked list plus pair cache for new input, then replays learned merges in order without needing the training-time AVL ranking structure.
- `decode()` expands token IDs back into bytes and then into UTF-8 text.

## Why It’s Interesting

- Shows how byte-level BPE works without hiding the core mechanics behind a framework.
- Demonstrates how the right data structures materially improve tokenizer training performance.
- Connects algorithm design, systems-style data structures, and practical NLP preprocessing in a small codebase.

## References

- [Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSF8StgHg) by Andrej Karpathy
- [The Unicode Standard](https://unicode.org/standard/standard.html)
