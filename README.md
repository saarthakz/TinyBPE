# miniBPE

A minimalist and educational implementation of Byte Pair Encoding (BPE), the tokenization algorithm used by modern Large Language Models (LLMs) like GPT. This repository explores the core concepts of tokenization, from raw byte manipulation to vocabulary construction and efficient merging.

## Prerequisites

- **uv**: This project requires the [uv](https://github.com/astral-sh/uv) package manager for fast and reliable Python environment management.

## Installation

To set up the development environment, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saarthakz/miniBPE.git
   cd miniBPE
   ```

2. **Set up the virtual environment:**
   We provide a convenience script to automate the setup process using `uv`:
   ```bash
   bash setup_venv.sh
   ```

3. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

## Repository Structure

- `linked_list.py`: A custom Doubly Linked List implementation designed for efficient token merging during the BPE process.
- `tokenization.ipynb`: A comprehensive notebook walkthrough covering the BPE algorithm, including training, encoding, and decoding.
- `workbook.ipynb`: An experimental workspace for testing and iterating on BPE optimizations.
- `data/`: Contains sample text data used for training the tokenizer.

## Core Concepts Explored

- **Byte Pair Encoding (BPE)**: Learning a vocabulary of subword units by iteratively merging the most frequent pairs of adjacent tokens.
- **Efficient Merging**: Utilizing optimized data structures (like Doubly Linked Lists) to perform merges in-place.
- **UTF-8 Handling**: Working directly with byte representations to ensure universal language support.
- **Tokenizer vs. LLM**: Understanding why tokenization is a separate, critical step in the LLM pipeline.

## References

- [YouTube: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSF8StgHg) by Andrej Karpathy.
- [The Unicode Standard](https://unicode.org/standard/standard.html)
