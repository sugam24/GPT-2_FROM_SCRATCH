# LLM FROM SCRATCH

# 01. Data_preparation_&_sampling.ipynb

This notebook covers the **foundational text processing pipeline** required before training a Large Language Model (LLM). It demonstrates how raw text is transformed into numerical representations that an LLM can learn from, using both **custom tokenizers** and **GPT-style Byte Pair Encoding (BPE)**, followed by **data sampling and embedding preparation**.

---

## Overview of Work Done

### 1. Loading and Inspecting Raw Text
- A short story (*The Verdict* by Edith Wharton) is downloaded from Kaggle.
- The dataset contains **20,479 characters**, serving as a manageable educational example.
- The raw text is inspected to understand its structure and content.

---

### 2. Tokenization from Scratch (Regex-based)
- Text is split into tokens using Python’s `re` module.
- Tokens include:
  - Words
  - Punctuation (`. , ? -- " ( )` etc.)
- Whitespaces are optionally removed for simplicity.
- The full story is tokenized into **4,690 tokens**.

**Key takeaway:** Tokenization converts raw text into discrete units that models can process.

---

### 3. Vocabulary Construction & Token IDs
- All unique tokens are collected and sorted.
- A vocabulary is built mapping:
  - **token → integer ID**
- Vocabulary size: **1,130 tokens**
- An inverse vocabulary (**ID → token**) is also created for decoding.

---

### 4. Custom Tokenizer Implementation
Two tokenizer versions are implemented:

#### `SimpleTokenizerV1`
- Encodes text into token IDs using the vocabulary.
- Decodes token IDs back into readable text.
- Fails on unseen words (out-of-vocabulary problem).

#### Limitation Highlighted
- Encoding unseen words (e.g. `"Hello"`) raises a `KeyError`.

---

### 5. Handling Unknown Tokens & Special Tokens
To address vocabulary limitations:

#### Added Special Tokens
- `<|unk|>` → unknown words
- `<|endoftext|>` → document boundary marker

Vocabulary size increases to **1,132 tokens**.

#### `SimpleTokenizerV2`
- Replaces unseen words with `<|unk|>`
- Supports multi-document text via `<|endoftext|>`
- Successfully encodes and decodes unseen inputs.

---

### 6. Discussion on Special Tokens
Conceptual explanation of commonly used tokens in LLMs:
- **BOS** (Beginning of Sequence)
- **EOS** (End of Sequence)
- **PAD** (Padding)
- GPT-style models primarily rely on `<|endoftext|>` and **subword tokenization** instead of `<|unk|>`.

---

### 7. Byte Pair Encoding (BPE) with `tiktoken`
- Introduces OpenAI’s **GPT-2 BPE tokenizer**.
- Demonstrates:
  - Encoding unseen words without `<|unk|>`
  - Subword-level tokenization
- Shows how GPT tokenizers handle arbitrary text robustly.

---

### 8. Data Sampling with Sliding Window
- Tokenized text is converted into **input–target pairs**.
- Uses a **sliding context window**:
  - Input: previous tokens
  - Target: next token
- This forms the basis of **next-token prediction**, the core LLM training objective.

---

### 9. PyTorch Dataset & DataLoader
- `GPTDatasetV1` is implemented to:
  - Chunk token sequences using `max_length` and `stride`
  - Generate `(input_ids, target_ids)` pairs
- A reusable `create_dataloader_v1` function is provided.
- Demonstrates batching, overlapping windows, and sequence alignment.

---

### 10. Token Embeddings
- Uses `torch.nn.Embedding` to map token IDs → dense vectors.
- Shows:
  - Embedding weight initialization
  - Lookup of embeddings for token sequences

---

### 11. Positional Embeddings
- Introduces **positional encoding** to retain word order.
- Separate embedding layers for:
  - Token embeddings
  - Position embeddings
- Final input embeddings are computed as:
  
