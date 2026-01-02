# 📘 LLM From Scratch

This repository contains a complete, step-by-step implementation of building a Large Language Model (LLM) from the ground up. The project is structured as a sequence of notebooks, where each notebook introduces and implements a core concept required to understand how modern LLMs work internally.

---

## 📂 Project Structure

```
├── 01. Data_preparation_&_sampling.ipynb
├── 02. Vector_embedding.ipynb

```

Each notebook builds on concepts introduced in the previous one.

---

## 📘 01. Data_preparation_&_sampling.ipynb

This notebook focuses on transforming raw text into structured numerical data that can be used as input for language models.

### Topics Covered

### 1. Loading and Inspecting Raw Text
- Loads a short story (*The Verdict* by Edith Wharton).
- Reads and inspects raw text length and structure.
- Demonstrates how real-world datasets are ingested.

---

### 2. Tokenization from Scratch
- Implements tokenization using Python regular expressions.
- Splits text into:
  - Words
  - Punctuation
  - Special symbols
- Explains why tokenization design matters for language models.

---

### 3. Vocabulary Construction
- Extracts all unique tokens from the dataset.
- Builds a token-to-ID mapping.
- Creates an inverse ID-to-token mapping.
- Calculates vocabulary size.

---

### 4. Custom Tokenizer Implementations

#### SimpleTokenizerV1
- Converts text to token IDs.
- Converts token IDs back to text.
- Fails on unseen words (out-of-vocabulary problem).

#### SimpleTokenizerV2
- Introduces special tokens:
  - `<|unk|>` for unknown words
  - `<|endoftext|>` for document boundaries
- Handles unseen tokens safely.
- Demonstrates why special tokens are necessary in real LLMs.

---

### 5. Limitations of Word-Level Tokenization
- Shows failure cases for unseen words.
- Motivates the need for subword tokenization techniques.

---

### 6. Byte Pair Encoding (BPE) with GPT-2 Tokenizer
- Uses OpenAI’s `tiktoken` tokenizer.
- Demonstrates:
  - Encoding text into token IDs
  - Decoding tokens back into text
  - Handling unseen and compound words
- Explains why GPT-style tokenization does not require `<unk>` tokens.

---

### 7. Training Data Generation (Sliding Window)
- Converts tokenized text into `(input, target)` pairs.
- Uses a sliding window approach.
- Prepares sequences suitable for next-token prediction tasks.

---

### 8. Dataset and DataLoader Construction
- Implements a custom PyTorch `Dataset`.
- Uses `DataLoader` for batching and iteration.
- Supports configurable sequence length and stride.

---

### 9. Token Embeddings
- Creates token embeddings using `torch.nn.Embedding`.
- Maps token IDs to dense vector representations.

---

### 10. Positional Embeddings
- Adds positional information to token embeddings.
- Combines token and positional embeddings to form final model input.

---

## 📘 02. Vector_embedding.ipynb

This notebook explores **pretrained word embeddings** and semantic relationships using Word2Vec.

---

### 1. Loading Pretrained Word2Vec Model
- Uses Google’s Word2Vec (300-dimensional) embeddings via `gensim`.
- Loads a large pretrained semantic space.

---

### 2. Inspecting Word Vectors
- Examines vector values for individual words.
- Understands embedding dimensionality and structure.

---

### 3. Semantic Similarity
- Computes cosine similarity between word pairs.
- Demonstrates semantic closeness (e.g., *king–queen*, *boy–girl*).

---

### 4. Vector Arithmetic
- Performs analogy reasoning:
  - `king - man + woman ≈ queen`
- Shows how meaning is encoded geometrically.

---

### 5. Distance-Based Semantic Comparison
- Measures distance between related and unrelated words.
- Demonstrates how embeddings encode semantic relationships.

---

