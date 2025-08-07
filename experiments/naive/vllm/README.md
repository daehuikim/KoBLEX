# Naive Question Answering - VLLM

This directory contains VLLM-based implementations for legal question answering using different prompting strategies.

## Structure

```
naive/vllm/
├── README.md           # This file
├── utils.py            # Utility functions for VLLM processing
├── cot_vllm.py         # Chain-of-Thought question answering
├── sp_vllm.py          # Simple Prompt question answering
├── cot_or_vllm.py      # Chain-of-Thought with One-time Retrieval
└── sp_or_vllm.py       # Simple Prompt with One-time Retrieval
```

## Files

### utils.py
Contains utility functions for:
- Loading and saving JSONL files
- Statute retrieval with BM25 indexing
- Text normalization and processing
- Answer extraction from completions

### cot_vllm.py
Chain-of-Thought (CoT) question answering using VLLM.
- Generates step-by-step reasoning before providing answers
- Uses structured prompts with reasoning and answer sections
- Optimized for local model inference

### sp_vllm.py
Simple Prompt (SP) question answering using VLLM.
- Generates direct answers without step-by-step reasoning
- Uses concise prompts for quick responses
- Optimized for local model inference

### cot_or_vllm.py
Chain-of-Thought with One-time Retrieval (CoT-OR) question answering using VLLM.
- Performs **one-time retrieval** of relevant legal passages using BM25
- Generates step-by-step reasoning based on retrieved context
- Uses `source/statute.jsonl` as the retrieval corpus
- Combines CoT reasoning with retrieved legal context

### sp_or_vllm.py
Simple Prompt with One-time Retrieval (SP-OR) question answering using VLLM.
- Performs **one-time retrieval** of relevant legal passages using BM25
- Generates direct answers based on retrieved context
- Uses `source/statute.jsonl` as the retrieval corpus
- Combines simple prompting with retrieved legal context

## Usage

### Chain-of-Thought Question Answering

```bash
python cot_vllm.py \
    --input source/koblex.jsonl \
    --output results/cot_answers.jsonl \
    --model_name Qwen/Qwen3-32B \
    --tensor_parallel_size 2 \
    --temperature 0.0 \
    --max_tokens 4000
```

### Simple Prompt Question Answering

```bash
python sp_vllm.py \
    --input source/koblex.jsonl \
    --output results/sp_answers.jsonl \
    --model_name Qwen/Qwen3-32B \
    --tensor_parallel_size 2 \
    --temperature 0.0 \
    --max_tokens 4000
```

### Chain-of-Thought with One-time Retrieval

```bash
python cot_or_vllm.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/cot_or_answers.jsonl \
    --model_name LGAI-EXAONE/EXAONE-3.5-32B-Instruct \
    --tensor_parallel_size 2 \
    --temperature 0.0 \
    --max_tokens 4000 \
    --n_hops 3
```

### Simple Prompt with One-time Retrieval

```bash
python sp_or_vllm.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/sp_or_answers.jsonl \
    --model_name LGAI-EXAONE/EXAONE-3.5-32B-Instruct \
    --tensor_parallel_size 2 \
    --temperature 0.0 \
    --max_tokens 4000 \
    --n_hops 3
```

## Input Format

The input JSONL file should contain records with the following structure:

```json
{
    "background": "Background information about the legal case...",
    "question": "The specific legal question to answer"
}
```

## Output Format

### Standard QA (cot_vllm.py, sp_vllm.py)
The output JSONL file will contain records with the following structure:

```json
{
    "question": "Full question (background + question)",
    "answer": "Generated answer"
}
```

### One-time Retrieval QA (cot_or_vllm.py, sp_or_vllm.py)
The output JSONL file will contain records with the following structure:

```json
{
    "question": "Full question (background + question)",
    "answers": [{
        "answer": "Generated answer",
        "logp": "Average log probability",
        "comb": ["retrieved_context_1", "retrieved_context_2", ...]
    }]
}
```

## One-time Retrieval (OR) Strategy

The OR (One-time Retrieval) variants perform **a single retrieval step** before generating answers:

1. **Retrieval**: Uses BM25 to retrieve the most relevant legal passages from `source/statute.jsonl`
2. **Context Integration**: Combines retrieved passages with the question
3. **Answer Generation**: Generates answers based on the retrieved context

Key features:
- **One-time retrieval**: Only performs retrieval once per question
- **BM25 indexing**: Uses BM25 algorithm for passage retrieval
- **Configurable n_hops**: Number of passages to retrieve (default: 3)
- **Context grounding**: Answers are grounded in retrieved legal passages

## Dependencies

- vllm
- transformers
- bm25s
- sentence-transformers
- torch

## Model Configuration

### Supported Models
- Qwen/Qwen3-32B
- LGAI-EXAONE/EXAONE-3.5-32B-Instruct
- Other models compatible with VLLM

### Key Parameters
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `temperature`: Sampling temperature (0.0 for deterministic)
- `top_p`: Top-p sampling parameter
- `max_tokens`: Maximum tokens to generate
- `trust_remote_code`: Whether to trust remote code for custom models
- `n_hops`: Number of context passages to retrieve (OR variants only)

## Notes

- VLLM provides efficient batch processing for local model inference
- The CoT version provides more detailed reasoning but may be slower
- The SP version provides direct answers and is generally faster
- OR variants perform one-time retrieval to ground answers in legal context
- All prompts are defined in `../prompts/naive_prompt.py`
- Tokenizer configurations are optimized for different models
