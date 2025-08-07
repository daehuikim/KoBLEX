# Naive Question Answering - OpenAI

This directory contains OpenAI-based implementations for legal question answering using different prompting strategies.

## Structure

```
naive/openai/
├── README.md           # This file
├── utils.py            # Utility functions for OpenAI API calls
├── cot_openai.py       # Chain-of-Thought question answering
├── sp_openai.py        # Simple Prompt question answering
├── cot_or_openai.py    # Chain-of-Thought with One-time Retrieval
└── sp_or_openai.py     # Simple Prompt with One-time Retrieval
```

## Files

### utils.py
Contains utility functions for:
- Loading and saving JSONL files
- OpenAI API key management
- Statute retrieval with BM25 indexing
- Text normalization and processing

### cot_openai.py
Chain-of-Thought (CoT) question answering using OpenAI API.
- Generates step-by-step reasoning before providing answers
- Uses structured prompts with reasoning and answer sections
- Supports parallel processing for efficiency

### sp_openai.py
Simple Prompt (SP) question answering using OpenAI API.
- Generates direct answers without step-by-step reasoning
- Uses concise prompts for quick responses
- Supports parallel processing for efficiency

### cot_or_openai.py
Chain-of-Thought with One-time Retrieval (CoT-OR) question answering using OpenAI API.
- Performs **one-time retrieval** of relevant legal passages using BM25
- Generates step-by-step reasoning based on retrieved context
- Uses `source/statute.jsonl` as the retrieval corpus
- Combines CoT reasoning with retrieved legal context
- Supports parallel processing for efficiency

### sp_or_openai.py
Simple Prompt with One-time Retrieval (SP-OR) question answering using OpenAI API.
- Performs **one-time retrieval** of relevant legal passages using BM25
- Generates direct answers based on retrieved context
- Uses `source/statute.jsonl` as the retrieval corpus
- Combines simple prompting with retrieved legal context
- Supports parallel processing for efficiency

## Usage

### Chain-of-Thought Question Answering

```bash
python cot_openai.py \
    --input source/koblex.jsonl \
    --output results/cot_answers.jsonl \
    --model_name gpt-4o \
    --max_parallel_calls 20
```

### Simple Prompt Question Answering

```bash
python sp_openai.py \
    --input source/koblex.jsonl \
    --output results/sp_answers.jsonl \
    --model_name gpt-4o \
    --max_parallel_calls 20
```

### Chain-of-Thought with One-time Retrieval

```bash
python cot_or_openai.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/cot_or_answers.jsonl \
    --model_name gpt-4o \
    --max_parallel_calls 20 \
    --n_hops 3
```

### Simple Prompt with One-time Retrieval

```bash
python sp_or_openai.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/sp_or_answers.jsonl \
    --model_name gpt-4o \
    --max_parallel_calls 20 \
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

### Standard QA (cot_openai.py, sp_openai.py)
The output JSONL file will contain records with the following structure:

```json
{
    "question": "Full question (background + question)",
    "answer": "Generated answer"
}
```

### One-time Retrieval QA (cot_or_openai.py, sp_or_openai.py)
The output JSONL file will contain records with the following structure:

```json
{
    "question": "Full question (background + question)",
    "answers": [{
        "answer": "Generated answer"
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
- **Parallel processing**: Supports concurrent API calls for efficiency

## Dependencies

- openai
- aiohttp
- tenacity
- bm25s
- sentence-transformers
- torch

## Environment Variables

Set the following environment variable:
- `OPENAI_API_KEY`: Your OpenAI API key

## Notes

- Both scripts support parallel processing to improve efficiency
- The CoT version provides more detailed reasoning but may be slower
- The SP version provides direct answers and is generally faster
- OR variants perform one-time retrieval to ground answers in legal context
- All prompts are defined in `../prompts/naive_prompt.py`
