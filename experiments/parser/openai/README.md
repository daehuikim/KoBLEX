# Legal Question Answering Pipeline (OpenAI)

This directory contains a three-stage pipeline for legal question answering using OpenAI API:

1. **Parametric Provision Generation** (`parametric_provision.py`)
2. **Selection Retrieval** (`selection_retrieval.py`) 
3. **Question Answering** (`question_answering.py`)

## Features

- **OpenAI API Integration**: Uses OpenAI's GPT models for text generation
- **Asynchronous Processing**: Parallel API calls for efficient processing
- **Retry Logic**: Automatic retry with exponential backoff for API failures
- **BM25 + CrossEncoder**: Advanced retrieval and reranking system
- **Modular Design**: Separated prompts, utilities, and main logic

## Quick Start

To run the complete pipeline:

```bash
python run_pipeline.py
```

This will:
1. Process `source/koblex.jsonl` through parametric provision generation
2. Use `source/statute.jsonl` as the search pool for selection retrieval
3. Generate final answers in `final_results.jsonl`

## File Structure

- `parametric_provision.py` - Generates parametric provisions from legal questions
- `selection_retrieval.py` - Retrieves and selects relevant statutory provisions
- `question_answering.py` - Generates final answers using selected provisions
- `run_pipeline.py` - Executable script that chains all three components
- `utils.py` - Common utility functions and API helpers
- `raw.py` - Original monolithic script (for reference)

## Input/Output Files

- **Input**: `source/koblex.jsonl` (legal questions)
- **Search Pool**: `source/statute.jsonl` (statutory provisions)
- **Intermediate**: `parametric_provisions.jsonl`, `retrieved_provisions.jsonl`
- **Output**: `final_results.jsonl` (final answers)

## Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Model Configuration

The pipeline uses `gpt-4o` by default, but you can change it in `run_pipeline.py` or pass different model names to individual scripts.

## API Rate Limiting

The pipeline includes:
- Parallel processing with configurable concurrency
- Automatic retry with exponential backoff
- Rate limiting to respect OpenAI's API limits

## Cost Estimation

The pipeline estimates costs based on token usage. Monitor your OpenAI usage to stay within budget limits.
