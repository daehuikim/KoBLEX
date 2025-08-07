# Legal Question Answering Pipeline

This directory contains a three-stage pipeline for legal question answering using vLLM:

1. **Parametric Provision Generation** (`parametric_provision.py`)
2. **Selection Retrieval** (`selection_retrieval.py`) 
3. **Question Answering** (`question_answering.py`)

## Reproducibility Note

Even with `temperature=0`, vLLM may produce slightly different results due to inherent randomness in:
- Model initialization and tokenizer processing
- Hardware/environment variations
- Multiprocessing and CUDA kernel scheduling
- Memory allocation patterns

For reproducible results, the code includes the following environment variables:
```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
```

**We strongly recommend using these reproducibility settings for consistent results.**

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
- `prompts/` - Contains prompt templates for each component
- `utils.py` - Common utility functions

## Input/Output Files

- **Input**: `source/koblex.jsonl` (legal questions)
- **Search Pool**: `source/statute.jsonl` (statutory provisions)
- **Intermediate**: `parametric_provisions.jsonl`, `retrieved_provisions.jsonl`
- **Output**: `final_results.jsonl` (final answers)
