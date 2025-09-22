# Parser Pipeline

This directory contains a parser pipeline for legal question-answering tasks.

## Usage

### GPT-based Pipeline

#### 1. Parametric Provision Generation
```bash
python 1.parametric_provision_gpt.py --model_name gpt-4o --max_parallel_calls 20
```

**Input**: `input-path` - JSONL file containing questions and backgrounds
**Output**: `output-path` - JSONL file with parametric provisions added

#### 2. Selection Retrieval
```bash
python 2.selection_retrieval_gpt.py --model_name gpt-4o --max_parallel_calls 20
```

**Input**: `input-path` - JSONL file containing parametric provisions
**Output**: `output-path` - JSONL file with selected provisions

#### 3. Question Answering
```bash
python 3.parser_qa_gpt.py --model_name gpt-4o --max_parallel_calls 20
```

**Input**: `input-path` - JSONL file containing selected provisions
**Output**: `output-path` - JSONL file with final answers

### vLLM-based Pipeline

#### 1. Parametric Provision Generation
```bash
python 1.parametric_provision.py
```

**Input**: Modify `input_file` variable in the script
**Output**: Modify `output_file` variable in the script

#### 2. Selection Retrieval
```bash
python 2.selection_retrieval.py
```

**Input**: Modify `input_file` and `collections_file` variables in the script
**Output**: Modify `output_path` variable in the script

#### 3. Question Answering
```bash
python 3.parser_qa.py
```

**Input**: Modify `input_file` variable in the script
**Output**: Modify `out_file` variable in the script

## File Structure

- `1.parametric_provision_gpt.py`: Parametric provision generation using GPT
- `2.selection_retrieval_gpt.py`: Provision selection and reranking using GPT
- `3.parser_qa_gpt.py`: Question-answering generation using GPT
- `1.parametric_provision.py`: Parametric provision generation using vLLM
- `2.selection_retrieval.py`: Provision selection and reranking using vLLM
- `3.parser_qa.py`: Question-answering generation using vLLM

## Configuration

Before running each script, modify the following paths to actual file paths:

- `input-path`: Input JSONL file path
- `output-path`: Output JSONL file path
- `statute-path`: Statute data JSONL file path (used in selection retrieval)
- `prompt-path`: Prompt text file path

## Environment Variables

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

---

## Postscript

This code has been refactored while maintaining the original logic for transparency. The complex structure of the original code has been simplified and modularized to improve readability and maintainability. Clear input/output formats have been defined for each step, and unnecessary code has been removed to build a more efficient pipeline.
