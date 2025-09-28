# KoBLEX Parser Experiments

This directory contains the parser experiments for the KoBLEX legal question answering system. The experiments are organized into three main stages:

## Overview

1. **Parametric Provision Generation** - Generate relevant legal provisions for given questions
2. **Selection Retrieval** - Retrieve and rank the most relevant provisions
3. **Parser QA** - Generate final answers based on selected provisions

## Files

### Stage 1: Parametric Provision Generation
- `1.parametric_provision.py` - Uses vLLM for local model inference
- `1.parametric_provision_gpt.py` - Uses OpenAI API for inference

### Stage 2: Selection Retrieval
- `2.selection_retrieval.py` - Uses vLLM for local model inference
- `2.selection_retrieval_gpt.py` - Uses OpenAI API for inference

### Stage 3: Parser QA
- `3.parser_qa.py` - Uses vLLM for local model inference
- `3.parser_qa_gpt.py` - Uses OpenAI API for inference

## Usage

### Prerequisites

1. **For vLLM scripts**: Install vLLM and required dependencies
2. **For GPT scripts**: Set `OPENAI_API_KEY` environment variable

### Stage 1: Parametric Provision Generation

#### Using vLLM (Local Model)
```bash
python 1.parametric_provision.py \
    --input_path data/input.jsonl \
    --output_path results/parametric_provisions.jsonl \
    --model_name "your-model-name" \
    --tensor_parallel_size 1 \
    --temperature 0.0 \
    --top_p 0.9 \
    --max_tokens 4000
```

#### Using OpenAI API
```bash
python 1.parametric_provision_gpt.py \
    --input_path data/input.jsonl \
    --output_path results/parametric_provisions.jsonl \
    --prompt_path prompts/parametric_provision.txt \
    --model_name gpt-4o \
    --max_parallel_calls 20
```

### Stage 2: Selection Retrieval

#### Using vLLM (Local Model)
```bash
python 2.selection_retrieval.py \
    --input_path results/parametric_provisions.jsonl \
    --output_path results/selected_provisions.jsonl \
    --statute_path data/statutes.jsonl \
    --model_name "your-model-name" \
    --tensor_parallel_size 1 \
    --temperature 0.0 \
    --top_p 0.9 \
    --max_tokens 2048
```

#### Using OpenAI API
```bash
python 2.selection_retrieval_gpt.py \
    --input_path results/parametric_provisions.jsonl \
    --output_path results/selected_provisions.jsonl \
    --statute_path data/statutes.jsonl \
    --prompt_path prompts/selection_retrieval.txt \
    --model_name gpt-4o \
    --max_parallel_calls 20
```

### Stage 3: Parser QA

#### Using vLLM (Local Model)
```bash
python 3.parser_qa.py \
    --input_path results/selected_provisions.jsonl \
    --output_path results/final_answers.jsonl \
    --model_name "your-model-name" \
    --tensor_parallel_size 2 \
    --temperature 0.0 \
    --top_p 0.9 \
    --max_tokens 4000
```

#### Using OpenAI API
```bash
python 3.parser_qa_gpt.py \
    --input_path results/selected_provisions.jsonl \
    --output_path results/final_answers.jsonl \
    --prompt_path prompts/parser_qa.txt \
    --model_name gpt-4o \
    --max_parallel_calls 20
```

## Input/Output Formats

### Input Format (Stage 1)
```json
{
    "background": "Legal case background...",
    "question": "What is the legal question?",
    "contexts": [
        {
            "hierarchy": "법명 조항",
            "content": "법조문 내용"
        }
    ]
}
```

### Output Format (Stage 1)
```json
{
    "background": "Legal case background...",
    "question": "What is the legal question?",
    "subs": ["Generated provision 1", "Generated provision 2"]
}
```

### Output Format (Stage 2)
```json
{
    "background": "Legal case background...",
    "question": "What is the legal question?",
    "parametric_provisions": ["Generated provision 1", "Generated provision 2"],
    "selected_provisions": ["Selected provision 1", "Selected provision 2"],
    "answers": ["Ground truth provision 1", "Ground truth provision 2"]
}
```

### Output Format (Stage 3)
```json
{
    "background": "Legal case background...",
    "question": "What is the legal question?",
    "answer": "Final generated answer",
    "provisions": ["Selected provision 1", "Selected provision 2"]
}
```

## Performance Optimization

### For vLLM Scripts
- Adjust `tensor_parallel_size` based on your GPU setup
- Increase `max_tokens` for longer outputs
- Use appropriate `temperature` and `top_p` values

### For GPT Scripts
- Adjust `max_parallel_calls` based on your API rate limits
- Monitor token usage and costs
- Use appropriate model (gpt-4o, gpt-4, etc.)

## Troubleshooting

### Common Issues
1. **GPU Memory Issues**: Reduce `tensor_parallel_size` or use smaller models
2. **API Rate Limits**: Reduce `max_parallel_calls`
3. **File Not Found**: Ensure all input files exist and paths are correct
4. **Empty Results**: Check if input data format is correct

### Performance Tips
- Use tqdm progress bars to monitor execution
- Monitor GPU memory usage for vLLM scripts
- Check API usage and costs for GPT scripts
- Consider using smaller models for faster inference

## Dependencies

### Required Packages
```
vllm
transformers
torch
sentence-transformers
bm25s
tiktoken
openai
aiohttp
tenacity
tqdm
```

### Installation
```bash
pip install vllm transformers torch sentence-transformers bm25s tiktoken openai aiohttp tenacity tqdm
```