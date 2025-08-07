# Beam Aggregation Question Answering

This directory contains implementations for beam aggregation question answering using both VLLM and OpenAI API.

## Overview

Beam aggregation uses a two-stage approach:
1. **Tree Generation**: Decompose complex legal questions into hierarchical question trees
2. **Beam Aggregation QA**: Use the tree structure to answer questions using beam search with multiple strategies (CB, OB)

## Structure

```
beamaggr/
├── README.md              # This file
├── run_pipeline.py        # Complete pipeline orchestrator
├── prompts/
│   ├── __init__.py        # Prompts module
│   └── beamaggr_prompt.py # All prompts and templates
├── utils.py               # Common utility functions
├── make_tree_vllm.py      # Tree generation using VLLM
├── make_tree_openai.py    # Tree generation using OpenAI
├── beamaggr_vllm.py       # Beam aggregation QA using VLLM
└── beamaggr_openai.py     # Beam aggregation QA using OpenAI
```

## Files

### run_pipeline.py
Complete pipeline orchestrator that runs both tree generation and question answering steps.

### prompts/beamaggr_prompt.py
Contains all prompts and templates:
- `MAKE_TREE_SYSTEM_PROMPT`: System prompt for tree generation
- `MAKE_TREE_QA_PROMPT`: Template for tree generation examples
- `CB_SYSTEM_PROMPT`, `OB_SYSTEM_PROMPT`: System prompts for different strategies
- `CB_TEMPLATE`, `OB_TEMPLATE`: Templates for different question types

### utils.py
Common utility functions for:
- Loading and saving JSONL files
- BM25 retriever setup
- JSON extraction from text
- Log probability calculations
- Softmax probability calculations

### make_tree_*.py
Generate hierarchical question decomposition trees:
- Decompose complex questions into atomic sub-questions
- Create tree structures for beam aggregation question answering
- Support both VLLM and OpenAI backends

### beamaggr_*.py
Execute beam aggregation question answering:
- Use tree structures to answer questions
- Implement beam search with multiple strategies (CB, OB)
- Support both VLLM and OpenAI backends

## Usage

### Complete Pipeline (Recommended)

```bash
# VLLM Pipeline
python run_pipeline.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/beamaggr_vllm.jsonl \
    --model_name "your-model-name"

# OpenAI Pipeline
python run_pipeline.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/beamaggr_openai.jsonl \
    --use_openai
```

### Individual Steps

#### Step 1: Generate Trees

```bash
# VLLM Tree Generation
python make_tree_vllm.py \
    --input source/koblex.jsonl \
    --output results/tree_vllm.jsonl \
    --model_name "your-model-name"

# OpenAI Tree Generation
python make_tree_openai.py \
    --input source/koblex.jsonl \
    --output results/tree_openai.jsonl \
    --model_name gpt-4o
```

#### Step 2: Run Beam Aggregation QA

```bash
# VLLM Beam Aggregation QA
python beamaggr_vllm.py \
    --input results/tree_vllm.jsonl \
    --context source/statute.jsonl \
    --output results/beamaggr_vllm.jsonl \
    --model_name "your-model-name"

# OpenAI Beam Aggregation QA
python beamaggr_openai.py \
    --input results/tree_openai.jsonl \
    --context source/statute.jsonl \
    --output results/beamaggr_openai.jsonl \
    --model_name gpt-4o
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

### Tree Generation Output
```json
{
    "answer": {
        "parent_question": ["child_question1", "child_question2"],
        "child_question1": ["atomic_question1", "atomic_question2"]
    },
    "qd_logprob": -2.5,
    "background": "Background information..."
}
```

### Beam Aggregation QA Output
```json
{
    "question": "Full question (background + question)",
    "answer": "Final answer",
    "trace": [
        {
            "idx": 0,
            "strategy": "ob",
            "logprob": -1.2,
            "probability": 0.7
        }
    ],
    "combinations": ["retrieved_context_1", "retrieved_context_2"]
}
```

## Beam Aggregation Strategy

Beam aggregation uses two different strategies for answering questions:

### 1. CB (Context-Based)
- Answers questions using general knowledge
- No retrieval performed
- Used when model has sufficient knowledge

### 2. OB (Out-of-Box)
- Performs BM25 retrieval to find relevant legal passages
- Answers based on retrieved context
- Used when specific legal information is needed

The algorithm uses beam search with softmax probabilities to select the best strategy at each node.

## Dependencies

### VLLM
- vllm
- transformers
- bm25s
- sentence-transformers
- torch

### OpenAI
- openai
- aiohttp
- tenacity
- bm25s
- sentence-transformers
- torch

## Environment Variables

For OpenAI usage, set:
- `OPENAI_API_KEY`: Your OpenAI API key

## Model Configuration

### Supported VLLM Models
- Any model compatible with VLLM (specify via --model_name)
- Qwen/Qwen3-8B
- LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
- Other models compatible with VLLM

### Supported OpenAI Models
- gpt-4o
- gpt-4-turbo
- gpt-3.5-turbo

## Notes

- The tree generation step creates hierarchical question decomposition structures
- The beam aggregation QA step uses these trees to answer questions using beam search
- CB strategy uses general knowledge, OB uses retrieval
- All prompts are defined in `prompts/beamaggr_prompt.py`
- The pipeline automatically handles the two-stage process
- Tree files are intermediate outputs that can be reused for different QA runs
- Beam search uses softmax probabilities to select the best strategy at each node

