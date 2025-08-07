# Probtree Question Answering

This directory contains implementations for hierarchical question decomposition tree-based question answering using both VLLM and OpenAI API.

## Overview

Probtree uses a two-stage approach:
1. **Tree Generation**: Decompose complex legal questions into hierarchical question trees
2. **Tree-based QA**: Use the tree structure to answer questions using multiple strategies (CB, OB, CA)

## Structure

```
probtree/
├── README.md              # This file
├── run_pipeline.py        # Complete pipeline orchestrator
├── prompts/
│   ├── __init__.py        # Prompts module
│   └── probtree_prompt.py # All prompts and templates
├── utils.py               # Common utility functions
├── make_tree_vllm.py      # Tree generation using VLLM
├── make_tree_openai.py    # Tree generation using OpenAI
├── probtree_vllm.py       # Tree-based QA using VLLM
└── probtree_openai.py     # Tree-based QA using OpenAI
```

## Files

### run_pipeline.py
Complete pipeline orchestrator that runs both tree generation and question answering steps.

### prompts/probtree_prompt.py
Contains all prompts and templates:
- `MAKE_TREE_SYSTEM_PROMPT`: System prompt for tree generation
- `MAKE_TREE_QA_PROMPT`: Template for tree generation examples
- `CB_SYSTEM_PROMPT`, `OB_SYSTEM_PROMPT`, `CA_SYSTEM_PROMPT`: System prompts for different strategies
- `CB_TEMPLATE`, `OB_TEMPLATE`, `CA_TEMPLATE`: Templates for different question types

### utils.py
Common utility functions for:
- Loading and saving JSONL files
- BM25 retriever setup
- JSON extraction from text
- Log probability calculations

### make_tree_*.py
Generate hierarchical question decomposition trees:
- Decompose complex questions into atomic sub-questions
- Create tree structures for probtree question answering
- Support both VLLM and OpenAI backends

### probtree_*.py
Execute tree-based question answering:
- Use tree structures to answer questions
- Implement multiple strategies (CB, OB, CA)
- Support both VLLM and OpenAI backends

## Usage

### Complete Pipeline (Recommended)

```bash
# VLLM Pipeline
python run_pipeline.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/probtree_vllm.jsonl

# OpenAI Pipeline
python run_pipeline.py \
    --input source/koblex.jsonl \
    --context source/statute.jsonl \
    --output results/probtree_openai.jsonl \
    --use_openai
```

### Individual Steps

#### Step 1: Generate Trees

```bash
# VLLM Tree Generation
python make_tree_vllm.py \
    --input source/koblex.jsonl \
    --output results/tree_vllm.jsonl \
    --model_name LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct

# OpenAI Tree Generation
python make_tree_openai.py \
    --input source/koblex.jsonl \
    --output results/tree_openai.jsonl \
    --model_name gpt-4o
```

#### Step 2: Run Probtree QA

```bash
# VLLM Probtree QA
python probtree_vllm.py \
    --input results/tree_vllm.jsonl \
    --context source/statute.jsonl \
    --output results/probtree_vllm.jsonl \
    --model_name LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct

# OpenAI Probtree QA
python probtree_openai.py \
    --input results/tree_openai.jsonl \
    --context source/statute.jsonl \
    --output results/probtree_openai.jsonl \
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

### Probtree QA Output
```json
{
    "question": "Full question (background + question)",
    "answer": "Final answer",
    "trace": [
        {
            "idx": 0,
            "strategy": "ob",
            "logprob": -1.2
        }
    ],
    "combinations": ["retrieved_context_1", "retrieved_context_2"]
}
```

## Probtree Strategy

Probtree uses three different strategies for answering questions:

### 1. CB (Context-Based)
- Answers questions using general knowledge
- No retrieval performed
- Used when model has sufficient knowledge

### 2. OB (Out-of-Box)
- Performs BM25 retrieval to find relevant legal passages
- Answers based on retrieved context
- Used when specific legal information is needed

### 3. CA (Compositional Answer)
- Uses answers from child nodes to compose parent answer
- Only available for non-leaf nodes
- Used for complex questions that can be decomposed

The algorithm selects the strategy with the highest log probability score.

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
- LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
- Qwen/Qwen3-8B
- Other models compatible with VLLM

### Supported OpenAI Models
- gpt-4o
- gpt-4-turbo
- gpt-3.5-turbo

## Notes

- The tree generation step creates hierarchical question decomposition structures
- The probtree QA step uses these trees to answer questions using multiple strategies
- CB strategy uses general knowledge, OB uses retrieval, CA uses compositional reasoning
- All prompts are defined in `prompts/probtree_prompt.py`
- The pipeline automatically handles the two-stage process
- Tree files are intermediate outputs that can be reused for different QA runs
