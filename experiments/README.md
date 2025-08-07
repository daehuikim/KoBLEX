# Legal Question Answering Experiments

This repository contains implementations of various retrieval-augmented generation (RAG) methods for legal question answering, adapted for legal domain requirements.

## Overview

The experiments implement several state-of-the-art RAG methods, originally designed for general question answering but adapted for legal domain specifics. Each method has been reimplemented following the original paper descriptions and repository logic, with modifications to work with legal search tools instead of general search engines.

## Proposed Method: Parser

**Parser** is our novel proposed methodology that generates parametric provisions and uses 3-stage retrieval to improve legal statute search accuracy. This approach enhances retrieval precision by creating structured parametric representations of legal provisions and implementing a multi-stage retrieval pipeline.

**Key Features**:
- Parametric provision generation for structured legal representation
- 3-stage retrieval pipeline for enhanced search accuracy
- Question answering with selection retrieval
- Support for both VLLM and OpenAI backends

## Implemented Baseline Methods

The following descriptions are for baseline methods that serve as comparison points for our proposed parser approach:

### 1. [Self-Ask](self-ask/)
**Original Paper**: [Measuring and Narrowing the Compositionality Gap in Language Models](https://github.com/ofirpress/self-ask)

Self-Ask is a prompting method that decomposes complex questions into simpler sub-questions. The model asks itself follow-up questions to gather information before providing a final answer.

**Key Features**:
- Question decomposition into sub-questions
- Step-by-step reasoning process
- Support for both VLLM and OpenAI backends

### 2. [IRCoT (Interleaving Retrieval with Chain-of-Thought)](ircot/)
**Original Paper**: [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://github.com/StonyBrookNLP/ircot)

IRCoT interleaves retrieval steps with chain-of-thought reasoning, allowing the model to retrieve relevant information at each reasoning step.

**Key Features**:
- Interleaved retrieval and reasoning
- Multi-step question decomposition
- Dynamic context retrieval during reasoning
- Support for both VLLM and OpenAI backends

### 3. [FLARE (Forward-Looking Active REtrieval)](flare/)
**Original Paper**: [FLARE: Forward-Looking Active REtrieval Augmented Generation](https://github.com/jzbjyb/FLARE)

FLARE uses forward-looking active retrieval to anticipate future information needs and retrieve relevant documents proactively.

**Key Features**:
- Forward-looking retrieval strategy
- Active information gathering
- Anticipatory context retrieval
- Support for both VLLM and OpenAI backends

### 4. [ProbTree (Probabilistic Tree)](probtree/)
**Original Paper**: [ProbTree: A Probabilistic Framework for Reasoning over Logical Trees](https://github.com/THU-KEG/ProbTree)

ProbTree uses hierarchical question decomposition trees with probabilistic reasoning to answer complex questions through multiple strategies.

**Key Features**:
- Hierarchical question decomposition
- Multiple reasoning strategies (CB, OB, CA)
- Probabilistic strategy selection
- Two-stage pipeline (tree generation + QA)
- Support for both VLLM and OpenAI backends

### 5. [BeamAggr (Beam Aggregation)](beamaggr/)
**Original Paper**: [Beam Aggregation: A Novel Framework for Retrieval-Augmented Generation](https://aclanthology.org/2024.acl-long.67/)

BeamAggr uses beam search with multiple strategies to aggregate information from different retrieval paths and reasoning approaches.

**Key Features**:
- Beam search with multiple strategies
- Softmax probability-based strategy selection
- Context-based (CB) and Out-of-box (OB) strategies
- Two-stage pipeline (tree generation + beam aggregation)
- Support for both VLLM and OpenAI backends

## Evaluation

### [LF-Eval (Legal Fidelity Evaluation)](lf-eval/)
**Based on**: [DeepEval](https://github.com/confident-ai/deepeval)

LF-Eval is a novel retrieval-augmented generation metric specifically designed for legal question answering. It uses G-Eval-based evaluation to assess legal accuracy and faithfulness.

**Evaluation Metrics**:
- **Token-level F1 Score**: Measures word-level overlap between predictions and ground truth
- **Retrieval Metrics**: Precision, Recall, F1, and Exact Match for context retrieval
- **Legal Fidelity Score**: LLM-based evaluation of legal accuracy and faithfulness (1-10 scale)

**Key Features**:
- Multiple evaluation types (token_f1, retrieval, legal_fidelity, all)
- Configurable input/output files
- Console output and log file generation
- Support for different prediction formats (naive, nhop)

## Common Features Across All Methods

### Architecture
- **Modular Design**: Each method is implemented as a separate module with consistent interfaces
- **Dual Backend Support**: All methods support both VLLM (local) and OpenAI (API) backends
- **Configurable Parameters**: Command-line argument parsing for all parameters
- **Pipeline Structure**: Consistent input/output formats across all methods

### Data Processing
- **Input**: `source/koblex.jsonl` - Legal question dataset
- **Context**: `source/statute.jsonl` - Legal statute collection
- **Output**: JSONL format with predictions and metadata

### Prompt Management
- **Centralized Prompts**: All prompts are stored in dedicated `prompts/` directories
- **No Hardcoding**: All file paths and parameters are configurable
- **Template System**: Uses string templates for dynamic prompt generation

### Utility Functions
- **Common Utils**: Shared utility functions for data loading, BM25 retrieval, etc.
- **Error Handling**: Robust error handling and validation
- **Progress Reporting**: Progress bars and status updates during processing

## Usage Examples

### Running Individual Methods

```bash
# Parser (our proposed method)
cd parser
python run_pipeline.py --input source/koblex.jsonl --output results/parser.jsonl

# Self-Ask with VLLM
cd self-ask
python run_selfask.sh

# IRCoT with OpenAI
cd ircot
python run_ircot.sh --use_openai

# FLARE with VLLM
cd flare
python run_flare.sh

# ProbTree complete pipeline
cd probtree
python run_pipeline.py --input source/koblex.jsonl --output results/probtree.jsonl

# BeamAggr complete pipeline
cd beamaggr
python run_pipeline.py --input source/koblex.jsonl --output results/beamaggr.jsonl --use_openai
```

### Evaluation

```bash
# Complete evaluation
cd lf-eval
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --gold_file source/koblex.jsonl \
    --collections_file source/statute.jsonl

# Quick token F1 check
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --eval_type token_f1 \
    --no_log
```

## Dependencies

### Core Dependencies
- `vllm`: For local model inference
- `openai`: For OpenAI API access
- `transformers`: For model loading and tokenization
- `torch`: For PyTorch operations
- `numpy`: For numerical computations

### Retrieval Dependencies
- `bm25s`: For BM25 retrieval
- `sentence-transformers`: For semantic search
- `Stemmer`: For text stemming

### Evaluation Dependencies
- `deepeval`: For legal fidelity evaluation
- `pathlib`: For file path handling

### Additional Dependencies
- `aiohttp`: For async HTTP requests
- `tenacity`: For retry logic
- `tqdm`: For progress bars

## Installation

```bash
# Core dependencies
pip install vllm openai transformers torch numpy

# Retrieval dependencies
pip install bm25s sentence-transformers Stemmer

# Evaluation dependencies
pip install deepeval

# Additional dependencies
pip install aiohttp tenacity tqdm
```

## Environment Variables

For OpenAI usage, set:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Project Structure

```
experiments/
├── README.md              # This file
├── source/                # Data files
│   ├── koblex.jsonl      # Legal questions dataset
│   └── statute.jsonl     # Legal statutes collection
├── parser/                # Parser (our proposed method)
├── self-ask/             # Self-Ask implementation
├── ircot/                # IRCoT implementation
├── flare/                # FLARE implementation
├── probtree/             # ProbTree implementation
├── beamaggr/             # BeamAggr implementation
└── lf-eval/              # Legal Fidelity Evaluation
```

## Notes

- All methods have been adapted from their original implementations to work with legal domain requirements
- BM25 retrieval is used instead of Elasticsearch for legal document search
- Each method maintains the core logic from the original papers while adapting to legal question answering
- The evaluation system (LF-Eval) is specifically designed for legal domain assessment
- All implementations support both local (VLLM) and cloud (OpenAI) model backends
- Comprehensive documentation is provided for each method in their respective directories

