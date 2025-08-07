# Legal Fidelity Evaluation (LF-Eval)

This directory contains the evaluation pipeline for legal question answering systems using multiple metrics including Legal Fidelity Score (LF-Eval).

## Overview

The evaluation pipeline supports multiple evaluation metrics:
- **Token-level F1 Score**: Measures word-level overlap between predictions and ground truth
- **Retrieval Metrics**: Precision, Recall, F1, and Exact Match for context retrieval
- **Legal Fidelity Score (LF-Eval)**: LLM-based evaluation of legal accuracy and faithfulness

## Structure

```
lf-eval/
├── README.md              # This file
├── eval_pipeline.py       # Main evaluation pipeline
└── results_log/           # Directory for evaluation results (auto-created)
```

## Files

### eval_pipeline.py
Main evaluation script that supports:
- Multiple evaluation types (token_f1, retrieval, legal_fidelity, all)
- Configurable input/output files
- Console output and log file generation
- Support for different prediction formats (naive, nhop)

## Usage

### Basic Usage

```bash
# Evaluate all metrics
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --gold_file source/koblex.jsonl \
    --collections_file source/statute.jsonl

# Evaluate only token-level F1
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --eval_type token_f1

# Evaluate only retrieval performance
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --eval_type retrieval

# Evaluate only legal fidelity
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --eval_type legal_fidelity
```

### Advanced Usage

```bash
# Custom output directory
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --output_dir custom_results

# Naive mode (for different prediction format)
python eval_pipeline.py \
    --input_file results/naive_predictions.jsonl \
    --mode naive

# No log file (console output only)
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --no_log
```

## Arguments

### File Arguments
- `--gold_file`: Path to gold standard data file (default: `source/koblex.jsonl`)
- `--input_file`: Path to input prediction file (required)
- `--collections_file`: Path to context collections file (default: `source/statute.jsonl`)
- `--output_dir`: Directory to save evaluation results (default: `results_log`)

### Evaluation Arguments
- `--eval_type`: Type of evaluation to perform
  - `all`: Evaluate all metrics (default)
  - `token_f1`: Evaluate only token-level F1
  - `retrieval`: Evaluate only retrieval metrics
  - `legal_fidelity`: Evaluate only legal fidelity
- `--mode`: Evaluation mode
  - `nhop`: Multi-hop prediction format (default)
  - `naive`: Simple prediction format
- `--no_log`: Don't write results to log file (console output only)

## Input Format

### Gold Standard Data (`--gold_file`)
```json
{
    "background": "Background information about the legal case...",
    "question": "The specific legal question to answer",
    "answer": "Gold standard answer",
    "contexts": [
        {
            "hierarchy": "Legal hierarchy information",
            "content": "Legal content"
        }
    ],
    "n_hops": 2
}
```

### Prediction Data (`--input_file`)

#### N-hop Mode (default)
```json
{
    "answers": [
        {
            "answer": "Predicted answer",
            "comb": ["retrieved_context_1", "retrieved_context_2"]
        }
    ]
}
```

#### Naive Mode
```json
{
    "answers": "Predicted answer"
}
```

### Context Collections (`--collections_file`)
```json
{
    "hierarchy": "Legal hierarchy information",
    "content": "Legal content"
}
```

## Output Format

### Console Output
```
File name: qa_exaone_32b_rerank_30
▶ Token-level F1   : 0.8234
▶ Retrieval Precision: 0.7567
▶ Retrieval   Recall : 0.8234
▶ Retrieval       F1 : 0.7889
▶ Retrieval       EM : 0.4567
▶ Average Legal Fidelity Score: 8.2345
```

### Log File (`results_log/{input_file_name}.txt`)
```
File name: qa_exaone_32b_rerank_30
▶ Token-level F1   : 0.8234
▶ Retrieval Precision: 0.7567
▶ Retrieval   Recall : 0.8234
▶ Retrieval       F1 : 0.7889
▶ Retrieval       EM : 0.4567

Average Legal Fidelity Score: 8.2345
```

## Metrics

### 1. Token-level F1 Score
- **Description**: Measures word-level overlap between predictions and ground truth
- **Range**: 0.0 to 1.0 (higher is better)
- **Calculation**: Uses normalized tokenization and F1 score computation

### 2. Retrieval Metrics
- **Precision**: Proportion of retrieved contexts that are relevant
- **Recall**: Proportion of relevant contexts that are retrieved
- **F1**: Harmonic mean of precision and recall
- **EM (Exact Match)**: Perfect retrieval (1.0) or not (0.0)
- **Range**: 0.0 to 1.0 (higher is better)

### 3. Legal Fidelity Score (LF-Eval)
- **Description**: LLM-based evaluation of legal accuracy and faithfulness
- **Range**: 1.0 to 10.0 (higher is better)
- **Criteria**:
  - Properly answers the input question
  - Doesn't contradict or omit legal facts
  - Matches expected output details
  - Uses all given legal provisions correctly
  - Penalizes generic responses

## Dependencies

- `deepeval`: For legal fidelity evaluation
- `numpy`: For numerical computations
- `pathlib`: For file path handling

## Installation

```bash
pip install deepeval numpy
```

## Example Execution Files

### Example 1: Complete Evaluation
```bash
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --gold_file source/koblex.jsonl \
    --collections_file source/statute.jsonl \
    --output_dir results_log
```

### Example 2: Quick Token F1 Check
```bash
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --eval_type token_f1 \
    --no_log
```

### Example 3: Legal Fidelity Only
```bash
python eval_pipeline.py \
    --input_file results/qa_exaone_32b_rerank_30.jsonl \
    --eval_type legal_fidelity \
    --output_dir legal_eval_results
```

### Example 4: Naive Mode Evaluation
```bash
python eval_pipeline.py \
    --input_file results/naive_predictions.jsonl \
    --mode naive \
    --eval_type all
```

## Notes

- The evaluation pipeline automatically creates the `results_log` directory if it doesn't exist
- Log files are appended to, so multiple runs will accumulate results
- Legal fidelity evaluation requires the `deepeval` library and may take longer than other metrics
- The pipeline supports both Korean and English text evaluation
- Token-level F1 uses normalized text processing for fair comparison

