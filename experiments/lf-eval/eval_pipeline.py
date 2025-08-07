#!/usr/bin/env python3
"""
Legal Fidelity Evaluation Pipeline

This script evaluates legal question answering systems using multiple metrics:
- Token-level F1 Score
- Retrieval Precision, Recall, F1, and EM
- Legal Fidelity Score (LF-Eval)
"""

import argparse
import re
import os
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.evaluate import DisplayConfig


def normalize(text: str) -> str:
    """Normalize text for tokenization."""
    text = text.lower()
    text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return text.split()


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = tokenize(normalize(prediction))
    gt_tokens = tokenize(normalize(ground_truth))

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }


def evaluate_retrieval(pred_combinations: List[List[str]], gold_combinations: List[List[str]]) -> Dict[str, float]:
    """Evaluate retrieval performance using precision, recall, F1, and EM."""
    precisions, recalls, f1s, ems = [], [], [], []

    for pred, gold in zip(pred_combinations, gold_combinations):
        P = set(pred)
        G = set(gold)
        tp = len(P & G)
        fp = len(P) - tp
        fn = len(G) - tp

        # Handle empty sets: if both P and G are empty, F1=1.0, EM=1.0
        if not P and not G:
            p = r = f1 = 1.0
            em = 1.0
        else:
            p = tp / len(P) if P else 0.0
            r = tp / len(G) if G else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            em = 1.0 if (fp + fn) == 0 else 0.0
            
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ems.append(em)

    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s),
        'em': np.mean(ems)
    }


def evaluate_legal_fidelity(questions: List[str], predictions: List[str], 
                          gold_answers: List[str], gold_contexts: List[List[str]]) -> float:
    """Evaluate legal fidelity using GEval."""
    legal_fidelity_eval = GEval(
        name="legal-fidelity",
        criteria="You will be given a complex legal question along with the relevant legal provisions that can be used to resolve it.\nEvaluate the legal accuracy of the response on a scale from 1 to 10. \nA score of 1 means the response addresses a completely unrelated situation and a score of 10 means the response correctly and thoroughly uses all of the given legal provisions.\n",
        evaluation_steps=[
            "Check whether the 'actual output' properly answers the 'input' question.",
            "Check whether the 'actual output' contradicts or omits any legal facts or provisions in the 'context'.",
            "Heavily penalize if the legal conclusion differs in detail from the 'expected output'.",
            "Heavily penalize if the 'actual output' contradicts or omits any specific elements from the 'context'.",
            "Heavily penalize responses that include statements like 'The given context does not include the answer, but generally speaking...'."
        ],
        async_mode=True,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.CONTEXT]
    )
    
    test_cases = []
    for question, pred, answer, gold_context in zip(questions, predictions, gold_answers, gold_contexts):
        test_case = LLMTestCase(
            input=question,
            actual_output=pred,
            expected_output=answer,
            context=gold_context,
        )
        test_cases.append(test_case)

    legal_fidelity_results = evaluate(test_cases=test_cases, metrics=[legal_fidelity_eval])
    lg_scores = []
    for item in legal_fidelity_results.test_results:
        score = item.metrics_data[0].score
        lg_scores.append(score)
    
    return sum(lg_scores) / len(lg_scores) if lg_scores else 0.0


def load_data(gold_file: str, input_file: str, collections_file: str) -> tuple:
    """Load gold data, input data, and context collections."""
    # Load gold answers
    gold_data_list = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            gold_data_list.append(json.loads(line))
    
    gold_answer_list = []
    gold_nhop_list = []
    gold_context_combinations = []
    question_list = []
    
    for data in gold_data_list:
        gold_answer_list.append(data['answer'])
        question_list.append(data['background'] + data['question'])
        combs = data['contexts']
        gold_nhop_list.append(data['n_hops'])
        temp = []
        for item in combs:
            temp.append(item['hierarchy'] + item['content'])
        gold_context_combinations.append(temp)
    
    # Load input data
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    # Load context collections
    context_list = []
    with open(collections_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            context_list.append(item['hierarchy'] + item['content'])
    
    return gold_answer_list, gold_nhop_list, gold_context_combinations, question_list, data_list, context_list


def extract_predictions(data_list: List[Dict], mode: str = "nhop") -> tuple:
    """Extract predictions and combinations from data list."""
    answer_list = []
    combination_list = []
    
    for item in data_list:
        if mode == "naive":
            temp_answer = item['answers']
            answer_list.append(temp_answer)
            combination_list.append([])
        else:
            temp_answer = item['answers'][0]['answer']
            answer_list.append(temp_answer.split("<\think>")[0][:800] if len(temp_answer) > 0 else "")
            combination_list.append(item['answers'][0]['comb'])
    
    return answer_list, combination_list


def write_results(results: Dict[str, Any], input_file_symbol: str, output_dir: str = "results_log"):
    """Write evaluation results to log file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{input_file_symbol}.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\nFile name: {input_file_symbol}")
        
        if 'token_f1' in results:
            f.write(f"\n▶ Token-level F1   : {results['token_f1']:.4f}")
        
        if 'retrieval' in results:
            retrieval = results['retrieval']
            f.write(f"\n▶ Retrieval Precision: {retrieval['precision']:.4f}")
            f.write(f"\n▶ Retrieval   Recall : {retrieval['recall']:.4f}")
            f.write(f"\n▶ Retrieval       F1 : {retrieval['f1']:.4f}")
            f.write(f"\n▶ Retrieval       EM : {retrieval['em']:.4f}")
        
        if 'legal_fidelity' in results:
            f.write(f"\n\nAverage Legal Fidelity Score: {results['legal_fidelity']:.4f}\n")


def print_results(results: Dict[str, Any], input_file_symbol: str):
    """Print evaluation results to console."""
    print(f"File name: {input_file_symbol}")
    
    if 'token_f1' in results:
        print(f"▶ Token-level F1   : {results['token_f1']:.4f}")
    
    if 'retrieval' in results:
        retrieval = results['retrieval']
        print(f"▶ Retrieval Precision: {retrieval['precision']:.4f}")
        print(f"▶ Retrieval   Recall : {retrieval['recall']:.4f}")
        print(f"▶ Retrieval       F1 : {retrieval['f1']:.4f}")
        print(f"▶ Retrieval       EM : {retrieval['em']:.4f}")
    
    if 'legal_fidelity' in results:
        print(f"▶ Average Legal Fidelity Score: {results['legal_fidelity']:.4f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate legal question answering systems using multiple metrics"
    )
    
    # File arguments
    parser.add_argument(
        '--gold_file', 
        type=str, 
        default="source/koblex.jsonl",
        help="Path to gold standard data file"
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True,
        help="Path to input prediction file"
    )
    parser.add_argument(
        '--collections_file', 
        type=str, 
        default="source/statute.jsonl",
        help="Path to context collections file"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="results_log",
        help="Directory to save evaluation results"
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--eval_type', 
        type=str, 
        choices=['all', 'token_f1', 'retrieval', 'legal_fidelity'],
        default='all',
        help="Type of evaluation to perform"
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['naive', 'nhop'],
        default='nhop',
        help="Evaluation mode (naive or nhop)"
    )
    parser.add_argument(
        '--no_log', 
        action='store_true',
        help="Don't write results to log file"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    gold_answer_list, gold_nhop_list, gold_context_combinations, question_list, data_list, context_list = load_data(
        args.gold_file, args.input_file, args.collections_file
    )
    
    # Extract predictions
    print("Extracting predictions...")
    answer_list, combination_list = extract_predictions(data_list, args.mode)
    
    # Get input file symbol for logging
    input_file_symbol = Path(args.input_file).stem
    
    results = {}
    
    # Perform evaluations based on eval_type
    if args.eval_type in ['all', 'token_f1']:
        print("Evaluating token-level F1...")
        f1_scores = []
        for prediction, gold in zip(answer_list, gold_answer_list):
            scores = compute_token_f1(prediction, gold)
            f1_scores.append(scores['f1'])
        results['token_f1'] = np.mean(f1_scores)
    
    if args.eval_type in ['all', 'retrieval']:
        print("Evaluating retrieval performance...")
        retrieval_results = evaluate_retrieval(combination_list, gold_context_combinations)
        results['retrieval'] = retrieval_results
    
    if args.eval_type in ['all', 'legal_fidelity']:
        print("Evaluating legal fidelity...")
        legal_fidelity_score = evaluate_legal_fidelity(
            question_list, answer_list, gold_answer_list, gold_context_combinations
        )
        results['legal_fidelity'] = legal_fidelity_score
    
    # Print and log results
    print_results(results, input_file_symbol)
    
    if not args.no_log:
        write_results(results, input_file_symbol, args.output_dir)
        print(f"Results saved to {args.output_dir}/{input_file_symbol}.txt")


if __name__ == "__main__":
    main()