#!/usr/bin/env python3
"""
Probtree Question Answering Generator (OpenAI)

This script performs hierarchical question decomposition tree-based question answering using OpenAI API.
It uses tree structures to decompose complex questions and answer them using multiple strategies.
"""

import argparse
import asyncio
import json
import os
from typing import List, Dict, Any, Tuple
import bm25s
from openai import OpenAI
from tqdm import tqdm

from prompts.probtree_prompt import (
    CB_SYSTEM_PROMPT, OB_SYSTEM_PROMPT, CA_SYSTEM_PROMPT,
    CB_TEMPLATE, OB_TEMPLATE, CA_TEMPLATE
)
from utils import load_jsonl, save_jsonl, setup_retriever, avg_logprob, lp_to_list, decide_retrieval


def load_api_key():
    """Load OpenAI API key from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def get_completion(prompt_text, system_prompt, model_name="gpt-4o"):
    """Get completion from OpenAI API."""
    client = OpenAI(api_key=load_api_key())
    
    params = {
        'model': model_name,
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ],
        'temperature': 0.0,
        'top_p': 0.9,
        'max_tokens': 4000
    }
    params['stop'] = []
    params['logprobs'] = True

    resp = client.chat.completions.create(**params)
    
    choice = resp.choices[0]
    text = choice.message.content.strip()
    lp_info = None
    if hasattr(choice, 'logprobs'):
        lp_info = choice.logprobs
    usage = int(resp.usage.completion_tokens)
    return text, lp_info, usage


def probtree(retriever, entry, context_list, model_name="gpt-4o"):
    """Execute probtree algorithm for a single entry."""
    nodes = entry['nodes']
    # Build helper maps
    nodes_by_idx = {n['idx']: n for n in nodes}
    children_map = {n['idx']: n['sons'] for n in nodes}

    trace = []
    comb = []
    answers = {}

    def post_order(idx):
        # 1) Process children first
        for c in children_map.get(idx, []):
            post_order(c)

        node = nodes_by_idx[idx]
        q_text = node['question_text']
        is_leaf = node['is_leaf']

        # 2) OB retrieval
        query_tokens = bm25s.tokenize([q_text], stopwords="en", show_progress=False)
        top5_idx, _ = retriever.retrieve(query_tokens, k=3, show_progress=False)
        docs = [context_list[i] for i in top5_idx[0]]

        # 3) CB, OB prompts
        cb_prompt = CB_TEMPLATE.substitute(question=q_text)
        ob_prompt = OB_TEMPLATE.substitute(question=q_text, context="\n".join(docs))

        # 4) CA prompt only for non-leaf nodes
        if not is_leaf:
            ca_context = ""
            for c in children_map[idx]:
                ca_context += f"{nodes_by_idx[c]['question_text']} {answers[c]}\n"
            ca_prompt = CA_TEMPLATE.substitute(question=q_text, context=ca_context)

        # 5) LLM generation
        cb_out, cb_lp_info, _ = get_completion(cb_prompt, CB_SYSTEM_PROMPT, model_name)
        ob_out, ob_lp_info, _ = get_completion(ob_prompt, OB_SYSTEM_PROMPT, model_name)
        if not is_leaf:
            ca_out, ca_lp_info, _ = get_completion(ca_prompt, CA_SYSTEM_PROMPT, model_name)

        # 6) Calculate scores
        cb_tokens, cb_logprobs = lp_to_list(cb_lp_info)
        ob_tokens, ob_logprobs = lp_to_list(ob_lp_info)
        
        scores = {
            "cb": avg_logprob(cb_logprobs),
            "ob": avg_logprob(ob_logprobs),
        }
        if not is_leaf:
            ca_tokens, ca_logprobs = lp_to_list(ca_lp_info)
            prefix = "So the answer is:"
            if prefix in ca_out:
                prefix_idx = ca_out.find(prefix)
                post_logprobs = ca_logprobs[prefix_idx:] if prefix_idx < len(ca_logprobs) else ca_logprobs
            else:
                post_logprobs = ca_logprobs
            scores["ca"] = avg_logprob(post_logprobs)

        # 7) Select best strategy
        best = max(scores, key=scores.get)
        best_ans = {
            "cb": cb_out,
            "ob": ob_out,
            "ca": ca_out if not is_leaf else None
        }[best]

        # 8) Add first document to combination if OB selected
        if best == "ob" and docs:
            comb.append(docs[0])

        # 9) Record
        trace.append({
            "idx": idx,
            "strategy": best,
            "logprob": scores[best]
        })
        answers[idx] = best_ans

    # Execute from root
    root_idxs = [n['idx'] for n in nodes if n['parent'] is None]
    for root in root_idxs:
        post_order(root)

    final_answer = answers[root_idxs[0]] if root_idxs else ""
    return trace, final_answer, comb


def main():
    """Main function to run the probtree question answering generator."""
    parser = argparse.ArgumentParser(
        description="Probtree question answering using hierarchical question decomposition trees with OpenAI API"
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to tree JSONL file (output from make_tree)"
    )
    parser.add_argument(
        '--context', 
        type=str, 
        default="source/statute.jsonl",
        help="Path to context JSONL file"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Path to output JSONL for QA answers"
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="gpt-4o",
        help="OpenAI model name"
    )
    
    args = parser.parse_args()
    
    # Load tree data
    print(f"Loading tree data from {args.input}")
    data_list = load_jsonl(args.input)
    print(f"Loaded {len(data_list)} tree records")
    
    # Setup retriever
    print(f"Setting up retriever with context from {args.context}")
    retriever, context_list = setup_retriever(args.context)
    
    # Process each entry
    print("Processing entries...")
    results = []
    for entry in tqdm(data_list, desc="Processing entries"):
        trace, final_answer, comb = probtree(retriever, entry, context_list, args.model_name)
        
        result = {
            "question": entry.get('background', '') + entry.get('question', ''),
            "answer": final_answer,
            "trace": trace,
            "combinations": comb
        }
        results.append(result)
    
    # Write results
    print(f"Writing results to {args.output}")
    save_jsonl(results, args.output)
    print(f"Saved probtree answers to {args.output}")


if __name__ == '__main__':
    main()
