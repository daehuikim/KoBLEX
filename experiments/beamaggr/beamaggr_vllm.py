#!/usr/bin/env python3
"""
Beam Aggregation Question Answering Generator (VLLM)

This script performs beam aggregation question answering using VLLM.
It uses tree structures to decompose complex questions and answer them using beam search with multiple strategies.
"""

import argparse
import json
import os
import math
import itertools
from typing import List, Dict, Any, Tuple
import bm25s
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from prompts.beamaggr_prompt import (
    CB_SYSTEM_PROMPT, OB_SYSTEM_PROMPT,
    CB_TEMPLATE, OB_TEMPLATE
)
from utils import load_jsonl, save_jsonl, setup_retriever, avg_logprob, softmax_probs

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


class LlmGenerator:
    def __init__(
        self,
        model_name: str,
        dtype: str,
        trust_remote_code: bool,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_tokens: int
    ):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=0,
            logprobs=1
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=10000
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.eos_token_id = 151645
        self.tokenizer.pad_token_id = 151644
        self.tokenizer.padding_side = "right"

    def _generate(self, prompt, system_prompt):
        params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=self.sampling_params.max_tokens,
            seed=self.sampling_params.seed,
            logprobs=self.sampling_params.logprobs
        )
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        inp = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        full_input = self.tokenizer.decode(inp[0], skip_special_tokens=False)
        result = self.llm.generate([full_input], params)[0]
        return result.outputs[0].text, result.outputs[0].logprobs


def beamaggr(llm, retriever, entry, context_list):
    """Execute beam aggregation algorithm for a single entry."""
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

        # 4) LLM generation
        cb_out, cb_lps = llm._generate(cb_prompt, CB_SYSTEM_PROMPT)
        ob_out, ob_lps = llm._generate(ob_prompt, OB_SYSTEM_PROMPT)

        # 5) Calculate scores
        scores = {
            "cb": avg_logprob(cb_lps),
            "ob": avg_logprob(ob_lps),
        }

        # 6) Beam search with softmax probabilities
        probs = softmax_probs(scores)
        
        # 7) Select best strategy based on beam search
        best = max(probs, key=probs.get)
        best_ans = {
            "cb": cb_out,
            "ob": ob_out,
        }[best]

        # 8) Add first document to combination if OB selected
        if best == "ob" and docs:
            comb.append(docs[0])

        # 9) Record
        trace.append({
            "idx": idx,
            "strategy": best,
            "logprob": scores[best],
            "probability": probs[best]
        })
        answers[idx] = best_ans

    # Execute from root
    root_idxs = [n['idx'] for n in nodes if n['parent'] is None]
    for root in root_idxs:
        post_order(root)

    final_answer = answers[root_idxs[0]] if root_idxs else ""
    return trace, final_answer, comb


def main():
    """Main function to run the beam aggregation question answering generator."""
    parser = argparse.ArgumentParser(
        description="Beam aggregation question answering using hierarchical question decomposition trees with VLLM"
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
        default="",
        help="Model name for VLLM"
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default="auto",
        help="Data type for model"
    )
    parser.add_argument(
        '--trust_remote_code', 
        action='store_true',
        help="Trust remote code"
    )
    parser.add_argument(
        '--tensor_parallel_size', 
        type=int, 
        default=2,
        help="Tensor parallel size"
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=4000,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load tree data
    print(f"Loading tree data from {args.input}")
    data_list = load_jsonl(args.input)
    print(f"Loaded {len(data_list)} tree records")
    
    # Setup retriever
    print(f"Setting up retriever with context from {args.context}")
    retriever, context_list = setup_retriever(args.context)
    
    # Initialize model
    print("Initializing model...")
    llm = LlmGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Process each entry
    print("Processing entries...")
    results = []
    for entry in tqdm(data_list, desc="Processing entries"):
        trace, final_answer, comb = beamaggr(llm, retriever, entry, context_list)
        
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
    print(f"Saved beam aggregation answers to {args.output}")


if __name__ == '__main__':
    main()
