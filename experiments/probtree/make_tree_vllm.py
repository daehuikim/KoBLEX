#!/usr/bin/env python3
"""
Make Tree Generator (VLLM)

This script generates hierarchical question decomposition trees using VLLM.
It processes legal questions and creates tree structures for probtree question answering.
"""

import argparse
import json
import os
import re
import ast
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts.probtree_prompt import MAKE_TREE_SYSTEM_PROMPT, MAKE_TREE_QA_PROMPT
from utils import load_jsonl, save_jsonl, extract_json_from_text

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


class LlmGenerator:
    def __init__(self,
                 model_name: str,
                 dtype: str,
                 trust_remote_code: bool,
                 tensor_parallel_size: int,
                 temperature: float,
                 top_p: float,
                 max_tokens: int):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=1,
            seed=1
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=10000
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "right"

    def _generate(self, messages_list):
        prompts = []
        for messages in messages_list:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            prompt_str = self.tokenizer.decode(
                input_ids[0],
                skip_special_tokens=False
            )
            prompts.append(prompt_str)

        outputs_list = self.llm.generate(prompts, self.sampling_params)

        results = []
        result_logprobs = []
        for out in outputs_list:
            seq = out.outputs[0]
            text = seq.text

            # Extract JSON from text
            obj = extract_json_from_text(text)
            results.append(obj)

            # Calculate log probability for JSON part
            full_ids = seq.token_ids
            json_str = json.dumps(obj, ensure_ascii=False)
            json_ids = self.tokenizer.encode(json_str, add_special_tokens=False)

            def find_subseq(full, sub):
                for i in range(len(full) - len(sub) + 1):
                    if full[i:i+len(sub)] == sub:
                        return i
                return None

            start_idx = find_subseq(full_ids, json_ids)
            if start_idx is not None:
                json_lps = []
                for offset, tid in enumerate(json_ids):
                    lp_info = seq.logprobs[start_idx + offset].get(tid)
                    if lp_info is not None:
                        json_lps.append(lp_info.logprob)
                avg_lp = sum(json_lps) / len(json_lps) if json_lps else None
            else:
                avg_lp = None

            result_logprobs.append(avg_lp)

        return results, result_logprobs


def make_tree_prompts(data_list):
    """Create prompts for tree generation."""
    messages_list = []
    
    for rec in data_list:
        bg = rec["background"]
        q = rec["question"]
        messages = [
            {"role": "system", "content": MAKE_TREE_SYSTEM_PROMPT},
            {"role": "user", "content": MAKE_TREE_QA_PROMPT.substitute(background=bg, question=q)}
        ]
        messages_list.append(messages)
    
    return messages_list


def main():
    """Main function to run the make tree generator."""
    parser = argparse.ArgumentParser(
        description="Generate hierarchical question decomposition trees using VLLM"
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', 
        type=str, 
        default="source/koblex.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Path to output JSONL file (use $TREE placeholder for tree file)"
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
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
        default=2048,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}")
    data_list = load_jsonl(args.input)
    print(f"Loaded {len(data_list)} records")
    
    # Initialize model
    print("Initializing model...")
    llm_generator = LlmGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Create prompts
    print("Creating prompts...")
    messages_list = make_tree_prompts(data_list)
    
    # Generate trees
    print("Generating trees...")
    generated, logprobs = llm_generator._generate(messages_list)
    
    # Prepare output records
    all_records = []
    for out, lp, rec in zip(generated, logprobs, data_list):
        all_records.append({
            "answer": out,
            "qd_logprob": lp,
            "background": rec["background"].strip()
        })
    
    # Write results
    print(f"Writing results to {args.output}")
    save_jsonl(all_records, args.output)
    print(f"Saved tree data to {args.output}")


if __name__ == '__main__':
    main()

