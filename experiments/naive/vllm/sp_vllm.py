#!/usr/bin/env python3
"""
Simple Prompt Question Answering Generator (VLLM)

This script generates legal answers using simple prompts with VLLM.
It processes legal questions and generates direct answers without step-by-step reasoning.
"""

import argparse
import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from ..prompts.naive_prompt import SP_SYSTEM_PROMPT, SP_QA_PROMPT
from utils import load_jsonl

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
            seed=0
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"

    def _generate(self, prompts):
        # Prepare batch requests
        requests = []
        for idx, prompt in enumerate(prompts):
            messages = [
                {"role": "system", "content": SP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            inp = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            token_ids = inp[0]
            full_input = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            requests.append(full_input)
        
        # Invoke vLLM
        responses = self.llm.generate(requests, self.sampling_params)
        
        results = []
        for resp in responses:
            output = resp.outputs[0]
            text = output.text
            token_lps = output.logprobs
            temp = []
            for step in token_lps:
                _, l = next(iter(step.items()))
                temp.append(l.logprob)
            
            avg_logprob = sum(temp) / len(temp)
            results.append((text, avg_logprob))
        return results


def make_qa_prompts(data):
    """Create prompts for question answering."""
    prompts = [SP_QA_PROMPT.substitute(question=data['background'] + data['question'])]
    return prompts


def main():
    """Main function to run the simple prompt question answering generator."""
    parser = argparse.ArgumentParser(
        description="Legal QA using simple prompts with VLLM"
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to JSONL with fields 'question' and 'background'"
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
        default="Qwen/Qwen3-32B",
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
        help="Temperature for sampling"
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.9,
        help="Top-p for sampling"
    )
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=4000,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}")
    data_list = load_jsonl(args.input)
    print(f"Loaded {len(data_list)} records")
    
    # Create all prompts
    all_prompts = []
    for entry in data_list:
        prompts = make_qa_prompts(entry)
        all_prompts.extend(prompts)
    
    # Initialize model
    llm_gen = LlmGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Generate answers
    print("Generating answers...")
    results = llm_gen._generate(all_prompts)
    
    # Write results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as fw:
        for entry, prediction in zip(data_list, results):
            question_text = entry['background'] + entry['question']
            answer = prediction[0].split("Answer:")[-1].strip()
            record = {
                "question": question_text,
                "answer": answer
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print("Done!")


if __name__ == "__main__":
    main()
