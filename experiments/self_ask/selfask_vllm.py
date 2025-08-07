#!/usr/bin/env python3
"""
Self-Ask Legal Reasoning (vLLM)

This script implements self-ask reasoning for legal questions using vLLM.
It follows a chain-of-thought approach with intermediate questions and answers.
"""

import argparse
import logging
import os
import torch
import numpy as np
import random
from typing import List, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts.selfask_prompt import SYSTEM_PROMPT, ANSWER_PROMPT
from utils import (
    extract_answer,
    extract_question,
    load_jsonl,
    save_jsonl,
    get_statute_retriever
)


class LlmGenerator:
    """vLLM generator for self-ask reasoning."""
    
    def __init__(self, model_name: str, dtype: str, trust_remote_code: bool,
                 tensor_parallel_size: int, temperature: float, top_p: float, 
                 max_tokens: int):
        """Initialize the LLM generator with specified parameters."""
        # Set reproducibility
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)
        
        # vLLM reproducibility settings
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=0
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
    
    def _generate(self, prompt: str, stop: List[str] = None) -> str:
        """Generate text with vLLM."""
        params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=self.sampling_params.max_tokens,
            seed=self.sampling_params.seed,
            stop=stop
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        inp = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        full_input = self.tokenizer.decode(inp[0], skip_special_tokens=False)
        result = self.llm.generate([full_input], params)[0]
        return result.outputs[0].text


def get_answer(llm_gen: LlmGenerator, retriever, question: str, context_list: List[str]) -> Tuple[str, str]:
    """Get answer based on legal provision."""
    import bm25s
    
    q_tokens = bm25s.tokenize([question])
    idxs, _ = retriever.retrieve(q_tokens, k=1)
    provision = context_list[idxs[0][0]]
    
    prompt = ANSWER_PROMPT.substitute(question=question, provision=provision)
    ans = llm_gen._generate(prompt, stop=[".", "!", "?"])
    
    return ans, provision


def self_ask(initial_prompt: str, llm_gen: LlmGenerator, retriever, context_list: List[str]) -> Tuple[str, str, List[str]]:
    """Perform self-ask reasoning."""
    prompt = initial_prompt
    trace = ""
    combination_list = []
    
    count = 0
    while count < 5:
        response = llm_gen._generate(
            prompt,
            stop=["\nFollow up:", "\nSo the final answer is:"]
        )
        trace += response
        
        if "Are follow up questions needed here: Yes" in response:
            question = extract_question(response)
            combination_list.append(question)
            
            answer, provision = get_answer(llm_gen, retriever, question, context_list)
            combination_list.append(provision)
            
            prompt += response + f"\nIntermediate answer: {answer}\n"
            continue
        
        if "So the final answer is:" in response:
            break
        
        count += 1
    
    if "So the final answer is:" not in response:
        prompt += "\nSo the final answer is:"
        response = llm_gen._generate(prompt, stop=["\n"])
        trace += response
    
    return trace, response, combination_list


def main():
    """Main function to run self-ask reasoning."""
    parser = argparse.ArgumentParser(
        description="Self-ask legal reasoning using vLLM"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen3-8B",
        help="Model name or path"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="auto",
        help="Model data type"
    )
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true",
        help="Trust remote code"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="Tensor parallel size"
    )
    
    # Generation arguments
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2000,
        help="Maximum tokens to generate"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="source/koblex.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="selfask_results.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--statute_file", 
        type=str, 
        default="source/statute.jsonl",
        help="Statute JSONL file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load data
    print(f"Loading data from {args.input_file}")
    data_list = load_jsonl(args.input_file)
    print(f"Loaded {len(data_list)} records")
    
    # Setup retriever
    print("Setting up statute retriever...")
    context_list, retriever = get_statute_retriever(args.statute_file)
    
    # Initialize LLM generator
    print("Initializing vLLM generator...")
    llm_gen = LlmGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Process each item
    results = []
    for item in tqdm(data_list, desc="Processing"):
        question = item['background'] + item['question']
        
        trace, resp, comb = self_ask(question, llm_gen, retriever, context_list)
        clean_ans = extract_answer(resp)
        
        record = {
            'question': question,
            'answers': [{'answer': clean_ans, 'logp': 0, 'comb': comb}],
            'full_trace': trace.split("<Query>")[-1].strip() if "<Query>" in trace else trace
        }
        results.append(record)
    
    # Save results
    print(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
