#!/usr/bin/env python3
"""
Parametric Provision Generator

This script generates parametric provisions using a language model.
It processes legal questions and extracts relevant statutory provisions.
"""

import argparse
import os
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from ..prompts.parametric_provision import SYSTEM_PROMPT, INSTRUCTION_PROMPT
from utils import (
    extract_first_list, 
    parse_list_block, 
    escape_quotes, 
    load_jsonl, 
    save_jsonl
)


class LlmGenerator:
    """Language model generator for parametric provision extraction."""
    
    def __init__(self, model_name: str, dtype: str, trust_remote_code: bool,
                 tensor_parallel_size: int, temperature: float, top_p: float, 
                 max_tokens: int):
        """Initialize the LLM generator with specified parameters."""
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
    
    def generate(self, prompts: List[str]) -> List[Any]:
        """Generate completions for the given prompts."""
        return self.llm.generate(prompts, self.sampling_params)


def create_prompts(data_list: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> List[str]:
    """Create prompts for all data items."""
    all_prompts = []
    
    for item in data_list:
        background = item['background']
        question = item['question']
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INSTRUCTION_PROMPT.substitute(
                background=background, question=question
            )}
        ]
        
        inp = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt", 
            enable_thinking=False
        )
        temp_prompt = tokenizer.decode(inp[0], skip_special_tokens=False)
        all_prompts.append(temp_prompt)
    
    return all_prompts


def process_completions(completions: List[Any]) -> List[List[str]]:
    """Process LLM completions and extract results."""
    results = []
    
    for out in completions:
        raw = out.outputs[0].text
        block = extract_first_list(raw)
        
        if block is None:
            results.append([])
            print(f"Warning: Could not extract list from: {raw}")
            continue
        
        subs = parse_list_block(block)
        subs = escape_quotes(subs)
        
        if not isinstance(subs, list):
            subs = [subs]
        
        results.append(subs)
    
    return results


def main():
    """Main function to run the parametric provision generator."""
    parser = argparse.ArgumentParser(
        description="Generate parametric provisions using language model"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="koleqa-real/filtered_data.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="refactor-test/results/refactored.jsonl",
        help="Output JSONL file path"
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
        default=4000,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_file}")
    data_list = load_jsonl(args.input_file)
    print(f"Loaded {len(data_list)} records")
    
    # Initialize LLM generator
    print("Initializing LLM generator...")
    llm_gen = LlmGenerator(
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
    all_prompts = create_prompts(data_list, llm_gen.tokenizer)
    
    # Generate completions
    print("Generating completions...")
    completions = llm_gen.generate(all_prompts)
    
    # Process results
    print("Processing results...")
    results = process_completions(completions)
    
    # Add results to data and save
    for item, result in zip(data_list, results):
        item["subs"] = result
    
    print(f"Saving results to {args.output_file}")
    save_jsonl(data_list, args.output_file)
    
    print(f"Successfully processed {len(data_list)} records")


if __name__ == '__main__':
    main()
