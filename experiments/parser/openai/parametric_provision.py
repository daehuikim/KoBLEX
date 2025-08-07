#!/usr/bin/env python3
"""
Parametric Provision Generator (OpenAI)

This script generates parametric provisions using OpenAI API.
It processes legal questions and extracts relevant statutory provisions.
"""

import argparse
import asyncio
from typing import List, Dict, Any

from ..prompts.parametric_provision import SYSTEM_PROMPT, INSTRUCTION_PROMPT
from utils import (
    extract_first_list, 
    parse_list_block, 
    escape_quotes, 
    load_jsonl, 
    save_jsonl,
    get_completion_list
)


async def create_prompts(data_list: List[Dict[str, Any]]) -> List[str]:
    """Create prompts for all data items."""
    all_prompts = []
    
    for item in data_list:
        background = item['background']
        question = item['question']
        
        # Create full prompt with system and instruction
        full_prompt = f"{SYSTEM_PROMPT}\n\n{INSTRUCTION_PROMPT.substitute(background=background, question=question)}"
        all_prompts.append(full_prompt)
    
    return all_prompts


def process_completions(completions: List[tuple]) -> List[List[str]]:
    """Process OpenAI completions and extract results."""
    results = []
    
    for pred, _ in completions:
        block = extract_first_list(pred)
        
        if block is None:
            results.append([])
            print(f"Warning: Could not extract list from: {pred}")
            continue
        
        subs = parse_list_block(block)
        subs = escape_quotes(subs)
        
        if not isinstance(subs, list):
            subs = [subs]
        
        results.append(subs)
    
    return results


async def main():
    """Main function to run the parametric provision generator."""
    parser = argparse.ArgumentParser(
        description="Generate parametric provisions using OpenAI API"
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
        default="parametric_provisions.jsonl",
        help="Output JSONL file path"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--max_parallel_calls", 
        type=int, 
        default=20,
        help="Maximum parallel API calls"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}")
    data_list = load_jsonl(args.input_file)
    print(f"Loaded {len(data_list)} records")
    
    # Create prompts
    print("Creating prompts...")
    all_prompts = await create_prompts(data_list)
    
    # Generate completions
    print("Generating completions...")
    completions = await get_completion_list(all_prompts, args.max_parallel_calls, args.model_name)
    
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
    asyncio.run(main())
