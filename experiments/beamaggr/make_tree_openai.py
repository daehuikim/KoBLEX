#!/usr/bin/env python3
"""
Make Tree Generator (OpenAI)

This script generates hierarchical question decomposition trees using OpenAI API.
It processes legal questions and creates tree structures for beam aggregation question answering.
"""

import argparse
import asyncio
import json
import os
from typing import List, Dict, Any

from prompts.beamaggr_prompt import MAKE_TREE_SYSTEM_PROMPT, MAKE_TREE_QA_PROMPT
from utils import load_jsonl, save_jsonl, extract_json_from_text, get_completion_list


async def create_prompts(data_list: List[Dict[str, Any]]) -> List[str]:
    """Create prompts for all data items."""
    prompts = []
    
    for item in data_list:
        background = item.get('background', '')
        question = item.get('question', '')
        
        # Create full prompt with system and instruction
        full_prompt = f"{MAKE_TREE_SYSTEM_PROMPT}\n\n{MAKE_TREE_QA_PROMPT.substitute(background=background, question=question)}"
        prompts.append(full_prompt)
    
    return prompts


def process_completions(completions: List[tuple]) -> List[Dict[str, Any]]:
    """Process OpenAI completions and extract JSON objects."""
    results = []
    
    for pred, _ in completions:
        # Extract JSON object from text
        obj = extract_json_from_text(pred)
        results.append(obj)
    
    return results


async def main():
    """Main function to run the make tree generator."""
    parser = argparse.ArgumentParser(
        description="Generate hierarchical question decomposition trees using OpenAI API"
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
        default="gpt-4o",
        help="OpenAI model name"
    )
    parser.add_argument(
        '--max_parallel_calls', 
        type=int, 
        default=20,
        help="Maximum parallel API calls"
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}")
    data_list = load_jsonl(args.input)
    print(f"Loaded {len(data_list)} records")
    
    # Build prompts
    print("Creating prompts...")
    prompts = await create_prompts(data_list)
    
    # Generate trees
    print("Generating trees...")
    outputs = await get_completion_list(prompts, args.max_parallel_calls, args.model_name)
    
    # Process results
    print("Processing results...")
    generated = process_completions(outputs)
    
    # Prepare output records
    all_records = []
    for out, rec in zip(generated, data_list):
        all_records.append({
            "answer": out,
            "qd_logprob": None,  # OpenAI doesn't provide logprobs in the same way
            "background": rec["background"].strip()
        })
    
    # Write results
    print(f"Writing results to {args.output}")
    save_jsonl(all_records, args.output)
    print(f"Saved tree data to {args.output}")


if __name__ == '__main__':
    asyncio.run(main())
