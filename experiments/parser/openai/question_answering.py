#!/usr/bin/env python3
"""
Question Answering Generator (OpenAI)

This script generates legal answers using selected statutory provisions with OpenAI API.
It processes legal questions and generates answers based on provided context.
"""

import argparse
import asyncio
import json
from typing import List, Dict, Any

from ..prompts.question_answering import SYSTEM_PROMPT, INSTRUCTION_PROMPT
from utils import get_completion_list


async def create_prompts(data_list: List[Dict[str, Any]]) -> List[str]:
    """Create prompts for all data items."""
    prompts = []
    
    for item in data_list:
        question = item['background'] + item['question']
        provisions = item.get('selected_provisions', [])
        
        # Create full prompt with system and instruction
        full_prompt = f"{SYSTEM_PROMPT}\n\n{INSTRUCTION_PROMPT.substitute(question=question, context_str='\n'.join(provisions))}"
        prompts.append(full_prompt)
    
    return prompts


def process_completions(completions: List[tuple]) -> List[str]:
    """Process OpenAI completions and extract answers."""
    answers = []
    
    for pred, _ in completions:
        # Extract answer portion
        answer = pred.split("Answer:")[-1].strip()
        answers.append(answer)
    
    return answers


async def main():
    """Main function to run the question answering generator."""
    parser = argparse.ArgumentParser(
        description="Legal QA using selected statutory provisions with OpenAI API"
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to JSONL with fields 'question' and 'selected_provisions'"
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
    parser.add_argument(
        '--max_parallel_calls', 
        type=int, 
        default=20,
        help="Maximum parallel API calls"
    )
    
    args = parser.parse_args()
    
    # Load retrieval-based input
    print(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} records")
    
    # Build prompts
    print("Creating prompts...")
    prompts = await create_prompts(data)
    
    # Generate answers
    print("Generating answers...")
    outputs = await get_completion_list(prompts, args.max_parallel_calls, args.model_name)
    
    # Process results
    print("Processing results...")
    answers = process_completions(outputs)
    
    # Write results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as fout:
        for item, provisions, answer in zip(data, [item.get('selected_provisions', []) for item in data], answers):
            record = {
                "background": item['background'],
                "question": item['question'],
                "answer": answer,
                "provisions": provisions
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Successfully processed {len(data)} records")


if __name__ == '__main__':
    asyncio.run(main())
