#!/usr/bin/env python3
"""
Simple Prompt with One-time Retrieval Question Answering Generator (OpenAI)

This script generates legal answers using simple prompts with one-time retrieval using OpenAI API.
It processes legal questions, retrieves relevant context, and generates direct answers.
"""

import argparse
import asyncio
import json
from typing import List, Dict, Any
import bm25s
import Stemmer

from ..prompts.naive_prompt import SP_OR_SYSTEM_PROMPT, SP_OR_QA_PROMPT
from utils import get_completion_list, load_jsonl


def setup_retriever(context_file: str):
    """Setup BM25 retriever with context data."""
    context_list = []
    doc_ids = []
    
    with open(context_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            context_list.append(item['hierarchy'] + item['content'])
            doc_ids.append(i)
    
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(context_list, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    
    return retriever, context_list


async def create_prompts(data_list: List[Dict[str, Any]], retriever, context_list: List[str], n_hops: int = None) -> List[str]:
    """Create prompts for all data items with retrieval."""
    prompts = []
    
    stemmer = Stemmer.Stemmer("english")
    
    for entry in data_list:
        temp_question = entry['background'] + entry['question']
        
        # Use n_hops from data if not specified
        n = n_hops if n_hops is not None else entry.get('n_hops', 3)
        
        question_tokens = bm25s.tokenize([temp_question], stemmer=stemmer)
        res, _ = retriever.retrieve(question_tokens, k=n)
        
        context_strs = []
        for item in res[0]:
            context_strs.append(context_list[item])
        
        context_str = "\n".join(context_strs)
        
        # Create full prompt with system and instruction
        full_prompt = f"{SP_OR_SYSTEM_PROMPT}\n\n{SP_OR_QA_PROMPT.substitute(question=temp_question, context_str=context_str)}"
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
    """Main function to run the simple prompt with one-time retrieval question answering generator."""
    parser = argparse.ArgumentParser(
        description="Legal QA using simple prompts with one-time retrieval using OpenAI API"
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to JSONL with fields 'question' and 'background'"
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
    parser.add_argument(
        '--max_parallel_calls', 
        type=int, 
        default=20,
        help="Maximum parallel API calls"
    )
    parser.add_argument(
        '--n_hops', 
        type=int, 
        default=None,
        help="Number of context passages to retrieve (overrides data n_hops)"
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}")
    data_list = load_jsonl(args.input)
    print(f"Loaded {len(data_list)} records")
    
    # Setup retriever
    print(f"Setting up retriever with context from {args.context}")
    retriever, context_list = setup_retriever(args.context)
    
    # Build prompts
    print("Creating prompts...")
    prompts = await create_prompts(data_list, retriever, context_list, args.n_hops)
    
    # Generate answers
    print("Generating answers...")
    outputs = await get_completion_list(prompts, args.max_parallel_calls, args.model_name)
    
    # Process results
    print("Processing results...")
    answers = process_completions(outputs)
    
    # Write results
    print(f"Writing results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as fw:
        for entry, answer in zip(data_list, answers):
            record = {
                "question": entry['background'] + entry['question'],
                "answer": answer
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Saved detailed answers to {args.output}")


if __name__ == '__main__':
    asyncio.run(main())
