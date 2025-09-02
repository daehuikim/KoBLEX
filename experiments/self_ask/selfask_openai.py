#!/usr/bin/env python3
"""
Self-Ask Legal Reasoning (OpenAI)

This script implements self-ask reasoning for legal questions using OpenAI API.
It follows a chain-of-thought approach with intermediate questions and answers.
"""

import argparse
import logging
from typing import List, Tuple
from tqdm import tqdm
from openai import OpenAI

from prompts.selfask_prompt import SYSTEM_PROMPT, ANSWER_PROMPT
from utils import (
    load_api_key,
    extract_answer,
    extract_question,
    load_jsonl,
    save_jsonl,
    get_statute_retriever
)


class OpenAIGenerator:
    """OpenAI API generator for self-ask reasoning."""
    
    def __init__(self, model_name: str, temperature: float, top_p: float, max_tokens: int):
        """Initialize the OpenAI generator."""
        self.client = OpenAI(api_key=load_api_key())
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tok_gen = 0
        self.llm_call = 0
    
    def get_completion(self, prompt_text: str, stop: List[str] = None) -> Tuple[str, int]:
        """Get completion from OpenAI API."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=stop or []
        )
        
        usage = int(resp.usage.completion_tokens)
        self.tok_gen += usage
        self.llm_call += 1
        
        return resp.choices[0].message.content.strip(), usage


def get_answer(llm_gen: OpenAIGenerator, retriever, question: str, context_list: List[str]) -> Tuple[str, str]:
    """Get answer based on legal provision."""
    import bm25s
    
    q_tokens = bm25s.tokenize([question])
    idxs, _ = retriever.retrieve(q_tokens, k=1)
    provision = context_list[idxs[0][0]]
    
    prompt = ANSWER_PROMPT.substitute(question=question, provision=provision)
    ans, _ = llm_gen.get_completion(
        prompt_text=prompt,
        stop=[".", "!", "?"]
    )
    
    return ans, provision


def self_ask(initial_prompt: str, llm_gen: OpenAIGenerator, retriever, context_list: List[str]) -> Tuple[str, str, List[str]]:
    """Perform self-ask reasoning."""
    prompt = initial_prompt
    trace = ""
    combination_list = []
    
    count = 0
    while count < 5:
        response, _ = llm_gen.get_completion(
            prompt_text=prompt,
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
        response, _ = llm_gen.get_completion(
            prompt_text=prompt,
            stop=["\n"]
        )
        trace += response
    
    return trace, response, combination_list


def main():
    """Main function to run self-ask reasoning."""
    parser = argparse.ArgumentParser(
        description="Self-ask legal reasoning using OpenAI API"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.01,
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
    print("Initializing OpenAI generator...")
    llm_gen = OpenAIGenerator(
        model_name=args.model_name,
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
            'answer': clean_ans,
            'full_trace': trace.split("<Query>")[-1].strip() if "<Query>" in trace else trace
        }
        results.append(record)
    
    # Save results
    print(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    # Save statistics
    stats_file = args.output_file.replace('.jsonl', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Total LLM Call: {llm_gen.llm_call}\n")
        f.write(f"Total generated tokens: {llm_gen.tok_gen}\n")
    
    print(f"Total LLM Call: {llm_gen.llm_call}")
    print(f"Total generated tokens: {llm_gen.tok_gen}")
    print(f"Results saved to {args.output_file}")
    print(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()