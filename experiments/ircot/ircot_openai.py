#!/usr/bin/env python3
"""
IRCoT Legal Reasoning (OpenAI)

This script implements IRCoT (Interleaved Retrieval and Chain-of-Thought) 
reasoning for legal questions using OpenAI API.
"""

import argparse
import logging
from typing import List,Tuple
from tqdm import tqdm
from openai import OpenAI

from prompts.ircot_prompt import SYSTEM_PROMPT, ST_STOP
from utils import (
    load_api_key,
    load_jsonl,
    save_jsonl,
    get_statute_retriever
)


class OpenAIGenerator:
    """OpenAI API generator for IRCoT reasoning."""
    
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


def ircot(llm_gen: OpenAIGenerator, retriever, question: str, context_list: List[str], k: int = 1) -> Tuple[str, str, List[str]]:
    """Perform IRCoT reasoning."""
    import bm25s
    
    # Initial context retrieval
    q_tokens = bm25s.tokenize([question])
    idxs, _ = retriever.retrieve(q_tokens, k=k)
    combination_list = [context_list[idxs[0][0]]]
    reasoning_list = []
    question_str = f"{question}"

    max_steps = 5
    for i in range(max_steps):
        cur_prompt = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
        
        ret_text, _ = llm_gen.get_completion(cur_prompt, stop=ST_STOP)
        reasoning_list.append(ret_text.strip())
        
        if "So the final answer is:" in ret_text:
            break

        # Retrieve next context based on model output
        out_tokens = bm25s.tokenize([ret_text], show_progress=False)
        res, _ = retriever.retrieve(out_tokens, k=k, show_progress=False)
        combination_list.append(context_list[res[0][0]])

    # Ensure final answer
    combined_prompt = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
    
    if not reasoning_list or "So the final answer is:" not in reasoning_list[-1]:
        combined_prompt += " So the final answer is:"
        final_text, _ = llm_gen.get_completion(combined_prompt, stop=None)
        marker = reasoning_list[-1]
        reasoning_list.append(final_text.split(marker)[-1])
        full_trace = combined_prompt + "\n" + reasoning_list[-1]
    else:
        full_trace = combined_prompt
    
    answer = reasoning_list[-1].split("So the final answer is:")[-1].strip()
    return full_trace, answer, combination_list


def main():
    """Main function to run IRCoT reasoning."""
    parser = argparse.ArgumentParser(
        description="IRCoT legal reasoning using OpenAI API"
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
    parser.add_argument(
        "--k", 
        type=int, 
        default=1,
        help="Number of contexts to retrieve each step"
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
        default="ircot_results.jsonl",
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
        
        trace, ans, comb = ircot(llm_gen, retriever, question, context_list, k=args.k)
        
        record = {
            'question': question,
            'answer': ans
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