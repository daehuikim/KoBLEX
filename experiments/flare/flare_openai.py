#!/usr/bin/env python3
"""
FLARE Legal Reasoning (OpenAI)

This script implements FLARE (Forward-Looking Active REtrieval) 
reasoning for legal questions using OpenAI API.
"""

import argparse
import logging
from typing import List, Tuple
from tqdm import tqdm
from openai import OpenAI

from prompts.flare_prompt import SYSTEM_PROMPT, ST_STOP, Q_FOR_PROMPT
from utils import (
    load_api_key,
    split_sentences,
    load_jsonl,
    save_jsonl,
    get_statute_retriever
)


class OpenAIGenerator:
    """OpenAI API generator for FLARE reasoning."""
    
    def __init__(self, model_name: str, temperature: float, top_p: float, max_tokens: int):
        """Initialize the OpenAI generator."""
        self.client = OpenAI(api_key=load_api_key())
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tok_gen = 0
        self.llm_call = 0
    
    def get_completion(self, prompt_text: str, stop: List[str] = None, logprobs: bool = False) -> Tuple[str, any, int]:
        """Get completion from OpenAI API."""
        params = {
            'model': self.model_name,
            'messages': [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens
        }
        
        if stop:
            params['stop'] = stop
        if logprobs:
            params['logprobs'] = True
        
        resp = self.client.chat.completions.create(**params)
        
        choice = resp.choices[0]
        text = choice.message.content.strip()
        lp_info = None
        if logprobs and hasattr(choice, 'logprobs'):
            lp_info = choice.logprobs
        usage = int(resp.usage.completion_tokens)
        
        self.tok_gen += usage
        self.llm_call += 1
        
        return text, lp_info, usage


def decide_retrieval(tokens: List[str], logprobs: List[float], token_threshold: float = -0.70) -> Tuple[str, bool]:
    """Decide which tokens to mask based on logprobs."""
    masked = []
    scores = []
    flag = False
    
    for tok, lp in zip(tokens, logprobs):
        scores.append(lp)
        if lp is not None and lp < token_threshold:
            masked.append("_")
            flag = True
        else:
            masked.append(tok)
    
    print(f"Average score: {sum(scores)/len(scores)}")
    return "".join(masked), flag


def flare(llm_gen: OpenAIGenerator, retriever, question: str, context_list: List[str], k: int = 1) -> Tuple[str, str, List[str], float]:
    """Perform FLARE reasoning."""
    import bm25s
    
    # Initial context retrieval
    q_tokens = bm25s.tokenize([question])
    idxs, _ = retriever.retrieve(q_tokens, k=k)
    combination_list = [context_list[idxs[0][0]]]
    reasoning_list = []
    question_str = f"{question}"
    ret_count = 0

    max_steps = 5
    for i in range(max_steps):
        cur_prompt = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
        
        ret_text, lp_info, _ = llm_gen.get_completion(cur_prompt, stop=ST_STOP, logprobs=True)
        reasoning_list.append(ret_text.strip())
        
        if "So the answer is:" in ret_text:
            break

        # Token masking and query generation
        if lp_info and hasattr(lp_info, 'content'):
            tokens = lp_info.content[0].tokens
            logprobs = lp_info.content[0].logprobs
            
            masked_text, should_retrieve = decide_retrieval(tokens, logprobs)
            
            if should_retrieve:
                query_prompt = Q_FOR_PROMPT.format(question=question_str, query=masked_text)
                new_query_text, _, _ = llm_gen.get_completion(query_prompt)
                
                q_tokens = bm25s.tokenize([new_query_text])
                idxs, _ = retriever.retrieve(q_tokens, k=k)
                combination_list.append(context_list[idxs[0][0]])
                ret_count += 1

    # Ensure final answer
    combined = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
    
    if not reasoning_list or "So the answer is:" not in reasoning_list[-1]:
        combined += " So the answer is:"
        final_text, _, _ = llm_gen.get_completion(combined)
        reasoning_list[-1] = final_text.strip()

    full_trace = combined + "\n" + reasoning_list[-1]
    answer = reasoning_list[-1].split("So the answer is:")[-1].strip()
    ret_ratio = ret_count / len(reasoning_list) if reasoning_list else 0
    
    return full_trace, answer, combination_list, ret_ratio


def main():
    """Main function to run FLARE reasoning."""
    parser = argparse.ArgumentParser(
        description="FLARE legal reasoning using OpenAI API"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
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
    parser.add_argument(
        "--token_threshold", 
        type=float, 
        default=-0.70,
        help="Token threshold for masking"
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
        default="flare_results.jsonl",
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
    total_ret_ratio = 0
    
    for item in tqdm(data_list, desc="Processing"):
        question = item['background'] + item['question']
        
        trace, ans, comb, ret_ratio = flare(llm_gen, retriever, question, context_list, k=args.k)
        clean_ans = ans.split("So the final answer is:")[-1].strip()
        
        record = {
            'question': question,
            'answer': clean_ans,
            'full_trace': trace.split("<Query>")[-1].strip() if "<Query>" in trace else trace
        }
        results.append(record)
        total_ret_ratio += ret_ratio
    
    # Calculate average retrieval ratio
    avg_ratio = total_ret_ratio / len(data_list) if data_list else 0
    
    # Save results
    print(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    # Save statistics
    stats_file = args.output_file.replace('.jsonl', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Total LLM Call: {llm_gen.llm_call}\n")
        f.write(f"Total generated tokens: {llm_gen.tok_gen}\n")
        f.write(f"Average retrieval ratio: {avg_ratio:.4f}\n")
    
    print(f"Total LLM Call: {llm_gen.llm_call}")
    print(f"Total generated tokens: {llm_gen.tok_gen}")
    print(f"Average retrieval ratio: {avg_ratio:.4f}")
    print(f"Results saved to {args.output_file}")
    print(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()