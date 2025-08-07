#!/usr/bin/env python3
"""
FLARE Legal Reasoning (ExaOne)

This script implements FLARE (Forward-Looking Active REtrieval) 
reasoning for legal questions using ExaOne model with vLLM.
"""

import argparse
import logging
from typing import List, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts.flare_prompt import SYSTEM_PROMPT, ST_STOP, Q_FOR_PROMPT
from utils import (
    load_jsonl,
    save_jsonl,
    get_statute_retriever
)


class LlmGenerator:
    """vLLM generator for FLARE reasoning with ExaOne."""
    
    def __init__(self, model_name: str, dtype: str, trust_remote_code: bool,
                 tensor_parallel_size: int, temperature: float, top_p: float, 
                 max_tokens: int):
        """Initialize the LLM generator with specified parameters."""
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=0,
            logprobs=1
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
    
    def _generate(self, prompt: str, stop: List[str] = None) -> Tuple[str, any]:
        """Generate text with vLLM."""
        params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=self.sampling_params.max_tokens,
            seed=self.sampling_params.seed,
            logprobs=self.sampling_params.logprobs,
            stop=stop
        )
        
        message = [{"role": "user", "content": prompt}]
        inp = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        full_input = self.tokenizer.decode(inp[0], skip_special_tokens=False)
        result = self.llm.generate([full_input], params)[0]
        return result.outputs[0].text, result.outputs[0].logprobs

    def _generate_query(self, prompt: str) -> str:
        """Generate query for uncertain token masking."""
        params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=self.sampling_params.max_tokens,
            seed=self.sampling_params.seed,
            logprobs=self.sampling_params.logprobs
        )
        
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        inp = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        full_input = self.tokenizer.decode(inp[0], skip_special_tokens=False)
        result = self.llm.generate([full_input], params)[0]
        text = result.outputs[0].text
        return text.split("New Query: ")[-1].strip()


def decide_retrieval(llm_gen: LlmGenerator, token_lps: List[float], token_threshold: float = -1.5) -> Tuple[str, bool]:
    """Decide which tokens to mask based on logprobs."""
    masked_tokens = []
    mask_token = "_"
    
    for lp in token_lps:
        if lp is not None and lp < token_threshold:
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append("")
    
    return "".join(masked_tokens), any(lp is not None and lp < token_threshold for lp in token_lps)


def flare(llm_gen: LlmGenerator, retriever, question: str, context_list: List[str], k: int = 1) -> Tuple[str, str, List[str], float]:
    """Perform FLARE reasoning."""
    import bm25s
    
    # Initialize
    q_tokens = bm25s.tokenize([question])
    res, _ = retriever.retrieve(q_tokens, k=k)
    combination_list = [context_list[res[0][0]]]
    reasoning_list = []
    question_str = f"{question}"
    ret_count = 0

    max_steps = 8
    for _ in range(max_steps):
        cur_prompt = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
        
        ret_text, logprobs = llm_gen._generate(cur_prompt, stop=ST_STOP)
        reasoning_list.append(ret_text.strip())
        
        if "So the answer is:" in ret_text:
            break

        # Token masking and query generation
        if logprobs:
            token_lps = [lp for lp in logprobs if lp is not None]
            masked_text, should_retrieve = decide_retrieval(llm_gen, token_lps)
            
            if should_retrieve:
                query_prompt = Q_FOR_PROMPT.format(question=question_str, query=masked_text)
                new_query_text = llm_gen._generate_query(query_prompt)
                
                q_tokens = bm25s.tokenize([new_query_text])
                res, _ = retriever.retrieve(q_tokens, k=k, show_progress=False)
                combination_list.append(context_list[res[0][0]])
                ret_count += 1

    # Ensure final answer
    combined = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
    
    if not reasoning_list or "So the answer is:" not in reasoning_list[-1]:
        combined += " So the answer is:"
        final_text, _ = llm_gen._generate(combined, stop=["\n"])
        reasoning_list[-1] = final_text.strip()
    
    answer = reasoning_list[-1].split("So the answer is:")[-1].strip()
    ret_ratio = ret_count / len(reasoning_list) if reasoning_list else 0
    
    return combined, answer, combination_list, ret_ratio


def main():
    """Main function to run FLARE reasoning."""
    parser = argparse.ArgumentParser(
        description="FLARE legal reasoning using ExaOne model"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
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
    parser.add_argument(
        "--k", 
        type=int, 
        default=1,
        help="Number of contexts to retrieve each step"
    )
    parser.add_argument(
        "--token_threshold", 
        type=float, 
        default=-1.5,
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
        default="flare_exaone_results.jsonl",
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
    print("Initializing ExaOne generator...")
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
    total_ret_ratio = 0
    
    for item in tqdm(data_list, desc="Processing"):
        question = item['background'] + item['question']
        
        trace, ans, comb, ret_ratio = flare(llm_gen, retriever, question, context_list, k=args.k)
        clean_ans = ans.split("So the final answer is:")[-1].strip()
        
        record = {
            'question': question,
            'answers': [{'answer': clean_ans, 'logp': 0, 'comb': comb}],
            'full_trace': trace.split("<Query>")[-1].strip() if "<Query>" in trace else trace
        }
        results.append(record)
        total_ret_ratio += ret_ratio
    
    # Calculate average retrieval ratio
    avg_ratio = total_ret_ratio / len(data_list) if data_list else 0
    
    # Save results
    print(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    print(f"Average retrieval ratio: {avg_ratio:.4f}")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
