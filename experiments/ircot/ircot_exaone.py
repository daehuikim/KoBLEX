#!/usr/bin/env python3
"""
IRCoT Legal Reasoning (ExaOne)

This script implements IRCoT (Interleaved Retrieval and Chain-of-Thought) 
reasoning for legal questions using ExaOne model with vLLM.
"""

import argparse
import logging
from typing import List, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts.ircot_prompt import SYSTEM_PROMPT, ST_STOP
from utils import (
    load_jsonl,
    save_jsonl,
    get_statute_retriever
)


class LlmGenerator:
    """vLLM generator for IRCoT reasoning with ExaOne."""
    
    def __init__(self, model_name: str, dtype: str, trust_remote_code: bool,
                 tensor_parallel_size: int, temperature: float, top_p: float, 
                 max_tokens: int):
    
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


def ircot(llm_gen: LlmGenerator, retriever, question: str, context_list: List[str], k: int = 1) -> Tuple[str, str, List[str]]:
    """Perform IRCoT reasoning."""
    import bm25s
    
    # Retrieve initial context
    q_tokens = bm25s.tokenize([question])
    res, _ = retriever.retrieve(q_tokens, k=k)
    combination_list = [context_list[res[0][0]]]
    reasoning_list = []
    question_str = f"{question}"

    max_steps = 8
    for _ in range(max_steps):
        cur_prompt = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
        
        ret_text = llm_gen._generate(cur_prompt, stop=ST_STOP)
        reasoning_list.append(ret_text.strip())
        
        if "So the answer is:" in ret_text:
            break

        # Retrieve next context based on model output
        out_tokens = bm25s.tokenize([ret_text], show_progress=False)
        res, _ = retriever.retrieve(out_tokens, k=k, show_progress=False)
        combination_list.append(context_list[res[0][0]])

    # Ensure final answer
    combined_prompt = f"""<Query> Context: {"\n".join(combination_list)}
Question: {question_str}
Answer: {" ".join(reasoning_list)}"""
    
    if not reasoning_list or "So the answer is:" not in reasoning_list[-1]:
        combined_prompt += " So the answer is:"
        final_text = llm_gen._generate(combined_prompt, stop=["\n"])
        reasoning_list.append(final_text)
    
    answer = reasoning_list[-1].split("So the answer is:")[-1].strip()
    return combined_prompt, answer, combination_list


def main():
    """Main function to run IRCoT reasoning."""
    parser = argparse.ArgumentParser(
        description="IRCoT legal reasoning using ExaOne model"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
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
        default="ircot_exaone_results.jsonl",
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
    for item in tqdm(data_list, desc="Processing"):
        question = item['background'] + item['question']
        
        trace, ans, comb = ircot(llm_gen, retriever, question, context_list, k=args.k)
        
        record = {
            'question': question,
            'answers': [{'answer': ans, 'logp': 0, 'comb': comb}],
            'full_trace': trace.split("<Query>")[-1].strip() if "<Query>" in trace else trace
        }
        results.append(record)
    
    # Save results
    print(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
