#!/usr/bin/env python3
"""
Chain-of-Thought with One-time Retrieval Question Answering Generator (VLLM)

This script generates legal answers using chain-of-thought reasoning with one-time retrieval using VLLM.
It processes legal questions, retrieves relevant context, and generates answers with step-by-step reasoning.
"""

import argparse
import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict
import bm25s
import Stemmer

from ..prompts.naive_prompt import COT_OR_SYSTEM_PROMPT, COT_OR_QA_PROMPT
from utils import load_jsonl

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


class LlmGenerator:
    def __init__(self,
                 model_name: str,
                 dtype: str,
                 trust_remote_code: bool,
                 tensor_parallel_size: int,
                 temperature: float,
                 top_p: float,
                 max_tokens: int):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=1,
            seed=0
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"

    def _generate(self, prompts):
        # Prepare batch requests
        requests = []
        for idx, prompt in enumerate(prompts):
            messages = [
                {"role": "system", "content": COT_OR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            inp = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            token_ids = inp[0]
            full_input = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            requests.append(full_input)
        
        # Invoke vLLM
        responses = self.llm.generate(requests, self.sampling_params)
        
        results = []
        for resp in responses:
            output = resp.outputs[0]
            text = output.text
            token_lps = output.logprobs
            temp = []
            for step in token_lps:
                _, l = next(iter(step.items()))
                temp.append(l.logprob)
            
            avg_logprob = sum(temp) / len(temp)
            results.append((text, avg_logprob))
        return results


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


def make_qa_prompts(data_list: List[Dict], retriever, context_list: List[str], n_hops: int = None):
    """Create prompts for question answering with retrieval."""
    prompts = []
    combinations = []
    
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
        combinations.append(context_strs)
        
        context_str = "\n".join(context_strs)
        prompt = COT_OR_QA_PROMPT.substitute(question=temp_question, context_str=context_str)
        prompts.append(prompt)
    
    return prompts, combinations


def main():
    """Main function to run the chain-of-thought with one-time retrieval question answering generator."""
    parser = argparse.ArgumentParser(
        description="Legal QA using chain-of-thought reasoning with one-time retrieval using VLLM"
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
        default="LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
        help="Model name for VLLM"
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default="auto",
        help="Data type for model"
    )
    parser.add_argument(
        '--trust_remote_code', 
        action='store_true',
        help="Trust remote code"
    )
    parser.add_argument(
        '--tensor_parallel_size', 
        type=int, 
        default=2,
        help="Tensor parallel size"
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=4000,
        help="Maximum tokens to generate"
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
    
    # Create prompts
    print("Creating prompts...")
    prompts, combinations = make_qa_prompts(data_list, retriever, context_list, args.n_hops)
    
    # Initialize model
    print("Initializing model...")
    llm_gen = LlmGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Generate answers
    print("Generating answers...")
    results = llm_gen._generate(prompts)
    
    # Write results
    print(f"Writing results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as fw:
        for entry, prediction, combi in zip(data_list, results, combinations):
            record = {
                "question": entry['background'] + entry['question'],
                "answers": [{"answer": prediction[0].split("Answer:")[-1].strip(), "logp": prediction[1], "comb": combi}]
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Saved detailed answers to {args.output}")


if __name__ == '__main__':
    main()

    
    
