#!/usr/bin/env python3
"""
Selection Retrieval Generator

This script retrieves and selects statutory provisions using BM25 + CrossEncoder reranking + vLLM.
It processes legal questions and selects the most relevant provisions from candidates.
"""

import argparse
import json
import gc
import torch
from typing import List, Any
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


from ..prompts.selection_retrieval import SYSTEM_PROMPT, INSTRUCTION_PROMPT
from utils import get_statute_retriever, get_reranker
import bm25s


class LlmGenerator:
    """Language model generator for selection retrieval."""
    
    def __init__(self, model_name: str, dtype: str, trust_remote_code: bool,
                 tensor_parallel_size: int, temperature: float, top_p: float, 
                 max_tokens: int):
        """Initialize the LLM generator with specified parameters."""
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
    
    def generate(self, prompts: List[str]) -> List[Any]:
        """Generate completions for the given prompts."""
        return self.llm.generate(prompts, self.sampling_params)


def rerank(questions: List[str], candidates: List[List[List[int]]], 
           statute: List[str], reranker) -> List[List[List[str]]]:
    """
    Given questions and BM25 candidate indices, rerank using CrossEncoder.
    Returns nested list of contexts ordered by score.
    """
    reranked = []
    for q, cand_list in zip(questions, candidates):
        per_q = []
        for indices in cand_list:
            texts = [statute[i] for i in indices]
            pairs = [(q, t) for t in texts]
            scores = reranker.predict(pairs, batch_size=50, show_progress_bar=False)
            sorted_texts = [t for _, t in sorted(zip(scores, texts), reverse=True)]
            per_q.append(sorted_texts)
        reranked.append(per_q)
    return reranked


def create_prompts(reranked: List[List[List[str]]], questions: List[str], 
                   backgrounds: List[str], tokenizer: AutoTokenizer) -> List[str]:
    """Create prompts for all data items."""
    prompts = []
    
    for subs_list, q, bg in zip(reranked, questions, backgrounds):
        for contexts in subs_list:
            top10 = contexts[:10]
            cand_lines = [f"{i}: {c}" for i, c in enumerate(top10)]
            cand_str = "\n".join(cand_lines)
            
            user_msg = INSTRUCTION_PROMPT.substitute(
                background=bg,
                question=q,
                candidates=cand_str
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
            
            inp = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            prompts.append(tokenizer.decode(inp[0], skip_special_tokens=False))
    
    return prompts


def process_completions(completions: List[Any], counts: List[int], 
                       reranked: List[List[List[str]]]) -> List[List[str]]:
    """Process LLM completions and extract selections."""
    selections = []
    ptr = 0
    
    for count, contexts_list in zip(counts, reranked):
        sel_per_q = []
        for _ in range(count):
            text = completions[ptr].outputs[0].text
            ans = text.split("Answer:")[-1].split("</think>")[0].strip()  # For Qwen families
            choice = int(ans) if ans.isdigit() and 0 <= int(ans) <= 9 else 0
            sel_per_q.append(contexts_list[0][choice])
            ptr += 1
        selections.append(sel_per_q)
    
    return selections


def main():
    """Main function to run the selection retrieval generator."""
    parser = argparse.ArgumentParser(
        description="Retrieve and select statutory provisions with BM25 + CrossEncoder reranking + vLLM"
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to input JSONL with 'subs', 'background', 'question'."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Path to output JSONL for selected provisions."
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="Qwen/Qwen3-8B",
        help="LLM model name."
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default="auto"
    )
    parser.add_argument(
        '--trust_remote_code', 
        action='store_true'
    )
    parser.add_argument(
        '--tensor_parallel_size', 
        type=int, 
        default=1
    )
    
    # Generation arguments
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.0
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.9
    )
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=2048
    )
    
    # Retrieval arguments
    parser.add_argument(
        '--statute_pool', 
        type=str, 
        default="parser/source/statute.jsonl",
        help="Optional path to statutes JSONL; default from utils."
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} records")
    
    # Setup retrieval and reranker
    print("Setting up retrieval and reranker...")
    statute, retriever = get_statute_retriever(args.statute_pool)
    reranker = get_reranker()
    
    # BM25 retrieval for sub-queries
    print("Performing BM25 retrieval...")
    questions, sub_queries, backgrounds = [], [], []
    bm25_candidates = []
    
    for item in tqdm(data, desc="BM25 retrieval"):
        q = item['question']
        backgrounds.append(item['background'])
        subs = item['subs']
        sub_queries.append(subs)
        questions.append(q)
        q_cands = []
        for sq in subs:
            tokens = bm25s.tokenize([sq], show_progress=False)
            idxs, _ = retriever.retrieve(tokens, k=100, show_progress=False)
            q_cands.append(idxs[0])
        bm25_candidates.append(q_cands)
    
    # Rerank with CrossEncoder
    print("Reranking with CrossEncoder...")
    reranked = rerank(questions, bm25_candidates, statute, reranker)
    
    # Free reranker GPU memory
    print("Freeing reranker memory...")
    reranker.to('cpu')
    del reranker
    gc.collect()
    torch.cuda.empty_cache()
    
    # Initialize LLM
    print("Initializing LLM generator...")
    llm_gen = LlmGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Construct prompts for selection
    print("Creating prompts...")
    prompts = []
    counts = []
    for subs_list, q, bg in zip(reranked, questions, backgrounds):
        counts.append(len(subs_list))
        for contexts in subs_list:
            top10 = contexts[:10]
            cand_lines = [f"{i}: {c}" for i, c in enumerate(top10)]
            cand_str = "\n".join(cand_lines)
            user_msg = INSTRUCTION_PROMPT.substitute(
                background=bg,
                question=q,
                candidates=cand_str
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
            inp = llm_gen.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            prompts.append(llm_gen.tokenizer.decode(inp[0], skip_special_tokens=False))
    
    # Generate selections
    print("Generating selections...")
    outputs = llm_gen.generate(prompts)
    
    # Parse selections
    print("Processing results...")
    selections = process_completions(outputs, counts, reranked)
    
    # Write results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as fout:
        for bg, q, subs, sel in zip(backgrounds, questions, sub_queries, selections):
            out = {"background":bg,"question": q, "parametric_provisions": subs, "selected_provisions": sel}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    print(f"Successfully processed {len(data)} records")


if __name__ == '__main__':
    main()
