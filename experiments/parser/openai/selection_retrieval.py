#!/usr/bin/env python3
"""
Selection Retrieval Generator (OpenAI)

This script retrieves and selects statutory provisions using BM25 + CrossEncoder reranking + OpenAI API.
It processes legal questions and selects the most relevant provisions from candidates.
"""

import argparse
import asyncio
import json
import gc
import torch
from typing import List, Any
from tqdm import tqdm

from ..prompts.selection_retrieval import SYSTEM_PROMPT, INSTRUCTION_PROMPT
from utils import get_statute_retriever, get_reranker, get_completion_list
import bm25s


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


async def create_prompts(reranked: List[List[List[str]]], questions: List[str], 
                         backgrounds: List[str]) -> List[str]:
    """Create prompts for all data items."""
    prompts = []
    
    for subs_list, q, bg in zip(reranked, questions, backgrounds):
        for contexts in subs_list:
            top10 = contexts[:10]
            cand_lines = [f"{i}: {c}" for i, c in enumerate(top10)]
            cand_str = "\n".join(cand_lines)
            
            # Create full prompt with system and instruction
            full_prompt = f"{SYSTEM_PROMPT}\n\n{INSTRUCTION_PROMPT.substitute(background=bg, question=q, candidates=cand_str)}"
            prompts.append(full_prompt)
    
    return prompts


def process_completions(completions: List[tuple], counts: List[int], 
                       reranked: List[List[List[str]]]) -> List[List[str]]:
    """Process OpenAI completions and extract selections."""
    selections = []
    ptr = 0
    
    for count, contexts_list in zip(counts, reranked):
        sel_per_q = []
        for _ in range(count):
            pred, _ = completions[ptr]
            answer = pred.split("Answer:")[-1].strip()
            choice = int(answer) if answer.isdigit() and 0 <= int(answer) <= 9 else 0
            sel_per_q.append(contexts_list[0][choice])
            ptr += 1
        selections.append(sel_per_q)
    
    return selections


async def main():
    """Main function to run the selection retrieval generator."""
    parser = argparse.ArgumentParser(
        description="Retrieve and select statutory provisions with BM25 + CrossEncoder reranking + OpenAI API"
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
        default="gpt-4o",
        help="OpenAI model name."
    )
    parser.add_argument(
        '--max_parallel_calls', 
        type=int, 
        default=20,
        help="Maximum parallel API calls"
    )
    
    # Retrieval arguments
    parser.add_argument(
        '--statute_pool', 
        type=str, 
        default="source/statute.jsonl",
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
            # Create full prompt with system and instruction
            full_prompt = f"{SYSTEM_PROMPT}\n\n{INSTRUCTION_PROMPT.substitute(background=bg, question=q, candidates=cand_str)}"
            prompts.append(full_prompt)
    
    # Generate selections
    print("Generating selections...")
    outputs = await get_completion_list(prompts, args.max_parallel_calls, args.model_name)
    
    # Parse selections
    print("Processing results...")
    selections = process_completions(outputs, counts, reranked)
    
    # Write results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as fout:
        for bg, q, subs, sel in zip(backgrounds, questions, sub_queries, selections):
            out = {"background":bg, "question":q, "parametric_provisions": subs, "selected_provisions": sel}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    print(f"Successfully processed {len(data)} records")


if __name__ == '__main__':
    asyncio.run(main())
