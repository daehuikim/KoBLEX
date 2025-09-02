#!/usr/bin/env python3
"""
Question Answering Generator

This script generates legal answers using selected statutory provisions with vLLM.
It processes legal questions and generates answers based on provided context.
"""

import argparse
import json
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


from ..prompts.question_answering import SYSTEM_PROMPT, INSTRUCTION_PROMPT


class LlmGenerator:
    """Language model generator for question answering."""
    
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
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate completion texts for each prompt."""
        batch_inputs = []
        for prompt in prompts:
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
            batch_inputs.append(self.tokenizer.decode(inp[0], skip_special_tokens=False))

        responses = self.llm.generate(batch_inputs, self.sampling_params)
        return [resp.outputs[0].text.strip() for resp in responses]


def create_prompts(data_list: List[Dict[str, Any]]) -> List[str]:
    """Create prompts for all data items."""
    prompts = []
    
    for item in data_list:
        question = item['background'] + item['question']
        provisions = item.get('selected_provisions', [])
        prompt = INSTRUCTION_PROMPT.substitute(
            question=question,
            context_str="\n".join(provisions)
        )
        prompts.append(prompt)
    
    return prompts


def process_completions(completions: List[str]) -> List[str]:
    """Process LLM completions and extract answers."""
    answers = []
    
    for text in completions:
        # Extract answer portion
        answer = text.split("Answer:")[-1].split("</think>")[0].strip()
        answers.append(answer)
    
    return answers


def main():
    """Main function to run the question answering generator."""
    parser = argparse.ArgumentParser(
        description="Legal QA using selected statutory provisions with vLLM generation"
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
        default="Qwen/Qwen3-8B",
        help="LLM model identifier for vLLM"
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
        default=4000
    )
    
    args = parser.parse_args()
    
    # Load retrieval-based input
    print(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} records")
    
    # Build prompts
    print("Creating prompts...")
    prompts = create_prompts(data)
    
    # Initialize LLM generator
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
    
    # Generate answers
    print("Generating answers...")
    outputs = llm_gen.generate(prompts)
    
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
    main()
