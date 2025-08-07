#!/usr/bin/env python3
"""
Beam Aggregation Pipeline Runner

This script orchestrates the complete beam aggregation pipeline:
1. Generate hierarchical question decomposition trees
2. Use trees to perform beam aggregation question answering
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error:")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main function to run the complete beam aggregation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete beam aggregation pipeline"
    )
    
    # Pipeline arguments
    parser.add_argument(
        '--input', 
        type=str, 
        default="source/koblex.jsonl",
        help="Path to input JSONL file"
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
        help="Path to output JSONL file (use $TREE placeholder for tree file)"
    )
    parser.add_argument(
        '--tree_output', 
        type=str, 
        default=None,
        help="Path to tree output file (if not specified, will be derived from output)"
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="",
        help="Model name for VLLM"
    )
    parser.add_argument(
        '--openai_model', 
        type=str, 
        default="gpt-4o",
        help="OpenAI model name"
    )
    parser.add_argument(
        '--use_openai', 
        action='store_true',
        help="Use OpenAI instead of VLLM"
    )
    parser.add_argument(
        '--tensor_parallel_size', 
        type=int, 
        default=2,
        help="Tensor parallel size for VLLM"
    )
    parser.add_argument(
        '--max_parallel_calls', 
        type=int, 
        default=20,
        help="Maximum parallel API calls for OpenAI"
    )
    
    args = parser.parse_args()
    
    # Determine tree output path
    if args.tree_output is None:
        # Extract directory and filename from output
        output_path = Path(args.output)
        tree_output = output_path.parent / f"{output_path.stem}_tree{output_path.suffix}"
    else:
        tree_output = args.tree_output
    
    print("🚀 Starting Beam Aggregation Pipeline")
    print(f"Input: {args.input}")
    print(f"Context: {args.context}")
    print(f"Tree Output: {tree_output}")
    print(f"Final Output: {args.output}")
    print(f"Model: {'OpenAI ' + args.openai_model if args.use_openai else 'VLLM ' + args.model_name}")
    
    # Step 1: Generate trees
    if args.use_openai:
        make_tree_cmd = [
            sys.executable, "make_tree_openai.py",
            "--input", args.input,
            "--output", str(tree_output),
            "--model_name", args.openai_model,
            "--max_parallel_calls", str(args.max_parallel_calls)
        ]
    else:
        make_tree_cmd = [
            sys.executable, "make_tree_vllm.py",
            "--input", args.input,
            "--output", str(tree_output),
            "--model_name", args.model_name,
            "--tensor_parallel_size", str(args.tensor_parallel_size)
        ]
    
    if not run_command(make_tree_cmd, "Step 1: Generate hierarchical question decomposition trees"):
        print("❌ Pipeline failed at tree generation step")
        return 1
    
    # Step 2: Run beam aggregation question answering
    if args.use_openai:
        beamaggr_cmd = [
            sys.executable, "beamaggr_openai.py",
            "--input", str(tree_output),
            "--context", args.context,
            "--output", args.output,
            "--model_name", args.openai_model
        ]
    else:
        beamaggr_cmd = [
            sys.executable, "beamaggr_vllm.py",
            "--input", str(tree_output),
            "--context", args.context,
            "--output", args.output,
            "--model_name", args.model_name,
            "--tensor_parallel_size", str(args.tensor_parallel_size)
        ]
    
    if not run_command(beamaggr_cmd, "Step 2: Run beam aggregation question answering"):
        print("❌ Pipeline failed at beam aggregation step")
        return 1
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"Tree file: {tree_output}")
    print(f"Final results: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

