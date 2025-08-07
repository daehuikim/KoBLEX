#!/usr/bin/env python3
"""
Legal Question Answering Pipeline

This script chains the execution of three components:
1. Parametric Provision Generation
2. Selection Retrieval 
3. Question Answering

The pipeline processes legal questions from source/koblex.jsonl and generates
final answers using statutory provisions from source/statute.jsonl.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main pipeline execution."""
    
    # Define file paths with placeholder variables
    # $PAR_PROV$ - Output from parametric provision generation
    # $RET_PROV$ - Output from selection retrieval  
    # $RESULT$ - Final output from question answering
    PAR_PROV = "parametric_provisions.jsonl"
    RET_PROV = "retrieved_provisions.jsonl" 
    RESULT = "final_results.jsonl"
    MODEL = "Qwen/Qwen3-8B"
    
    # Input files
    INPUT_FILE = "source/koblex.jsonl"
    SEARCH_POOL = "source/statute.jsonl"
    
    print("Legal Question Answering Pipeline")
    print("="*50)
    print(f"Input file: {INPUT_FILE}")
    print(f"Search pool: {SEARCH_POOL}")
    print(f"Intermediate files: {PAR_PROV}, {RET_PROV}")
    print(f"Final output: {RESULT}")
    print("="*50)
    
    # Step 1: Parametric Provision Generation
    # Input: source/koblex.jsonl
    # Output: $PAR_PROV$ (parametric_provisions.jsonl)
    cmd1 = [
        "python", "parametric_provision.py",
        "--input_file", INPUT_FILE,
        "--output_file", PAR_PROV,
        "--model_name", MODEL
    ]
    
    if not run_command(cmd1, "Step 1: Parametric Provision Generation"):
        print("Pipeline failed at Step 1")
        sys.exit(1)
    
    # Step 2: Selection Retrieval
    # Input: $PAR_PROV$ (parametric_provisions.jsonl)
    # Search pool: source/statute.jsonl  
    # Output: $RET_PROV$ (retrieved_provisions.jsonl)
    cmd2 = [
        "python", "selection_retrieval.py",
        "--input", PAR_PROV,
        "--statute_pool", SEARCH_POOL,
        "--output", RET_PROV,
        "--model_name", MODEL
    ]
    
    if not run_command(cmd2, "Step 2: Selection Retrieval"):
        print("Pipeline failed at Step 2")
        sys.exit(1)
    
    # Step 3: Question Answering
    # Input: $RET_PROV$ (retrieved_provisions.jsonl)
    # Output: $RESULT$ (final_results.jsonl)
    cmd3 = [
        "python", "question_answering.py", 
        "--input", RET_PROV,
        "--output", RESULT,
        "--model_name", MODEL
    ]
    
    if not run_command(cmd3, "Step 3: Question Answering"):
        print("Pipeline failed at Step 3")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("Pipeline completed successfully!")
    print(f"Final results saved to: {RESULT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
