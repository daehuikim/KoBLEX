#!/bin/bash

# Self-Ask Legal Reasoning Pipeline
# This script runs self-ask reasoning with either OpenAI or vLLM

echo "Self-Ask Legal Reasoning Pipeline"
echo "================================="

# Default parameters
INPUT_FILE="filtered_data.jsonl"
STATUTE_FILE="collections3.jsonl"
OUTPUT_DIR="results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model type is specified
if [ "$1" = "openai" ]; then
    echo "Running Self-Ask with OpenAI..."
    python selfask_openai.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/selfask_openai_results.jsonl \
        --model_name "gpt-4o" \
        --temperature 0.01 \
        --top_p 0.9 \
        --max_tokens 2000

elif [ "$1" = "vllm" ]; then
    echo "Running Self-Ask with vLLM..."
    python selfask_vllm.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/selfask_vllm_results.jsonl \
        --model_name "Qwen/Qwen3-8B" \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 2000 \
        --trust_remote_code

else
    echo "Usage: $0 {openai|vllm}"
    echo ""
    echo "Examples:"
    echo "  $0 openai    # Run with OpenAI GPT-4o"
    echo "  $0 vllm      # Run with vLLM Qwen3-8B"
    echo ""
    echo "Make sure to set OPENAI_API_KEY environment variable for OpenAI runs."
    exit 1
fi

echo "Pipeline completed!"
echo "Results saved to: $OUTPUT_DIR/"
