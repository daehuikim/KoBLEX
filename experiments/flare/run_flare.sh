#!/bin/bash

# FLARE Legal Reasoning Pipeline
# This script runs FLARE (Forward-Looking Active REtrieval) reasoning with OpenAI, Qwen, or ExaOne

echo "FLARE Legal Reasoning Pipeline"
echo "============================="

# Default parameters
INPUT_FILE="source/koblex.jsonl"
STATUTE_FILE="source/statute.jsonl"
OUTPUT_DIR="results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model type is specified
if [ "$1" = "openai" ]; then
    echo "Running FLARE with OpenAI..."
    python flare_openai.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/flare_openai_results.jsonl \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 4000 \
        --k 1 \
        --token_threshold -0.70

elif [ "$1" = "qwen" ]; then
    echo "Running FLARE with Qwen..."
    python flare_qwen.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/flare_qwen_results.jsonl \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 4000 \
        --k 1 \
        --token_threshold -0.60 \
        --trust_remote_code

elif [ "$1" = "exaone" ]; then
    echo "Running FLARE with ExaOne..."
    python flare_exaone.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/flare_exaone_results.jsonl \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 4000 \
        --k 1 \
        --token_threshold -1.5 \
        --trust_remote_code

else
    echo "Usage: $0 {openai|qwen|exaone}"
    echo ""
    echo "Examples:"
    echo "  $0 openai    # Run with OpenAI GPT-4o"
    echo "  $0 qwen      # Run with Qwen3-8B"
    echo "  $0 exaone    # Run with ExaOne-3.5-7.8B"
    echo ""
    echo "Make sure to set OPENAI_API_KEY environment variable for OpenAI runs."
    echo "Note: Model name must be specified via --model_name argument."
    exit 1
fi

echo "Pipeline completed!"
echo "Results saved to: $OUTPUT_DIR/"
