#!/bin/bash

# IRCoT Legal Reasoning Pipeline
# This script runs IRCoT reasoning with OpenAI, Qwen, or ExaOne

echo "IRCoT Legal Reasoning Pipeline"
echo "=============================="

# Default parameters
INPUT_FILE="source/koblex.jsonl"
STATUTE_FILE="source/statute.jsonl"
OUTPUT_DIR="results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model type is specified
if [ "$1" = "openai" ]; then
    echo "Running IRCoT with OpenAI..."
    python ircot_openai.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/ircot_openai_results.jsonl \
        --model_name "gpt-4o" \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 4000 \
        --k 1

elif [ "$1" = "qwen" ]; then
    echo "Running IRCoT with Qwen..."
    python ircot_qwen.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/ircot_qwen_results.jsonl \
        --model_name "Qwen/Qwen3-8B" \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 4000 \
        --k 1 \
        --trust_remote_code

elif [ "$1" = "exaone" ]; then
    echo "Running IRCoT with ExaOne..."
    python ircot_exaone.py \
        --input_file $INPUT_FILE \
        --statute_file $STATUTE_FILE \
        --output_file $OUTPUT_DIR/ircot_exaone_results.jsonl \
        --model_name "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" \
        --temperature 0.0 \
        --top_p 0.9 \
        --max_tokens 4000 \
        --k 1 \
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
    exit 1
fi

echo "Pipeline completed!"
echo "Results saved to: $OUTPUT_DIR/"
