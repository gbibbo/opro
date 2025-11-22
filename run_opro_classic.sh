#!/bin/bash
#
# OPRO Classic: Run prompt optimization with local LLM
#
# Usage:
#   ./run_opro_classic.sh [base|lora] [seed]
#
# Examples:
#   ./run_opro_classic.sh base 42
#   ./run_opro_classic.sh lora 123
#

set -e

# Default values
MODE=${1:-base}
SEED=${2:-42}

# Configuration
MANIFEST="data/processed/conditions_final/conditions_manifest_split.parquet"
SPLIT="dev"
OPTIMIZER_LLM="Qwen/Qwen2.5-3B-Instruct"  # Smaller LLM for 8GB VRAM
NUM_ITERATIONS=15
CANDIDATES_PER_ITER=3
EARLY_STOPPING=5

echo "============================================"
echo "OPRO CLASSIC OPTIMIZATION"
echo "============================================"
echo "Mode: $MODE"
echo "Seed: $SEED"
echo "Optimizer LLM: $OPTIMIZER_LLM"
echo "Iterations: $NUM_ITERATIONS"
echo "Candidates/iter: $CANDIDATES_PER_ITER"
echo "============================================"

if [ "$MODE" == "base" ]; then
    OUTPUT_DIR="results/opro_classic_base_seed${SEED}"

    echo "Running OPRO on BASE MODEL (no fine-tuning)..."

    python scripts/opro_classic_optimize.py \
        --manifest "$MANIFEST" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --no_lora \
        --optimizer_llm "$OPTIMIZER_LLM" \
        --num_iterations $NUM_ITERATIONS \
        --candidates_per_iter $CANDIDATES_PER_ITER \
        --early_stopping $EARLY_STOPPING \
        --seed $SEED

elif [ "$MODE" == "lora" ]; then
    CHECKPOINT="checkpoints/qwen_lora_seed${SEED}/final"
    OUTPUT_DIR="results/opro_classic_lora_seed${SEED}"

    if [ ! -d "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        echo "Please train a LoRA checkpoint first or use 'base' mode."
        exit 1
    fi

    echo "Running OPRO on FINE-TUNED MODEL (LoRA)..."
    echo "Checkpoint: $CHECKPOINT"

    python scripts/opro_classic_optimize.py \
        --manifest "$MANIFEST" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --checkpoint "$CHECKPOINT" \
        --optimizer_llm "$OPTIMIZER_LLM" \
        --num_iterations $NUM_ITERATIONS \
        --candidates_per_iter $CANDIDATES_PER_ITER \
        --early_stopping $EARLY_STOPPING \
        --seed $SEED

else
    echo "ERROR: Invalid mode: $MODE"
    echo "Usage: ./run_opro_classic.sh [base|lora] [seed]"
    exit 1
fi

echo ""
echo "============================================"
echo "OPRO CLASSIC COMPLETE"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Best prompt:"
cat "$OUTPUT_DIR/best_prompt.txt"
echo ""
echo ""
echo "Metrics:"
cat "$OUTPUT_DIR/best_metrics.json"
echo ""
