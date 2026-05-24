#!/bin/bash

# When2Tool Pipeline - Complete workflow for dataset splitting, feature extraction, probe training
# Usage: ./run_when2tool_pipeline.sh <dataset_path> [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./when2tool_outputs"
TRAIN_RATIO="0.3"
MODEL="Qwen/Qwen3-4B"
DEVICE="cuda"
BATCH_SIZE="4"
MAX_LENGTH="512"
STEPS="1,2,3"
REGULARIZATION="1.0"
PREFILL_MODE="soft"
SEED="42"
SPLIT_BY_TYPE="false"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [options]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR         Output directory (default: ./when2tool_outputs)"
    echo "  --train-ratio RATIO      Training ratio (default: 0.3 for 30%)"
    echo "  --model MODEL            Model name (default: Qwen/Qwen3-4B-Instruct)"
    echo "  --device DEVICE          cuda or cpu (default: cuda)"
    echo "  --batch-size SIZE        Batch size (default: 4)"
    echo "  --max-length LENGTH      Max sequence length (default: 512)"
    echo "  --steps STEPS            Steps to run: 1,2,3 (default: 1,2,3)"
    echo "  --regularization REG     L2 regularization (default: 1.0)"
    echo "  --prefill-mode MODE      soft or hard (default: soft)"
    echo "  --seed SEED              Random seed (default: 42)"
    echo "  --split-by-type          Split by dataset type"
    echo ""
    exit 1
fi

DATASET_PATH="$1"
shift

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --regularization)
            REGULARIZATION="$2"
            shift 2
            ;;
        --prefill-mode)
            PREFILL_MODE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --split-by-type)
            SPLIT_BY_TYPE="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset file not found: $DATASET_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}When2Tool Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Train ratio: $TRAIN_RATIO"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Max length: $MAX_LENGTH"
echo "  Steps: $STEPS"
echo "  Regularization: $REGULARIZATION"
echo "  Prefill mode: $PREFILL_MODE"
echo "  Seed: $SEED"
echo "  Split by type: $SPLIT_BY_TYPE"
echo ""

# Build command
CMD="python -m sragents.when2tool.pipeline"
CMD="$CMD \"$DATASET_PATH\""
CMD="$CMD --output-dir \"$OUTPUT_DIR\""
CMD="$CMD --train-ratio $TRAIN_RATIO"
CMD="$CMD --model \"$MODEL\""
CMD="$CMD --device $DEVICE"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --max-length $MAX_LENGTH"
CMD="$CMD --steps $STEPS"
CMD="$CMD --regularization $REGULARIZATION"
CMD="$CMD --prefill-mode $PREFILL_MODE"
CMD="$CMD --seed $SEED"

if [ "$SPLIT_BY_TYPE" = "true" ]; then
    CMD="$CMD --split-by-type"
fi

# Run pipeline
echo -e "${GREEN}Running pipeline...${NC}"
echo ""
eval $CMD

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results saved to: $OUTPUT_DIR"
