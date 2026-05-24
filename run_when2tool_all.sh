#!/bin/bash

# Script to run When2Tool pipeline on all 6 datasets with optimal hyperparameters
# Optimal config found from grid-sweep: --train-ratio 0.5 --regularization 1000.0 --pca-components 32

set -e

# Datasets
DATASETS="medcalcbench theoremqa logicbench champ bigcodebench toolqa"
MODEL="Qwen/Qwen3-4B"
DEVICE="cuda"

echo "======================================================================"
echo "Starting When2Tool Pipeline for all 6 datasets"
echo "Model: ${MODEL} | Device: ${DEVICE}"
echo "Optimal Hyperparameters: --train-ratio 0.5 --regularization 1000.0 --pca-components 32"
echo "======================================================================"
echo ""

for DATASET in $DATASETS; do
    echo "----------------------------------------------------------------------"
    echo "Processing dataset: ${DATASET}"
    echo "----------------------------------------------------------------------"
    
    # Define output directory per dataset
    OUT_DIR="./when2tool_outputs_${DATASET}"
    
    python3 -m sragents.when2tool.pipeline \
      "data/bench/instances/${DATASET}.json" \
      --output-dir "${OUT_DIR}" \
      --train-ratio 0.5 \
      --batch-size 4 \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --steps 1,2,3 \
      --eval-file "results/eval_noskill_${DATASET}.json" \
      --regularization 1000.0 \
      --pca-components 32
      
    echo "✅ Completed pipeline for ${DATASET}!"
    echo "Results saved to: ${OUT_DIR}"
    echo ""
done

echo "======================================================================"
echo "🎉 When2Tool Pipeline run completed successfully for all 6 datasets!"
echo "======================================================================"
