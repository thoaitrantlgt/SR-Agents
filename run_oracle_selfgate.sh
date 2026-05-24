#!/bin/bash

# Script to run oracle provider with react engine for ALL datasets.
# Model sees gold skill names/descriptions and must call LoadSkill[index] to load content.
# This measures skill loading rate like Table 6 in the paper.

MODEL="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"
DATASETS="medcalcbench theoremqa logicbench champ bigcodebench toolqa"

for DATASET in $DATASETS; do
    echo "------------------------------------------------"
    echo "Running oracle (react_progressive_disclosure) for: ${DATASET}"
    echo "------------------------------------------------"

    python3 -m sragents.cli.main infer \
        --instances data/bench/instances/${DATASET}.json \
        --output results/oracle_${DATASET}.jsonl \
        --model ${MODEL} \
        --api-base ${API_BASE} \
        --provider oracle \
        --engine react_progressive_disclosure \
        --workers 8 \
        --temperature 0.0
done

echo "All tasks complete. Results saved in SR-Agents/results/ directory."
