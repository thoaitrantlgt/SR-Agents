#!/bin/bash

# Script to run inference using oracle_rejector provider.
# Uses gold skills + DSTL-Gate to test if the need-aware gate works correctly.

MODEL="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"

# Datasets using engine=direct
DIRECT_DATASETS="medcalcbench theoremqa logicbench champ bigcodebench"

# Datasets using engine=react
REACT_DATASETS="toolqa"

for DATASET in $DIRECT_DATASETS; do
    echo "------------------------------------------------"
    echo "Running oracle_rejector (direct) for: ${DATASET}"
    echo "------------------------------------------------"

    python3 -m sragents.cli.main --plugin sragents.infer.providers.oracle_rejector infer \
        --instances data/bench/instances/${DATASET}.json \
        --output results/oracle_rejector_${DATASET}.jsonl \
        --model ${MODEL} \
        --api-base ${API_BASE} \
        --provider oracle_rejector \
        --engine direct \
        --workers 8 \
        --temperature 0.0
done

for DATASET in $REACT_DATASETS; do
    echo "------------------------------------------------"
    echo "Running oracle_rejector (react) for: ${DATASET}"
    echo "------------------------------------------------"

    python3 -m sragents.cli.main --plugin sragents.infer.providers.oracle_rejector infer \
        --instances data/bench/instances/${DATASET}.json \
        --output results/oracle_rejector_${DATASET}.jsonl \
        --model ${MODEL} \
        --api-base ${API_BASE} \
        --provider oracle_rejector \
        --engine react \
        --workers 8 \
        --temperature 0.0
done

echo "All tasks complete. Results saved in SR-Agents/results/ directory."
