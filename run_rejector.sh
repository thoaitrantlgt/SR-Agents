#!/bin/bash

# Script to run inference using the LLM Classifier Rejector provider.
# This provider evaluates if internal knowledge is sufficient or if a tool is needed.

MODEL="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"

# Datasets using engine=direct
DIRECT_DATASETS="medcalcbench theoremqa logicbench champ bigcodebench"

# Datasets using engine=react
REACT_DATASETS="toolqa"

for DATASET in $DIRECT_DATASETS; do
    echo "------------------------------------------------"
    echo "Running inference (direct) for: ${DATASET}"
    echo "------------------------------------------------"

    python3 -m sragents.cli.main --plugin sragents.infer.providers.llm_rejector infer \
        --instances data/bench/instances/${DATASET}.json \
        --output results/rejector_${DATASET}.jsonl \
        --model ${MODEL} \
        --api-base ${API_BASE} \
        --provider llm_rejector \
        --provider-arg source=results/final_${DATASET}.json \
        --provider-arg pool=1 \
        --engine direct \
        --workers 16 \
        --temperature 0.0
done

for DATASET in $REACT_DATASETS; do
    echo "------------------------------------------------"
    echo "Running inference (react) for: ${DATASET}"
    echo "------------------------------------------------"

    python3 -m sragents.cli.main --plugin sragents.infer.providers.llm_rejector infer \
        --instances data/bench/instances/${DATASET}.json \
        --output results/rejector_${DATASET}.jsonl \
        --model ${MODEL} \
        --api-base ${API_BASE} \
        --provider llm_rejector \
        --provider-arg source=results/final_${DATASET}.json \
        --provider-arg pool=1 \
        --engine react \
        --workers 8 \
        --temperature 0.0
done

echo "All tasks complete. Results saved in SR-Agents/results/ directory."
