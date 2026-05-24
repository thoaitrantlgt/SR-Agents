#!/bin/bash

# Script to run inference on 6 datasets WITHOUT any skills (baseline).
# Uses the "none" provider which returns empty skill list.

MODEL="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"
DATASETS="medcalcbench theoremqa logicbench champ bigcodebench toolqa"

for DATASET in $DATASETS; do
    echo "================================================"
    echo "Running NO-SKILL baseline for: ${DATASET}"
    echo "================================================"

    python3 -m sragents.cli.main infer \
        --instances data/bench/instances/${DATASET}.json \
        --output results/noskill_${DATASET}.jsonl \
        --model ${MODEL} \
        --api-base ${API_BASE} \
        --provider none \
        --engine direct \
        --workers 16 \
        --temperature 0.7

    echo "Done: results/noskill_${DATASET}.jsonl"
done

# Evaluate all no-skill results
echo ""
echo "================================================"
echo "Evaluating no-skill baseline results..."
echo "================================================"

for DATASET in $DATASETS; do
    if [ ! -f "results/noskill_${DATASET}.jsonl" ]; then
        echo "Skip eval: results/noskill_${DATASET}.jsonl not found."
        continue
    fi

    echo "Evaluating: ${DATASET}..."
    python3 -m sragents.cli.main evaluate \
        --input results/noskill_${DATASET}.jsonl \
        --instances data/bench/instances/${DATASET}.json \
        --output results/eval_noskill_${DATASET}.json \
        --workers 32

    python3 -c "import json; d=json.load(open('results/eval_noskill_${DATASET}.json')); print(f'  {DATASET}: {d[\"metrics\"][\"correct\"]}/{d[\"metrics\"][\"total\"]} = {d[\"metrics\"][\"accuracy\"]:.2%}')" 2>/dev/null
done

echo ""
echo "All no-skill baseline tasks complete."
