#!/bin/bash

# Script to evaluate inference results against ground truth.

DATASETS="champ logicbench bigcodebench theoremqa medcalcbench toolqa"

for DATASET in $DATASETS; do
    if [ ! -f "results/rejector_${DATASET}.jsonl" ]; then
        echo "Skip: results/rejector_${DATASET}.jsonl not found."
        continue
    fi

    echo "Evaluating: ${DATASET}..."

    python3 -m sragents.cli.main evaluate \
        --input results/oracle_${DATASET}.jsonl \
        --instances data/bench/instances/${DATASET}.json \
        --output results/eval_oracle_${DATASET}.json \
        --workers 32
    # Print a quick summary
    python3 -c "import json; d=json.load(open('results/eval_oracle_${DATASET}.json')); print(f'Dataset: {DATASET} | Accuracy: {d.get(\"accuracy\", \"N/A\")}')" 2>/dev/null
done

echo "Evaluation tasks complete."
