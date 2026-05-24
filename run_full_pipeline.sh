#!/bin/bash
mkdir -p results

DATASETS=("bigcodebench" "champ" "logicbench" "medcalcbench" "theoremqa" "toolqa")
MODEL="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"
WORKERS_RERANK=64
WORKERS_FUSION=16

for ds in "${DATASETS[@]}"; do
    echo ">>> Stage 1: Retrieval (RAG-Fusion) for $ds"
    python3 -m sragents.cli.main retrieve \
        --retriever rag_fusion \
        --retriever-arg expansion_model=$MODEL \
        --retriever-arg api_base=$API_BASE \
        --retriever-arg workers=$WORKERS_FUSION \
        --corpus data/bench/corpus/corpus.json \
        --instances data/bench/instances/${ds}.json \
        --output results/retrieved_${ds}.json \
        --top-k 50

    echo ">>> Stage 2: Reranking for $ds"
    python3 -m sragents.cli.main rerank \
        --input results/retrieved_${ds}.json \
        --output results/final_${ds}.json \
        --model $MODEL \
        --api-base $API_BASE \
        --instances data/bench/instances/${ds}.json \
        --top-k 50 \
        --workers $WORKERS_RERANK
done
