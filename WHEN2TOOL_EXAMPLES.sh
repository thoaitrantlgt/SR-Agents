#!/bin/bash

# Example: Run When2Tool pipeline on SR-Agents datasets
# This script demonstrates how to run the full pipeline with different datasets

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}When2Tool Pipeline Examples${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Example 1: Run on BigCodeBench with default settings
echo -e "${GREEN}Example 1: BigCodeBench with 30/70 split${NC}"
echo "Command:"
echo "python -m sragents.when2tool.pipeline \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --output-dir ./when2tool_outputs/bigcodebench \\"
echo "    --train-ratio 0.3 \\"
echo "    --model meta-llama/Llama-3.1-8B-Instruct \\"
echo "    --steps 1,2,3 \\"
echo "    --batch-size 4"
echo ""
echo "To run this example:"
echo "python -m sragents.when2tool.pipeline data/bench/instances/bigcodebench.json --output-dir ./when2tool_outputs/bigcodebench"
echo ""
echo "---"
echo ""

# Example 2: Run only steps 1-2 for quick testing
echo -e "${GREEN}Example 2: Only split and extract (quick test)${NC}"
echo "Command:"
echo "python -m sragents.when2tool.pipeline \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --output-dir ./when2tool_outputs/test \\"
echo "    --steps 1,2 \\"
echo "    --batch-size 2 \\"
echo "    --device cpu"  # Use CPU for quick test
echo ""
echo "---"
echo ""

# Example 3: Multiple models
echo -e "${GREEN}Example 3: Using Qwen3-4B model${NC}"
echo "Command:"
echo "python -m sragents.when2tool.pipeline \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --output-dir ./when2tool_outputs/qwen3-4b \\"
echo "    --model Qwen/Qwen3-4B-Instruct \\"
echo "    --batch-size 8"
echo ""
echo "---"
echo ""

# Example 4: Run individual steps
echo -e "${GREEN}Example 4: Running individual steps${NC}"
echo ""
echo "Step 1: Split dataset"
echo "python -m sragents.when2tool.split_dataset \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --train-ratio 0.3 \\"
echo "    --output-dir ./when2tool_outputs/splits"
echo ""
echo "Step 2: Extract hidden states"
echo "python -m sragents.when2tool.extract_hidden_states \\"
echo "    Qwen/Qwen3-4B-Instruct \\"
echo "    --train-data ./when2tool_outputs/splits/train_json.json \\"
echo "    --test-data ./when2tool_outputs/splits/test_json.json \\"
echo "    --output-dir ./when2tool_outputs/features"
echo ""
echo "Step 3: Train linear probe"
echo "python -m sragents.when2tool.train_linear_probe \\"
echo "    --train-hidden ./when2tool_outputs/features/train/hidden_states.pt \\"
echo "    --test-hidden ./when2tool_outputs/features/test/hidden_states.pt \\"
echo "    --train-labels ./when2tool_outputs/features/train/items.json \\"
echo "    --output-dir ./when2tool_outputs/probe"
echo ""
echo "---"
echo ""

# Example 5: Shell script wrapper
echo -e "${GREEN}Example 5: Using shell script wrapper${NC}"
echo "Command:"
echo "bash run_when2tool_pipeline.sh \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --output-dir ./when2tool_outputs/shell_example \\"
echo "    --train-ratio 0.3 \\"
echo "    --model Qwen/Qwen3-4B-Instruct"
echo ""
echo "---"
echo ""

# Example 6: Different split ratios
echo -e "${GREEN}Example 6: Different train/test ratios${NC}"
echo "20% train / 80% test:"
echo "python -m sragents.when2tool.pipeline \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --train-ratio 0.2 \\"
echo "    --output-dir ./when2tool_outputs/20_80_split"
echo ""
echo "50% train / 50% test:"
echo "python -m sragents.when2tool.pipeline \\"
echo "    data/bench/instances/bigcodebench.json \\"
echo "    --train-ratio 0.5 \\"
echo "    --output-dir ./when2tool_outputs/50_50_split"
echo ""
echo "---"
echo ""

# Example 7: Using Python API directly
echo -e "${GREEN}Example 7: Using Python API${NC}"
echo "Create a Python script (e.g., my_when2tool.py):"
echo ""
cat > /tmp/when2tool_api_example.py << 'EOF'
from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    dataset_path="data/bench/instances/bigcodebench.json",
    output_base_dir="./when2tool_outputs/api_example",
    train_ratio=0.3,
    model_name="Qwen/Qwen3-4B-Instruct",
    device="cuda",
    batch_size_extraction=4,
    steps="1,2,3",
)

# Run pipeline
pipeline = When2ToolPipeline(config)
results = pipeline.run()

# Access results
print("Training set size:", results["step1_split"]["train_size"])
print("Test set size:", results["step1_split"]["test_size"])
print("Probe path:", results["step3_probe"]["probe_path"])
EOF
echo "cat /tmp/when2tool_api_example.py"
echo ""
echo "Then run:"
echo "python my_when2tool.py"
echo ""
echo "---"
echo ""

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}More Options${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "For complete list of options, run:"
echo "python -m sragents.when2tool.pipeline --help"
echo ""
echo "For more information, see: WHEN2TOOL_README.md"
echo ""
