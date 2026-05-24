#!/bin/bash

# When2Tool Pipeline - Validation Script
# Checks that all modules are properly installed and accessible

set -e

echo "When2Tool Pipeline - Validation Check"
echo "======================================"
echo ""

# Check Python version
echo -n "Checking Python version... "
python --version
echo "✓"
echo ""

# Check imports
echo "Checking imports..."
python << 'EOF'
try:
    print("  - torch...", end=" ")
    import torch
    print("✓")
    
    print("  - transformers...", end=" ")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✓")
    
    print("  - scikit-learn...", end=" ")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("✓")
    
    print("  - tqdm...", end=" ")
    from tqdm import tqdm
    print("✓")
    
    print("  - numpy...", end=" ")
    import numpy as np
    print("✓")
    
    print("  - sragents.when2tool...", end=" ")
    from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig
    print("✓")
    
    print("  - sragents.when2tool.split_dataset...", end=" ")
    from sragents.when2tool.split_dataset import split_dataset
    print("✓")
    
    print("  - sragents.when2tool.extract_hidden_states...", end=" ")
    from sragents.when2tool.extract_hidden_states import HiddenStateExtractor
    print("✓")
    
    print("  - sragents.when2tool.train_linear_probe...", end=" ")
    from sragents.when2tool.train_linear_probe import LinearProbe
    print("✓")
    
    print("  - sragents.when2tool.prefill_inference...", end=" ")
    from sragents.when2tool.prefill_inference import PrefillInference
    print("✓")
    
    print("\n✓ All imports successful!")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    exit(1)
EOF

echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
python << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA is available")
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  ⚠ CUDA is not available (CPU mode will be used)")
EOF

echo ""

# Check file structure
echo "Checking file structure..."
FILES=(
    "src/sragents/when2tool/__init__.py"
    "src/sragents/when2tool/__main__.py"
    "src/sragents/when2tool/split_dataset.py"
    "src/sragents/when2tool/extract_hidden_states.py"
    "src/sragents/when2tool/train_linear_probe.py"
    "src/sragents/when2tool/prefill_inference.py"
    "src/sragents/when2tool/pipeline.py"
    "run_when2tool_pipeline.sh"
    "WHEN2TOOL_README.md"
    "WHEN2TOOL_QUICKSTART.md"
    "WHEN2TOOL_EXAMPLES.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (NOT FOUND)"
    fi
done

echo ""

# Check CLI access
echo "Checking CLI access..."
python -m sragents.when2tool.pipeline --help > /dev/null 2>&1 && echo "  ✓ Can run: python -m sragents.when2tool.pipeline" || echo "  ✗ CLI not accessible"

echo ""
echo "======================================"
echo "✓ Validation complete!"
echo ""
echo "Next steps:"
echo "1. Read the quick start guide:"
echo "   cat WHEN2TOOL_QUICKSTART.md"
echo ""
echo "2. Try the example:"
echo "   python -m sragents.when2tool.pipeline --help"
echo ""
echo "3. Run the full pipeline:"
echo "   python -m sragents.when2tool.pipeline \\
"
echo "       data/bench/instances/bigcodebench.json \\"
echo "       --output-dir ./outputs"
echo ""
