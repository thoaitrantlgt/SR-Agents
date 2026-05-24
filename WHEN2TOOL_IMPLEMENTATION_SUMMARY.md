"""
Summary of When2Tool Pipeline Implementation for SR-Agents

This module implements the complete When2Tool pipeline for training
tool-calling decision models based on hidden states.

Components Created:
==================

1. CORE MODULES (src/sragents/when2tool/)
   ├── split_dataset.py
   │   └─ Dataset splitting into 30% train / 70% test
   │   └─ Support for JSON/JSONL formats
   │   └─ Optional: Split by dataset type to preserve distribution
   │
   ├── extract_hidden_states.py
   │   └─ HiddenStateExtractor class for extracting last token states
   │   └─ Extract from all layers: (n_samples, n_layers, hidden_dim)
   │   └─ Batch processing with progress bars
   │   └─ Save to PyTorch format
   │
   ├── train_linear_probe.py
   │   └─ LinearProbe class wrapping sklearn's LogisticRegression
   │   └─ Feature scaling with StandardScaler
   │   └─ Train/validation split with metrics (accuracy, AUC)
   │   └─ Save/load probe with pickle
   │
   ├── prefill_inference.py
   │   └─ PrefillInference class for probe-guided generation
   │   └─ Soft prefill: "I can solve..." / "I need a tool..."
   │   └─ Hard prefill: "\boxed{" / '{"name": "'
   │   └─ Configurable threshold for tool necessity
   │
   ├── pipeline.py
   │   └─ When2ToolPipeline orchestrator class
   │   └─ PipelineConfig dataclass for configuration
   │   └─ Run all steps or individual steps
   │   └─ CLI interface with full argument support
   │
   ├── __init__.py
   ├── __main__.py
   └─ Utils and helpers

2. SHELL SCRIPTS
   ├── run_when2tool_pipeline.sh
   │   └─ Bash wrapper for easy execution
   │   └─ Parameter parsing and validation
   │
   └── WHEN2TOOL_EXAMPLES.sh
       └─ Example commands for different scenarios

3. DOCUMENTATION
   ├── WHEN2TOOL_README.md
   │   └─ Complete documentation with API reference
   │   └─ Configuration options and troubleshooting
   │   └─ Performance notes and memory requirements
   │
   ├── WHEN2TOOL_QUICKSTART.md
   │   └─ Quick start guide with copy-paste commands
   │   └─ Step-by-step explanation of pipeline
   │   └─ Configuration examples
   │
   └── This file (IMPLEMENTATION_SUMMARY.md)

Pipeline Architecture:
======================

[Input Dataset]
        ↓
[Step 1: Split Dataset] → train_json.json, test_json.json
        ↓
[Step 2: Extract Hidden States]
├─ Train: (n_train, n_layers, hidden_dim)
└─ Test: (n_test, n_layers, hidden_dim)
        ↓
[Step 3: Train Linear Probe]
├─ Input: Hidden states + binary labels
├─ Output: Trained probe (sklearn LogisticRegression)
└─ Metrics: Accuracy, AUROC
        ↓
[Step 4: Prefill Inference] (Optional)
├─ Predict tool necessity using probe
├─ Apply threshold τ
├─ Generate with guided prefilling
└─ Output: Generations with decision statistics

Key Features:
=============

✓ Dataset Splitting
  - 30% train / 70% test (configurable)
  - Random seeding for reproducibility
  - Support for JSON and JSONL formats
  - Optional: Preserve dataset type distribution

✓ Hidden State Extraction
  - Extract at last token position (no extra compute)
  - Extract from all layers for rich features
  - Batch processing for efficiency
  - Automatic progress tracking

✓ Linear Probe Training
  - L2-regularized logistic regression
  - Feature normalization with StandardScaler
  - Validation split for early stopping
  - Comprehensive metrics (accuracy, AUC, ROC)

✓ Prefill-Guided Inference
  - Two prefill modes: soft (natural language) and hard (forced format)
  - Configurable decision threshold
  - Temperature and top-p sampling
  - Statistics tracking

✓ Full Pipeline Orchestration
  - Run all steps in sequence or individually
  - Checkpoint-based execution (resume from any step)
  - Configuration serialization
  - Comprehensive logging and progress tracking

Usage Examples:
===============

1. Quick Start (Full Pipeline)
   python -m sragents.when2tool.pipeline \\
       data/bench/instances/bigcodebench.json \\
       --output-dir ./outputs \\
       --train-ratio 0.3

2. Individual Steps
   # Step 1: Split
   python -m sragents.when2tool.split_dataset \\
       data.json --output-dir ./splits
   
   # Step 2: Extract
   python -m sragents.when2tool.extract_hidden_states \\
       Qwen/Qwen3-4B-Instruct \\
       --train-data ./splits/train_json.json \\
       --test-data ./splits/test_json.json \\
       --output-dir ./features
   
   # Step 3: Train
   python -m sragents.when2tool.train_linear_probe \\
       --train-hidden ./features/train/hidden_states.pt \\
       --test-hidden ./features/test/hidden_states.pt \\
       --train-labels ./features/train/items.json \\
       --output-dir ./probe

3. Python API
   from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig
   
   config = PipelineConfig(
       dataset_path="data.json",
       output_base_dir="./outputs",
       train_ratio=0.3,
   )
   pipeline = When2ToolPipeline(config)
   results = pipeline.run()

4. Shell Script
   bash run_when2tool_pipeline.sh data.json --output-dir ./outputs

Files Created:
==============

Core Implementation:
  /network-volume/SR-Agents/src/sragents/when2tool/__init__.py
  /network-volume/SR-Agents/src/sragents/when2tool/__main__.py
  /network-volume/SR-Agents/src/sragents/when2tool/split_dataset.py
  /network-volume/SR-Agents/src/sragents/when2tool/extract_hidden_states.py
  /network-volume/SR-Agents/src/sragents/when2tool/train_linear_probe.py
  /network-volume/SR-Agents/src/sragents/when2tool/prefill_inference.py
  /network-volume/SR-Agents/src/sragents/when2tool/pipeline.py

Scripts:
  /network-volume/SR-Agents/run_when2tool_pipeline.sh
  /network-volume/SR-Agents/WHEN2TOOL_EXAMPLES.sh

Documentation:
  /network-volume/SR-Agents/WHEN2TOOL_README.md (6500+ words)
  /network-volume/SR-Agents/WHEN2TOOL_QUICKSTART.md
  /network-volume/SR-Agents/IMPLEMENTATION_SUMMARY.md (this file)

Output Structure:
=================

when2tool_outputs/
├── config.json                    # Pipeline configuration
├── pipeline_results.json          # Summary of all results
├── splits/
│   ├── train_json.json           # 30% training set
│   └── test_json.json            # 70% test set
├── features/
│   ├── train/
│   │   ├── hidden_states.pt      # (n_train, n_layers, hidden_dim)
│   │   ├── items.json            # Training items with labels
│   │   └── metadata.json
│   └── test/
│       ├── hidden_states.pt      # (n_test, n_layers, hidden_dim)
│       ├── items.json
│       └── metadata.json
└── probe/
    ├── probe.pkl                 # Trained LinearProbe
    └── results.json              # Training metrics

Configuration Options:
======================

Data Configuration:
  - dataset_path: Path to input dataset
  - train_ratio: Training data ratio (0.3 = 30%)
  - split_by_dataset_type: Preserve distribution for multi-type datasets
  - random_seed: Reproducibility seed

Model Configuration:
  - model_name: HuggingFace model identifier
  - device: cuda or cpu
  - dtype: torch.float32 or torch.float16

Feature Extraction:
  - batch_size_extraction: Batch size for extraction
  - max_length: Maximum sequence length

Probe Training:
  - regularization: L2 regularization strength
  - concatenate_layers: Use all layers vs last layer only
  - validation_split: Held-out validation ratio

Pipeline Control:
  - steps: Which steps to run (1,2,3,4)
  - output_base_dir: Output directory

Performance Notes:
==================

Timing (approximate):
  - Step 1 (Split): < 1 second
  - Step 2 (Extract): 1-5 minutes (depends on model and batch size)
  - Step 3 (Train): < 1 minute
  - Step 4 (Inference): 5-15 minutes (generation is slow)

Memory Requirements:
  - Model: ~16GB GPU for 8B models, ~40GB+ for 70B models
  - Features: ~500MB for 1000 samples with 32-layer model
  - Probe: < 100MB

Supported Models:
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen3-4B-Instruct
  - Qwen/Qwen3-14B
  - Qwen/Qwen3-32B
  - meta-llama/Llama-3.1-8B-Instruct
  - meta-llama/Llama-3.3-70B-Instruct
  - (Any HuggingFace causal language model)

Key Implementation Details:
==========================

Hidden State Extraction:
  - Extracts at position: seq_len - 1 (last non-padding token)
  - From all layers: tuple of (batch_size, seq_len, hidden_dim)
  - Output shape: (n_samples, n_layers, hidden_dim)
  - No additional forward passes needed

Linear Probe:
  - Feature concatenation: (n_layers * hidden_dim,) per sample
  - Training: L2-regularized logistic regression
  - Scaler: StandardScaler for feature normalization
  - Output: Binary probability P(tool_necessary)

Prefilling:
  - Soft: Prepend natural language guidance (model can override)
  - Hard: Prepend forced format (cannot be overridden)
  - Threshold: Decision boundary for binary classification

Reproducibility:
  - Random seed controls dataset shuffling
  - Sklearn uses fixed random_state=42
  - PyTorch operations are deterministic with CUDA

Error Handling:
  - Graceful handling of missing files
  - Validation of input formats
  - Informative error messages
  - Progress tracking for long operations

Testing & Validation:
====================

To test the implementation:

1. Quick test (CPU only):
   python -m sragents.when2tool.pipeline \\
       data/bench/instances/bigcodebench.json \\
       --output-dir ./test_output \\
       --train-ratio 0.1 \\  # Use 10% for speed
       --device cpu \\
       --batch-size 1 \\
       --steps 1,2,3

2. Check output structure:
   ls -R ./test_output/

3. Verify metrics:
   cat ./test_output/probe/results.json

4. Inspect sample features:
   python -c "import torch; x = torch.load('./test_output/features/train/hidden_states.pt'); print(x.shape)"

Dependencies:
==============

Required:
  - torch >= 2.4.0
  - transformers >= 4.45.0
  - scikit-learn >= 1.0.0
  - numpy >= 2.0.0
  - tqdm >= 4.67.0

Optional:
  - cuda toolkit for GPU acceleration

All dependencies are already in SR-Agents pyproject.toml

Integration with SR-Agents:
===========================

The When2Tool pipeline is fully integrated as a new module in SR-Agents:

Location: src/sragents/when2tool/

Can be imported as:
  from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig
  from sragents.when2tool.split_dataset import split_dataset
  from sragents.when2tool.extract_hidden_states import HiddenStateExtractor
  from sragents.when2tool.train_linear_probe import LinearProbe
  from sragents.when2tool.prefill_inference import PrefillInference

Next Steps:
===========

1. Review the implementation:
   - cat WHEN2TOOL_QUICKSTART.md
   - cat WHEN2TOOL_README.md

2. Try a quick example:
   python -m sragents.when2tool.pipeline data/bench/instances/bigcodebench.json

3. Explore individual modules:
   python -m sragents.when2tool.split_dataset --help
   python -m sragents.when2tool.extract_hidden_states --help

4. Integrate with your workflow:
   - Use PipelineConfig for custom configurations
   - Extend individual modules as needed
   - Build on top of the API

References:
===========

Paper: LLM Agents Already Know When to Call Tools — Even Without Reasoning
  https://arxiv.org/abs/2605.09252

Official Repository:
  https://github.com/Trustworthy-ML-Lab/when2tool

Dataset on HuggingFace:
  https://huggingface.co/datasets/cesun/When2Tool

Citation:
  @article{sun2026when2tool,
    title={LLM Agents Already Know When to Call Tools -- Even Without Reasoning},
    author={Sun, Chung-En and Liu, Linbo and Yan, Ge and Wang, Zimo and Weng, Tsui-Wei},
    journal={arXiv preprint arXiv:2605.09252},
    year={2026}
  }
"""

__version__ = "1.0.0"
__author__ = "SR-Agents Contributors"
__date__ = "2026-05-23"

if __name__ == "__main__":
    print(__doc__)
