# When2Tool Pipeline for SR-Agents

Implementation of the **When2Tool** method for training tool-calling decision models. This pipeline enables you to:

1. **Split datasets** into 30% train / 70% test
2. **Extract hidden states** from language models at the last token position
3. **Train linear probes** to predict when tools are necessary
4. **Generate responses** with guided prefilling (optional Step 4)

Based on the paper: [LLM Agents Already Know When to Call Tools — Even Without Reasoning](https://arxiv.org/abs/2605.09252)

## Overview

The When2Tool method uses a lightweight linear probe trained on hidden states to make binary decisions: **Do we need to call a tool?**

### Pipeline Steps

#### Step 1: Dataset Splitting
- Splits input dataset into 30% training and 70% test sets
- Preserves reproducibility with random seeding
- Optional: split by dataset type to maintain distribution

**Files:**
- Input: Any JSON or JSONL dataset
- Output: `splits/train_json.json`, `splits/test_json.json`

#### Step 2: Hidden State Extraction
- Runs a forward pass on the model for each prompt
- Extracts hidden states at the **last token position** from all layers
- No additional compute overhead (happens during normal processing)
- Output shape: `(n_samples, n_layers, hidden_dim)`

**Files:**
- Output:
  - `features/train/hidden_states.pt` - PyTorch tensor
  - `features/test/hidden_states.pt` - PyTorch tensor
  - `features/train/items.json` - Original dataset items
  - `features/test/items.json` - Original dataset items

#### Step 3: Linear Probe Training
- Trains an L2-regularized logistic regression on concatenated hidden states
- Uses all layers for richer features
- Validates on a held-out split
- Outputs probability predictions: $p = P(\text{tool\_necessary})$

**Files:**
- Output:
  - `probe/probe.pkl` - Trained sklearn model + scaler
  - `probe/results.json` - Training metrics (accuracy, AUC)

#### Step 4: Prefill-Guided Inference (Optional)
- Uses trained probe to make per-task decisions
- Applies threshold $\tau$: if $p < \tau$, no tool needed; otherwise tool needed
- Generates responses with steering prefills:

**Soft Prefill** (model can override):
- Tool unnecessary: `"I can solve this directly without using a tool."`
- Tool necessary: `"I need to use a tool for this question."`

**Hard Prefill** (forced format):
- Tool unnecessary: `"\boxed{"`
- Tool necessary: `'{"name": "'`

## Installation

1. Clone the repository and navigate to it:
```bash
cd SR-Agents
```

2. Install dependencies (if not already installed):
```bash
pip install torch transformers scikit-learn tqdm
```

## Quick Start

### Option 1: Using the Pipeline Script

```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --output-dir ./my_when2tool_output \
    --train-ratio 0.3 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --device cuda \
    --batch-size 4 \
    --steps 1,2,3
```

### Option 2: Using the Shell Script

```bash
bash run_when2tool_pipeline.sh \
    data/bench/instances/bigcodebench.json \
    --output-dir ./my_when2tool_output \
    --train-ratio 0.3 \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Option 3: Run Individual Steps

#### Step 1: Split Dataset
```bash
python -m sragents.when2tool.split_dataset \
    data/bench/instances/bigcodebench.json \
    --train-ratio 0.3 \
    --output-dir ./splits \
    --seed 42
```

#### Step 2: Extract Hidden States
```bash
python -m sragents.when2tool.extract_hidden_states \
    meta-llama/Llama-3.1-8B-Instruct \
    --train-data ./splits/train_json.json \
    --test-data ./splits/test_json.json \
    --output-dir ./features \
    --batch-size 4 \
    --max-length 512 \
    --device cuda
```

#### Step 3: Train Linear Probe
```bash
python -m sragents.when2tool.train_linear_probe \
    --train-hidden ./features/train/hidden_states.pt \
    --test-hidden ./features/test/hidden_states.pt \
    --train-labels ./features/train/items.json \
    --output-dir ./probe \
    --regularization 1.0 \
    --concatenate-layers \
    --validation-split 0.1
```

#### Step 4: Prefill-Guided Inference (Optional)
```bash
python -m sragents.when2tool.prefill_inference \
    meta-llama/Llama-3.1-8B-Instruct \
    --probe ./probe/probe.pkl \
    --dataset ./splits/test_json.json \
    --hidden-states ./features/test/hidden_states.pt \
    --output-dir ./inference \
    --threshold 0.5 \
    --prefill-mode soft \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 0.95
```

## API Usage

### Python API

```python
from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    dataset_path="data/bench/instances/bigcodebench.json",
    output_base_dir="./when2tool_outputs",
    train_ratio=0.3,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    batch_size_extraction=4,
    max_length=512,
    regularization=1.0,
    concatenate_layers=True,
    validation_split=0.1,
    steps="1,2,3",
)

# Run pipeline
pipeline = When2ToolPipeline(config)
results = pipeline.run()
```

### Individual Module Usage

```python
# Dataset splitting
from sragents.when2tool.split_dataset import split_dataset

train_items, test_items = split_dataset(
    "data.json",
    train_ratio=0.3,
    output_dir="./splits",
)

# Hidden state extraction
from sragents.when2tool.extract_hidden_states import HiddenStateExtractor

extractor = HiddenStateExtractor("meta-llama/Llama-3.1-8B-Instruct")
hidden_states = extractor.extract_hidden_states(
    texts=["What is 2+2?", "What is the capital of France?"],
    batch_size=4,
    max_length=512,
)

# Linear probe training
from sragents.when2tool.train_linear_probe import LinearProbe

probe = LinearProbe(n_features=4096)  # Example feature size
metrics = probe.fit(X_train, y_train)
predictions, probabilities = probe.predict(X_test)

# Prefill inference
from sragents.when2tool.prefill_inference import PrefillInference

inference = PrefillInference(
    "meta-llama/Llama-3.1-8B-Instruct",
    probe_path="./probe/probe.pkl",
)
result = inference.generate_with_prefill(
    prompt="What is 2+2?",
    hidden_state=hidden_states[0],
    threshold=0.5,
    prefill_mode="soft",
)
```

## Output Structure

After running the pipeline, you'll get:

```
when2tool_outputs/
├── config.json                      # Pipeline configuration
├── pipeline_results.json            # Summary of all results
├── splits/
│   ├── train_json.json             # 30% training data
│   └── test_json.json              # 70% test data
├── features/
│   ├── train/
│   │   ├── hidden_states.pt        # (n_train, n_layers, hidden_dim)
│   │   ├── items.json              # Training items with labels
│   │   └── metadata.json           # Feature information
│   └── test/
│       ├── hidden_states.pt        # (n_test, n_layers, hidden_dim)
│       ├── items.json              # Test items
│       └── metadata.json
├── probe/
│   ├── probe.pkl                   # Trained LinearProbe object
│   └── results.json                # Training metrics (accuracy, AUC)
└── inference/                      # (Optional, if Step 4 run)
    └── threshold_0.5/
        ├── results_prefill_soft_threshold_0.5.jsonl
        └── stats_prefill_soft_threshold_0.5.json
```

## Configuration Options

### Main Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_path` | - | Path to input dataset (JSON or JSONL) |
| `output_base_dir` | `./when2tool_outputs` | Output directory |
| `train_ratio` | `0.3` | Training data ratio (0.3 = 30%) |
| `model_name` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `device` | `cuda` | Device (cuda/cpu) |

### Feature Extraction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size_extraction` | `4` | Batch size for hidden state extraction |
| `max_length` | `512` | Maximum sequence length |

### Probe Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `regularization` | `1.0` | L2 regularization strength (1/C) |
| `concatenate_layers` | `True` | Concatenate all layers or use last only |
| `validation_split` | `0.1` | Validation set ratio |

### Inference (Optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prefill_mode` | `soft` | Prefill strategy (soft/hard) |
| `threshold_values` | `[0.5]` | Thresholds to sweep |
| `max_new_tokens` | `128` | Max tokens to generate |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.95` | Nucleus sampling parameter |

## Supported Models

The pipeline works with any HuggingFace causal language model:

- **Qwen Series**: `Qwen/Qwen3-4B-Instruct`, `Qwen/Qwen3-14B`, etc.
- **Llama Series**: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`
- **Other**: Any model on HuggingFace that supports `AutoModelForCausalLM`

## Interpreting Results

### Training Metrics (probe/results.json)

- `train_accuracy`: Accuracy on training set
- `val_accuracy`: Accuracy on validation set
- `train_auc`: Area under ROC curve on training
- `val_auc`: Area under ROC curve on validation
- **Target**: AUROC > 0.85 indicates good probe performance

### Inference Statistics (inference/.../stats_*.json)

- `tool_calls`: Number of tasks where tool was called
- `tool_call_rate`: Percentage of tasks calling tools
- `avg_tool_probability`: Average probe probability
- `threshold`: Decision threshold used

## Example: Complete Workflow

```bash
# 1. Split dataset
python -m sragents.when2tool.split_dataset \
    data/bench/instances/bigcodebench.json \
    --output-dir ./splits

# 2. Extract features
python -m sragents.when2tool.extract_hidden_states \
    Qwen/Qwen3-4B-Instruct \
    --train-data ./splits/train_json.json \
    --test-data ./splits/test_json.json \
    --output-dir ./features

# 3. Train probe
python -m sragents.when2tool.train_linear_probe \
    --train-hidden ./features/train/hidden_states.pt \
    --test-hidden ./features/test/hidden_states.pt \
    --train-labels ./features/train/items.json \
    --output-dir ./probe

# 4. Run inference with multiple thresholds
for threshold in 0.3 0.5 0.7 0.9; do
    python -m sragents.when2tool.prefill_inference \
        Qwen/Qwen3-4B-Instruct \
        --probe ./probe/probe.pkl \
        --dataset ./splits/test_json.json \
        --hidden-states ./features/test/hidden_states.pt \
        --output-dir ./inference \
        --threshold $threshold \
        --prefill-mode soft
done
```

## Performance Notes

- **Step 1 (Splitting)**: < 1 second
- **Step 2 (Feature Extraction)**: ~1-5 minutes (depends on model size and batch size)
- **Step 3 (Probe Training)**: < 1 minute
- **Step 4 (Inference)**: ~5-15 minutes (generation is slow)

**Memory Requirements:**
- Model: ~16GB GPU for 8B models, ~40GB for 70B models
- Features: ~500MB for 1000 samples with 32-layer model

## Troubleshooting

### Out of Memory during feature extraction
```bash
# Reduce batch size
--batch-size 2  # or even 1
```

### Model not found
```bash
# Make sure you have HuggingFace credentials configured
huggingface-cli login
```

### Labels not being generated correctly
Ensure your dataset items have a `tool_necessary` field:
```json
{
  "question": "What is 2+2?",
  "tool_necessary": 0
}
```

## Citation

```bibtex
@article{sun2026when2tool,
  title={LLM Agents Already Know When to Call Tools -- Even Without Reasoning},
  author={Sun, Chung-En and Liu, Linbo and Yan, Ge and Wang, Zimo and Weng, Tsui-Wei},
  journal={arXiv preprint arXiv:2605.09252},
  year={2026},
  url={https://arxiv.org/abs/2605.09252}
}
```

## References

- [When2Tool GitHub](https://github.com/Trustworthy-ML-Lab/when2tool)
- [When2Tool Dataset on HuggingFace](https://huggingface.co/datasets/cesun/When2Tool)
- [Paper](https://arxiv.org/abs/2605.09252)

## License

See the main SR-Agents repository LICENSE file.
