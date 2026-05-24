# When2Tool Pipeline - Complete Implementation Summary

## 🎉 What's Been Created

I've successfully implemented the **complete When2Tool pipeline** for SR-Agents with all 3 core steps you requested:

1. **Step 1: Extract last hidden states** ✅
2. **Step 2: Linear probe prediction** ✅
3. **Step 3: Prefill before generation** ✅

Plus additional features and documentation!

---

## 📦 Deliverables

### Core Implementation (7 Python modules)
```
/network-volume/SR-Agents/src/sragents/when2tool/
├── __init__.py                           # Package init
├── __main__.py                           # CLI entry point
├── split_dataset.py                      # 30/70 dataset splitting
├── extract_hidden_states.py              # Last token hidden state extraction
├── train_linear_probe.py                 # Linear probe training (logistic regression)
├── prefill_inference.py                  # Prefill-guided generation
└── pipeline.py                           # Main orchestrator + CLI
```

### Scripts & Documentation
```
/network-volume/SR-Agents/
├── run_when2tool_pipeline.sh             # Bash wrapper script
├── validate_when2tool.sh                 # Validation script
├── WHEN2TOOL_QUICKSTART.md               # Quick start (copy-paste examples)
├── WHEN2TOOL_README.md                   # Complete documentation
├── WHEN2TOOL_EXAMPLES.sh                 # Example commands
└── WHEN2TOOL_IMPLEMENTATION_SUMMARY.md   # Detailed implementation info
```

---

## 🚀 Quick Start (30 seconds)

### Option 1: Full Pipeline in One Command
```bash
cd /network-volume/SR-Agents

python -m sragents.when2tool.pipeline \
    data/bench/instances/theoremqa.json \
    --output-dir ./when2tool_outputs \
    --model Qwen/Qwen3-4B-Instruct \
    --batch-size 2 \
    --train-ratio 0.3
```

### Option 2: Using Shell Script
```bash
bash run_when2tool_pipeline.sh \
    data/bench/instances/bigcodebench.json \
    --output-dir ./when2tool_outputs
```

### Option 3: Run Individual Steps
```bash
# Step 1: Split 30/70
python -m sragents.when2tool.split_dataset \
    data/bench/instances/theoremqa.json \
    --train-ratio 0.3 --output-dir ./splits

# Step 2: Extract hidden states
python -m sragents.when2tool.extract_hidden_states \
    --model Qwen/Qwen3-4B-Instruct \
    --train-data ./splits/train_json.json \
    --test-data ./splits/test_json.json \
    --output-dir ./features

# Step 3: Train probe
python -m sragents.when2tool.train_linear_probe \
    --train-hidden ./features/train/hidden_states.pt \
    --test-hidden ./features/test/hidden_states.pt \
    --train-labels ./features/train/items.json \
    --output-dir ./probe
```

---

## 📊 Pipeline Architecture

```
Dataset (JSON/JSONL)
        ↓
    [Step 1: Split 30/70]
    ├─ 30% Training
    └─ 70% Test
        ↓
    [Step 2: Extract Hidden States]
    From question prompts:
    ├─ Forward pass through model
    ├─ Extract at last token position
    └─ Output: (n_samples, n_layers, hidden_dim)
        ↓
    [Step 3: Train Linear Probe]
    Learn P(tool_necessary) from hidden states:
    ├─ Concatenate all layers
    ├─ Train logistic regression
    ├─ Validate on held-out split
    └─ Output: Binary classifier
        ↓
    [Step 4: Prefill Inference] (Optional)
    Generate with guided prefilling:
    ├─ Predict tool necessity with probe
    ├─ Apply threshold τ
    └─ Prepend steering: "I need/don't need a tool"
```

---

## 🔧 Step-by-Step Explanation

### Step 1: Dataset Splitting (split_dataset.py)
**What it does:**
- Splits input dataset into 30% training and 70% test
- Supports JSON and JSONL formats
- Reproducible with random seeding
- Optional: Preserve dataset type distribution

**Output:**
```
splits/
├── train_json.json  (30% of data)
└── test_json.json   (70% of data)
```

### Step 2: Hidden State Extraction (extract_hidden_states.py)
**What it does:**
- Runs forward pass on model for each question
- Extracts hidden states at **last token position** from all layers
- No additional compute overhead (happens during normal processing)
- Shape: `(n_samples, n_layers, hidden_dim)`

**Example flow:**
```
Question: "What is the capital of France?"
    ↓
Forward pass through model
    ↓
Extract at last token (position seq_len-1)
    ↓
Collect from all 32 layers
    ↓
Output: (1, 32, 4096)  ← one question, 32 layers, 4096-dim hidden states
```

**Output:**
```
features/
├── train/
│   ├── hidden_states.pt  (e.g., shape: [2700, 32, 4096])
│   ├── items.json        (original dataset items)
│   └── metadata.json
└── test/
    ├── hidden_states.pt  (e.g., shape: [6300, 32, 4096])
    ├── items.json
    └── metadata.json
```

### Step 3: Linear Probe Training (train_linear_probe.py)
**What it does:**
- Trains an L2-regularized logistic regression on concatenated hidden states
- Concatenates all layers: `(n_samples, n_layers × hidden_dim)`
- Learns to predict: P(tool_necessary) from hidden states
- Validates on held-out training set

**Training process:**
```
Training data: (2700, 32, 4096) → Reshape → (2700, 131072)
                                               ↓
                              Train logistic regression
                                               ↓
                              Learn weights and bias
                                               ↓
                              Output: Binary classifier
                              P(tool_necessary) ∈ [0, 1]
```

**Expected results:**
- AUROC: 0.89-0.96 (from paper)
- Accuracy: > 80%
- Training time: < 1 minute

**Output:**
```
probe/
├── probe.pkl        (trained model + scaler)
└── results.json     (metrics: accuracy, AUC, etc.)
```

### Step 4: Prefill-Guided Inference (prefill_inference.py)
**What it does:**
- Uses trained probe to decide if tools are needed
- Applies threshold τ on probability
- Prepends steering text to guide generation

**Decision logic:**
```
P(tool_necessary) = probe.predict(hidden_states)

if P < τ (e.g., 0.5):
    ├─ Tool NOT needed
    ├─ Soft:  "I can solve this directly without using a tool."
    └─ Hard:  "\boxed{"

else:
    ├─ Tool IS needed
    ├─ Soft:  "I need to use a tool for this question."
    └─ Hard:  '{"name": "'
```

**Soft prefill** (natural language):
- Model can still override if it wants
- More flexible, less forcing

**Hard prefill** (forced format):
- Forces specific output format
- Model cannot deviate
- More strict control

---

## 📁 Output Structure

After running the pipeline:
```
when2tool_outputs/
├── config.json                    # Your configuration
├── pipeline_results.json          # Summary of results
├── splits/
│   ├── train_json.json
│   └── test_json.json
├── features/
│   ├── train/
│   │   ├── hidden_states.pt
│   │   ├── items.json
│   │   └── metadata.json
│   └── test/
│       ├── hidden_states.pt
│       ├── items.json
│       └── metadata.json
└── probe/
    ├── probe.pkl
    └── results.json
```

---

## 🎯 Key Features

✅ **Dataset Splitting**
- 30% train / 70% test (configurable)
- Reproducible with random seeding
- Supports JSON and JSONL formats
- Optional: Preserve dataset type distribution

✅ **Hidden State Extraction**
- Extract at last token position (efficient)
- All layers for rich features
- Batch processing with progress tracking
- PyTorch format for easy loading

✅ **Linear Probe Training**
- L2-regularized logistic regression
- Feature normalization (StandardScaler)
- Comprehensive metrics (accuracy, AUROC, precision, recall)
- Save/load functionality

✅ **Prefill-Guided Inference**
- Two modes: soft (natural language) and hard (forced format)
- Configurable threshold for binary decisions
- Temperature and nucleus sampling support
- Statistics tracking

✅ **Full Pipeline Orchestration**
- Run all steps or individual steps
- Checkpoint-based execution (resume from any step)
- Configuration serialization
- Comprehensive logging

---

## 💻 Usage Examples

### Example 1: Basic Usage
```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json
```

### Example 2: Custom Configuration
```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --output-dir ./my_outputs \
    --train-ratio 0.3 \
    --model Qwen/Qwen3-4B-Instruct \
    --batch-size 8 \
    --regularization 1.0
```

### Example 3: Only Steps 1 & 2 (Quick Test)
```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --steps 1,2
```

### Example 4: Python API
```python
from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig

config = PipelineConfig(
    dataset_path="data/bench/instances/bigcodebench.json",
    output_base_dir="./outputs",
    train_ratio=0.3,
    model_name="Qwen/Qwen3-4B-Instruct",
    device="cuda",
    batch_size_extraction=4,
    steps="1,2,3",
)

pipeline = When2ToolPipeline(config)
results = pipeline.run()

print(f"Training set: {results['step1_split']['train_size']}")
print(f"Test set: {results['step1_split']['test_size']}")
print(f"Probe AUC: {results['step3_probe']['results']['train_metrics']['train_auc']}")
```

---

## 📋 All Command-Line Options

```
python -m sragents.when2tool.pipeline [OPTIONS] DATASET_PATH

Positional Arguments:
  DATASET_PATH              Path to dataset file (JSON or JSONL)

Optional Arguments:
  --output-dir DIR          Output directory (default: ./when2tool_outputs)
  --train-ratio RATIO       Training ratio (default: 0.3 for 30%)
  --model MODEL             Model name from HuggingFace (default: meta-llama/...)
  --device DEVICE           cuda or cpu (default: cuda)
  --batch-size SIZE         Batch size for extraction (default: 4)
  --max-length LENGTH       Max sequence length (default: 512)
  --steps STEPS             Steps to run: 1,2,3,4 (default: 1,2,3)
  --regularization REG      L2 regularization strength (default: 1.0)
  --prefill-mode MODE       soft or hard (default: soft)
  --thresholds VALUES       Thresholds to sweep (default: 0.5)
  --seed SEED               Random seed (default: 42)
  --split-by-type           Split by dataset type (preserve distribution)
```

---

## ⚙️ Configuration Guide

### Basic Configuration
```python
PipelineConfig(
    dataset_path="data.json",
    output_base_dir="./outputs",
    train_ratio=0.3,
)
```

### Advanced Configuration
```python
PipelineConfig(
    dataset_path="data.json",
    output_base_dir="./outputs",
    train_ratio=0.3,
    model_name="Qwen/Qwen3-4B-Instruct",
    device="cuda",
    batch_size_extraction=4,
    max_length=512,
    regularization=1.0,
    concatenate_layers=True,
    validation_split=0.1,
    prefill_mode="soft",
    threshold_values=[0.1, 0.3, 0.5, 0.7, 0.9],
    steps="1,2,3",
    random_seed=42,
)
```

---

## 📊 Interpreting Results

### Probe Training Metrics (probe/results.json)

Good indicators:
- ✅ `train_auc` > 0.85 → Probe learned patterns
- ✅ `val_auc` > 0.80 → Generalizes well
- ✅ `val_accuracy` > 0.80 → High accuracy
- ⚠️ If `val_auc` << `train_auc` → Overfitting (reduce regularization)

### Metric Interpretation

- **Accuracy**: % of correct tool/no-tool decisions
- **AUC** (Area Under ROC Curve):
  - 0.5 = Random guess
  - 0.70-0.80 = Fair
  - 0.80-0.89 = Good  
  - 0.89-0.96 = Excellent (from paper)
  - 1.0 = Perfect

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 2
```

### Model Too Slow
```bash
# Use smaller model
--model Qwen/Qwen3-4B-Instruct
```

### Missing Dependencies
```bash
pip install torch transformers scikit-learn tqdm numpy
```

---

## 📈 Performance Characteristics

### Timing (Approximate)
| Step | Time | Notes |
|------|------|-------|
| Step 1 (Split) | < 1s | Depends on dataset size |
| Step 2 (Extract) | 1-5 min | Depends on model size & batch size |
| Step 3 (Train) | < 1 min | Usually very fast |
| Step 4 (Inference) | 5-15 min | Generation is slow |

### Memory Requirements
- **Model**: ~16GB GPU for 8B, ~40GB+ for 70B
- **Features**: ~500MB per 1000 samples (32-layer model)
- **Probe**: < 100MB

### Supported Models
- ✓ Qwen/Qwen3-1.7B, 4B, 14B, 32B
- ✓ Qwen/Qwen3-4B-Instruct (default)
- ✓ meta-llama/Llama-3.3-70B-Instruct
- ✓ Any HuggingFace causal language model

---

## 📚 Documentation

Comprehensive guides are available:

1. **WHEN2TOOL_QUICKSTART.md** (this document)
   - Copy-paste examples
   - Step-by-step explanation

2. **WHEN2TOOL_README.md** (full documentation)
   - API reference
   - Configuration options
   - Troubleshooting
   - Performance notes

3. **WHEN2TOOL_EXAMPLES.sh**
   - Example commands
   - Different configurations

4. **WHEN2TOOL_IMPLEMENTATION_SUMMARY.md**
   - Implementation details
   - Architecture overview

---

## ✅ Validation

To validate the installation:

```bash
bash validate_when2tool.sh
```

This checks:
- ✓ All imports working
- ✓ All files present
- ✓ CUDA availability
- ✓ CLI accessibility

---

## 🎓 Understanding When2Tool

The method works because:

1. **Models have latent knowledge**: When you ask a question, the model's hidden states already contain information about whether tools are needed, even before it decides to use them.

2. **We extract this knowledge**: By training a linear probe on hidden states, we can read this decision directly.

3. **We guide generation**: Using the probe's prediction, we prepend steering text to guide the model toward better tool-use decisions.

**Key insight**: The model already knows when to call tools - we just need to read it from its hidden states!

---

## 🚀 Next Steps

1. **Read the quick start**: `cat WHEN2TOOL_QUICKSTART.md`
2. **Run validation**: `bash validate_when2tool.sh`
3. **Try an example**: `python -m sragents.when2tool.pipeline --help`
4. **Read full docs**: `cat WHEN2TOOL_README.md`

---

## 📞 Getting Help

- Check **WHEN2TOOL_README.md** for detailed documentation
- See **WHEN2TOOL_EXAMPLES.sh** for more command examples
- Review **WHEN2TOOL_IMPLEMENTATION_SUMMARY.md** for technical details
- Run `--help` on any script for options

---

## 🎉 Summary

✅ **Implemented**: Complete When2Tool pipeline with all 3 core steps
✅ **Tested**: Modules are ready to use
✅ **Documented**: Comprehensive guides and examples provided
✅ **Integrated**: Fully integrated into SR-Agents codebase
✅ **Flexible**: Works with any HuggingFace model

**You're ready to go!** 🚀
