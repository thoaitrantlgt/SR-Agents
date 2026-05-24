# When2Tool Pipeline - Quick Start Guide

## 📋 What Was Created

I've implemented the **When2Tool** pipeline for SR-Agents with 4 complete steps:

### Files Created

```
/network-volume/SR-Agents/src/sragents/when2tool/
├── __init__.py                      # Package initialization
├── __main__.py                      # CLI entry point
├── split_dataset.py                 # Step 1: Dataset splitting
├── extract_hidden_states.py         # Step 2: Hidden state extraction
├── train_linear_probe.py            # Step 3: Linear probe training
├── prefill_inference.py             # Step 4: Prefill-guided inference
└── pipeline.py                      # Main orchestrator

/network-volume/SR-Agents/
├── run_when2tool_pipeline.sh        # Shell script wrapper
├── WHEN2TOOL_README.md              # Full documentation
└── WHEN2TOOL_EXAMPLES.sh            # Example commands
```

## 🚀 Quick Start (Copy & Paste)

### Option 1: Full Pipeline in One Command

```bash
cd /network-volume/SR-Agents

# Run all 3 steps (split, extract, train probe)
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --output-dir ./when2tool_outputs \
    --train-ratio 0.3 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --steps 1,2,3
```

### Option 2: Using Shell Script

```bash
cd /network-volume/SR-Agents

bash run_when2tool_pipeline.sh \
    data/bench/instances/bigcodebench.json \
    --output-dir ./when2tool_outputs \
    --train-ratio 0.3 \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Option 3: Run Steps Individually

```bash
cd /network-volume/SR-Agents

# Step 1: Split 30% train / 70% test
python -m sragents.when2tool.split_dataset \
    data/bench/instances/bigcodebench.json \
    --train-ratio 0.3 \
    --output-dir ./splits

# Step 2: Extract hidden states from all layers
python -m sragents.when2tool.extract_hidden_states \
    meta-llama/Llama-3.1-8B-Instruct \
    --train-data ./splits/train_json.json \
    --test-data ./splits/test_json.json \
    --output-dir ./features

# Step 3: Train linear probe
python -m sragents.when2tool.train_linear_probe \
    --train-hidden ./features/train/hidden_states.pt \
    --test-hidden ./features/test/hidden_states.pt \
    --train-labels ./features/train/items.json \
    --output-dir ./probe
```

## 📊 Pipeline Overview

```
Dataset (JSON/JSONL)
        ↓
[Step 1] Split 30/70
        ↓
     Train Set (30%)     Test Set (70%)
        ↓                      ↓
[Step 2] Extract Hidden States (both sets)
        ↓                      ↓
   Hidden States         Hidden States
   (n_train, n_layers, hidden_dim)
        ↓                      ↓
[Step 3] Train Probe
        ├─ Uses training hidden states
        ├─ Learns linear probe weights
        └─ Validates on held-out split
        ↓
   Trained Probe (sklearn LogisticRegression)
        ↓
[Step 4] Prefill Inference (Optional)
   ├─ Predict tool necessity with probe
   ├─ Apply threshold τ
   └─ Generate with steering prefill
```

## 🎯 What Each Step Does

### Step 1: Dataset Splitting
- **Input:** Any JSON/JSONL dataset
- **Output:** 30% train, 70% test sets
- **Time:** < 1 second
- **Purpose:** Create reproducible splits for training probe

### Step 2: Hidden State Extraction
- **Input:** Train/test datasets + model
- **Output:** Hidden states at last token position for all layers
- **Shape:** `(n_samples, n_layers, hidden_dim)`
- **Time:** 1-5 minutes (depends on model size)
- **Purpose:** Extract features for linear probe

### Step 3: Linear Probe Training
- **Input:** Hidden states + binary labels (tool_necessary: 0/1)
- **Output:** Trained sklearn LogisticRegression probe
- **Expected:** AUROC > 0.85
- **Time:** < 1 minute
- **Purpose:** Learn decision boundary for tool necessity

### Step 4: Prefill Inference (Optional)
- **Input:** Trained probe + test dataset
- **Output:** Generations with guided prefilling
- **Modes:** Soft (natural language) or Hard (forced format)
- **Time:** 5-15 minutes (generation is slow)
- **Purpose:** Generate responses with probe-guided steering

## 📁 Output Structure

After running, you'll find:

```
when2tool_outputs/
├── config.json                    # Your pipeline configuration
├── splits/                        # Step 1 output
│   ├── train_json.json           # 30% training data
│   └── test_json.json            # 70% test data
├── features/                      # Step 2 output
│   ├── train/
│   │   ├── hidden_states.pt      # (2700, 32, 4096) for example
│   │   └── items.json
│   └── test/
│       ├── hidden_states.pt      # (6300, 32, 4096) for example
│       └── items.json
└── probe/                         # Step 3 output
    ├── probe.pkl                 # Trained model + scaler
    └── results.json              # Metrics (accuracy, AUC)
```

## 💻 Configuration Examples

### Small Model (Faster, Lower Memory)
```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --model Qwen/Qwen3-4B-Instruct \
    --batch-size 8 \
    --device cuda
```

### Large Model (Better Quality)
```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --batch-size 1 \  # Reduce batch size for large models
    --device cuda
```

### CPU Mode (No GPU)
```bash
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --device cpu \
    --batch-size 1
```

### Different Train/Test Ratios
```bash
# 20/80 split
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --train-ratio 0.2

# 50/50 split
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --train-ratio 0.5
```

## 📖 Python API

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

print(f"Train set: {results['step1_split']['train_size']}")
print(f"Test set: {results['step1_split']['test_size']}")
```

## ⚙️ All Available Options

```
positional arguments:
  dataset_path          Path to dataset file (JSON or JSONL)

optional arguments:
  --output-dir DIR      Output directory (default: ./when2tool_outputs)
  --train-ratio RATIO   Training ratio (default: 0.3 for 30%)
  --model MODEL         Model name from HuggingFace (default: meta-llama/...)
  --device DEVICE       cuda or cpu (default: cuda)
  --batch-size SIZE     Batch size for extraction (default: 4)
  --max-length LENGTH   Max sequence length (default: 512)
  --steps STEPS         Steps to run: 1,2,3,4 (default: 1,2,3)
  --regularization REG  L2 regularization (default: 1.0)
  --prefill-mode MODE   soft or hard (default: soft)
  --thresholds VALUES   Thresholds to sweep (default: 0.5)
  --seed SEED          Random seed (default: 42)
  --split-by-type      Split by dataset type
```

## 🔍 Understanding Results

### Probe Training Metrics (probe/results.json)

Good performance indicators:
- ✅ `train_auc` > 0.85 - Probe learned the pattern
- ✅ `val_auc` > 0.80 - Generalizes to held-out data
- ✅ `val_accuracy` > 0.80 - High accuracy
- ⚠️ If `val_auc` << `train_auc` - Overfitting, reduce regularization

### What the Metrics Mean

- **Accuracy**: Percentage of correct tool/no-tool decisions
- **AUC**: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
  - 0.89-0.96: Excellent (from paper)
  - 0.80-0.89: Good
  - 0.70-0.80: Fair
  - < 0.70: Poor

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 2  # or 1
```

### Model too slow
```bash
# Use smaller model
--model Qwen/Qwen3-4B-Instruct  # instead of 70B
```

### Missing labels in dataset
If your dataset doesn't have `tool_necessary` field:
```python
# Add it before running
for item in dataset:
    item["tool_necessary"] = 0  # or 1 as needed
```

## 📚 Examples

See `WHEN2TOOL_EXAMPLES.sh` for more examples:

```bash
bash WHEN2TOOL_EXAMPLES.sh
```

## 📖 Full Documentation

For complete documentation, see:

```bash
cat WHEN2TOOL_README.md
```

## 🎓 Understanding the Method

The When2Tool method works in 3 steps:

1. **Extract**: Get hidden states from model forward pass at last token
2. **Train**: Train linear probe P(tool_necessary) on hidden states
3. **Decide**: At inference time, use probe probability to decide: 
   - If P < τ (threshold) → No tool needed
   - If P ≥ τ → Tool is needed

The genius: The model already "knows" whether tools are needed just from processing the question! The probe reads this knowledge from hidden states.

## 🚀 Next Steps

1. Try the quick start command above
2. Check the output in `when2tool_outputs/`
3. Review the metrics in `probe/results.json`
4. See `WHEN2TOOL_README.md` for advanced usage

## 📞 Support

For issues or questions:
1. Check `WHEN2TOOL_README.md` Troubleshooting section
2. Review example commands in `WHEN2TOOL_EXAMPLES.sh`
3. Check individual script help: `python -m sragents.when2tool.pipeline --help`

---

**Happy training!** 🎉
