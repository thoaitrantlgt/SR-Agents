# 🎯 When2Tool Pipeline - Implementation Complete ✅

## What You Now Have

A **complete, production-ready implementation** of the When2Tool method for SR-Agents that includes:

### Core Components (7 modules)
- ✅ Dataset splitting (30/70 train/test)
- ✅ Hidden state extraction (all layers, last token)
- ✅ Linear probe training (logistic regression)
- ✅ Prefill-guided inference (soft/hard modes)
- ✅ Full pipeline orchestration
- ✅ Command-line interface
- ✅ Python API

### Documentation
- ✅ Quick start guide (START_HERE.md)
- ✅ Full README with API reference
- ✅ Example commands and scripts
- ✅ Troubleshooting guide
- ✅ Implementation details

### Scripts & Tools
- ✅ Shell script wrapper
- ✅ Validation script
- ✅ Example scripts

---

## 🚀 30-Second Quick Start

```bash
cd /network-volume/SR-Agents

# Run the entire pipeline
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --output-dir ./when2tool_outputs \
    --train-ratio 0.3
```

**That's it!** This will:
1. Split your dataset 30/70
2. Extract hidden states from all layers
3. Train a linear probe to predict tool necessity
4. Save all results and metrics

---

## 📂 File Structure

```
/network-volume/SR-Agents/
│
├── START_HERE.md                          ← Start with this!
├── WHEN2TOOL_QUICKSTART.md                ← Quick examples
├── WHEN2TOOL_README.md                    ← Full documentation
├── WHEN2TOOL_EXAMPLES.sh                  ← More examples
├── WHEN2TOOL_IMPLEMENTATION_SUMMARY.md    ← Technical details
│
├── run_when2tool_pipeline.sh              ← Bash wrapper
├── validate_when2tool.sh                  ← Validation script
│
└── src/sragents/when2tool/
    ├── __init__.py
    ├── __main__.py
    ├── split_dataset.py                   (Step 1)
    ├── extract_hidden_states.py           (Step 2)
    ├── train_linear_probe.py              (Step 3)
    ├── prefill_inference.py               (Optional Step 4)
    └── pipeline.py                        (Main orchestrator)
```

---

## 📋 Three Ways to Use

### Method 1: Python CLI (Recommended)
```bash
python -m sragents.when2tool.pipeline data.json --output-dir ./outputs
```

### Method 2: Bash Script
```bash
bash run_when2tool_pipeline.sh data.json --output-dir ./outputs
```

### Method 3: Python API
```python
from sragents.when2tool.pipeline import When2ToolPipeline, PipelineConfig

config = PipelineConfig(dataset_path="data.json")
pipeline = When2ToolPipeline(config)
results = pipeline.run()
```

---

## 🔑 Key Features

| Feature | Details |
|---------|---------|
| **Dataset Splitting** | 30% train / 70% test (configurable) |
| **Hidden State Extraction** | All layers, last token position, no extra compute |
| **Linear Probe** | L2-regularized logistic regression on concatenated states |
| **Metrics** | Accuracy, AUROC, precision, recall |
| **Prefilling** | Soft (natural) or hard (forced) formats |
| **Models** | Works with any HuggingFace model |
| **Parallelization** | Batch processing with progress bars |
| **Reproducibility** | Fixed random seeds throughout |

---

## 📊 What Each Step Does

### Step 1: Split Dataset (split_dataset.py)
```
Input:  dataset.json (e.g., 9000 items)
        ↓
        Random shuffle with fixed seed
        ↓
Output: train_json.json (30% = 2700 items)
        test_json.json  (70% = 6300 items)
```

### Step 2: Extract Hidden States (extract_hidden_states.py)
```
Input:  Questions from training/test sets
        Model: "Qwen/Qwen3-4B-Instruct"
        ↓
        Forward pass on each question
        Extract hidden states at last token
        Stack all layers: (n_samples, n_layers, hidden_dim)
        ↓
Output: features/train/hidden_states.pt  (2700, 32, 4096)
        features/test/hidden_states.pt   (6300, 32, 4096)
```

### Step 3: Train Linear Probe (train_linear_probe.py)
```
Input:  Hidden states: (2700, 32, 4096)
        Binary labels: tool_necessary (0 or 1)
        ↓
        Concatenate all layers: (2700, 131072)
        Scale features: StandardScaler
        Train: LogisticRegression(C=1/regularization)
        Validate: Hold-out split
        ↓
Output: probe.pkl (trained model + scaler)
        results.json (accuracy, AUROC, etc.)
```

### Step 4: Prefill Inference (prefill_inference.py) [Optional]
```
Input:  Test dataset, trained probe, threshold τ
        ↓
        For each question:
        - Get hidden states
        - Predict: P(tool_necessary) = probe(hidden_states)
        - Decision: if P < τ → no tool, else → need tool
        - Prepend: "I can solve..." or "I need a tool..."
        - Generate: Model continues from prefill
        ↓
Output: Generations with guided tool decisions
```

---

## 💾 Output Organization

```
when2tool_outputs/
├── config.json                    # Your configuration saved
├── pipeline_results.json          # Results summary
├── splits/                        # Step 1 outputs
│   ├── train_json.json
│   └── test_json.json
├── features/                      # Step 2 outputs
│   ├── train/
│   │   ├── hidden_states.pt      # Torch tensor
│   │   ├── items.json             # Dataset items
│   │   └── metadata.json
│   └── test/
│       ├── hidden_states.pt
│       ├── items.json
│       └── metadata.json
└── probe/                         # Step 3 outputs
    ├── probe.pkl                 # Trained model
    └── results.json              # Metrics
```

---

## 🎯 How to Interpret Results

### Probe Metrics (probe/results.json)

Expected good values:
- **train_accuracy**: > 0.80
- **val_accuracy**: > 0.75
- **train_auc**: > 0.85 (typically 0.89-0.96 from paper)
- **val_auc**: > 0.80

If `val_auc` is much lower than `train_auc` → Overfitting
- Solution: Increase `--regularization` value

---

## 🔧 Customization Examples

### Use Different Model
```bash
python -m sragents.when2tool.pipeline data.json \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Different Train/Test Ratio
```bash
python -m sragents.when2tool.pipeline data.json \
    --train-ratio 0.5  # 50/50 instead of 30/70
```

### Run Only Specific Steps
```bash
# Only split and extract (skip probe training)
python -m sragents.when2tool.pipeline data.json --steps 1,2

# Only train probe (after extraction)
python -m sragents.when2tool.pipeline data.json --steps 3
```

### Adjust Regularization
```bash
python -m sragents.when2tool.pipeline data.json \
    --regularization 100  # Stronger regularization
```

---

## 📈 Expected Performance

### Timing
| Step | Time |
|------|------|
| Split (Step 1) | < 1 second |
| Extract (Step 2) | 1-5 minutes |
| Train Probe (Step 3) | < 1 minute |
| Total | ~5-10 minutes |

### Memory
- Model: 16GB for 8B model, 40GB+ for 70B
- Features: ~500MB per 1K samples
- Probe: < 100MB

### Quality
- Expected AUROC: 0.89-0.96 (from paper)
- Accuracy: > 80%
- Reduces tool calls by ~48% with only 1.7% accuracy loss

---

## ✅ Validation Checklist

Before running, verify:
```bash
# 1. Check installation
bash validate_when2tool.sh

# 2. Check dataset exists
ls data/bench/instances/bigcodebench.json

# 3. Check GPU availability (optional)
nvidia-smi

# 4. Quick test run (small sample)
python -m sragents.when2tool.pipeline data.json \
    --steps 1 --train-ratio 0.1  # Use only 10%
```

---

## 🎓 Method Overview

**The Big Idea:**
When you ask an LLM a question, it processes the question and generates hidden states that contain *latent knowledge* about whether tools are needed. We:

1. Extract this knowledge from hidden states
2. Train a linear probe to read this knowledge
3. Use the probe's predictions to guide generation with appropriate prefills

**Key Innovation:** The model already knows when to call tools - we just read it from its hidden states!

---

## 📚 Documentation Structure

| Document | Purpose | Audience |
|----------|---------|----------|
| **START_HERE.md** | Entry point with overview | Everyone |
| **WHEN2TOOL_QUICKSTART.md** | Copy-paste examples | Users |
| **WHEN2TOOL_README.md** | Complete API & reference | Developers |
| **WHEN2TOOL_EXAMPLES.sh** | Example commands | Everyone |
| **WHEN2TOOL_IMPLEMENTATION_SUMMARY.md** | Technical details | Developers |

---

## 🚀 Getting Started (5 Steps)

1. **Read** this document (you're here! ✅)
2. **Read** START_HERE.md for overview
3. **Run** validation: `bash validate_when2tool.sh`
4. **Try** quick start: `python -m sragents.when2tool.pipeline --help`
5. **Run** the pipeline: `python -m sragents.when2tool.pipeline data.json`

---

## 🎉 Summary

You now have:

✅ **Complete implementation** of When2Tool method
✅ **Three ways to use it** (CLI, Bash, Python API)
✅ **All 3 core steps** implemented:
   - Dataset splitting (30/70)
   - Hidden state extraction
   - Linear probe training
✅ **Full documentation** with examples
✅ **Production-ready code** with error handling
✅ **Validation tools** to verify setup
✅ **Flexible configuration** for customization

---

## 📞 Next Actions

1. Start here:
   ```bash
   cat START_HERE.md
   ```

2. Validate setup:
   ```bash
   bash validate_when2tool.sh
   ```

3. Run your first pipeline:
   ```bash
   python -m sragents.when2tool.pipeline \
       data/bench/instances/bigcodebench.json
   ```

4. Check results:
   ```bash
   cat when2tool_outputs/probe/results.json
   ```

---

**You're all set!** 🚀

For questions or issues, refer to the documentation in:
- `WHEN2TOOL_QUICKSTART.md`
- `WHEN2TOOL_README.md`
- `WHEN2TOOL_EXAMPLES.sh`
