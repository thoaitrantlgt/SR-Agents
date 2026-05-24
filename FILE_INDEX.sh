#!/bin/bash

# When2Tool Pipeline - File Index and Navigation Guide

cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║                  WHEN2TOOL PIPELINE - FILE INDEX              ║
║              Complete Implementation for SR-Agents             ║
╚════════════════════════════════════════════════════════════════╝

📂 PROJECT STRUCTURE
════════════════════════════════════════════════════════════════

1. DOCUMENTATION (Start Here)
   ─────────────────────────────────────────────────────────────
   
   📄 README_WHEN2TOOL.md (This file's companion)
      → Overview of entire implementation
      → Quick reference table
      → Getting started in 5 steps
      
   📄 START_HERE.md  
      → Main entry point for users
      → 30-second quick start
      → Step-by-step explanation of pipeline
      → Configuration examples
      
   📄 WHEN2TOOL_QUICKSTART.md
      → Copy-paste ready examples
      → All command options
      → Interpreting results
      → Troubleshooting
      
   📄 WHEN2TOOL_README.md
      → Complete API documentation
      → Detailed configuration reference
      → Performance notes
      → Advanced usage
      
   📄 WHEN2TOOL_IMPLEMENTATION_SUMMARY.md
      → Implementation details
      → Architecture overview
      → Technical decisions
      → Dependencies

2. SCRIPTS (Executable)
   ─────────────────────────────────────────────────────────────
   
   🔧 run_when2tool_pipeline.sh
      → Bash wrapper for the pipeline
      → Parses command-line arguments
      → Usage: bash run_when2tool_pipeline.sh <data.json>
      
   🔧 validate_when2tool.sh
      → Validates installation
      → Checks all imports
      → Verifies file structure
      → Usage: bash validate_when2tool.sh
      
   🔧 WHEN2TOOL_EXAMPLES.sh
      → Example command patterns
      → Different configuration scenarios
      → Use cases (quick test, large model, etc.)
      → Usage: bash WHEN2TOOL_EXAMPLES.sh | less

3. CORE IMPLEMENTATION
   ─────────────────────────────────────────────────────────────
   
   Location: src/sragents/when2tool/
   
   📦 __init__.py
      → Package initialization
      
   📦 __main__.py
      → CLI entry point for "python -m sragents.when2tool"
      
   Step 1: DATASET SPLITTING
   ─────────────────────────
   📦 split_dataset.py
      • Splits dataset into 30% train / 70% test
      • Supports JSON and JSONL formats
      • Optional: Split by dataset type
      • CLI: python -m sragents.when2tool.split_dataset --help
      
   Step 2: HIDDEN STATE EXTRACTION
   ───────────────────────────────
   📦 extract_hidden_states.py
      • Extracts hidden states at last token position
      • From all layers: (n_samples, n_layers, hidden_dim)
      • Batch processing with progress tracking
      • CLI: python -m sragents.when2tool.extract_hidden_states --help
      
   Step 3: LINEAR PROBE TRAINING
   ─────────────────────────────
   📦 train_linear_probe.py
      • Trains logistic regression on hidden states
      • L2 regularization support
      • Validation split for metrics
      • CLI: python -m sragents.when2tool.train_linear_probe --help
      
   Optional Step 4: PREFILL INFERENCE
   ──────────────────────────────────
   📦 prefill_inference.py
      • Probe-guided generation with prefilling
      • Soft (natural) and hard (forced) modes
      • Configurable threshold for tool decisions
      • CLI: python -m sragents.when2tool.prefill_inference --help
      
   MAIN ORCHESTRATOR
   ─────────────────
   📦 pipeline.py
      • When2ToolPipeline class
      • PipelineConfig dataclass
      • Orchestrates all steps
      • Main CLI entry point
      • CLI: python -m sragents.when2tool.pipeline --help

════════════════════════════════════════════════════════════════
📖 READING GUIDE
════════════════════════════════════════════════════════════════

For Different Audiences:

👤 FIRST TIME USERS
   1. Read: README_WHEN2TOOL.md (5 min)
   2. Read: START_HERE.md (10 min)
   3. Run: bash validate_when2tool.sh (1 min)
   4. Try: python -m sragents.when2tool.pipeline --help (1 min)
   5. Run: Copy example from WHEN2TOOL_QUICKSTART.md (varies)

👨‍💻 DEVELOPERS
   1. Read: WHEN2TOOL_IMPLEMENTATION_SUMMARY.md
   2. Read: WHEN2TOOL_README.md (API reference section)
   3. Review: src/sragents/when2tool/*.py (code)
   4. Check: Test with validate_when2tool.sh

🔧 DEVOPS / DEPLOYMENT
   1. Read: WHEN2TOOL_README.md (Performance notes)
   2. Review: run_when2tool_pipeline.sh (wrapper)
   3. Setup: Configure batch size and memory limits
   4. Monitor: Check outputs in when2tool_outputs/

════════════════════════════════════════════════════════════════
⚡ QUICK START COMMANDS
════════════════════════════════════════════════════════════════

# 1. Validate setup
bash validate_when2tool.sh

# 2. Run full pipeline
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json

# 3. Run only steps 1-2 (quick test)
python -m sragents.when2tool.pipeline \
    data/bench/instances/bigcodebench.json \
    --steps 1,2 \
    --batch-size 1

# 4. Check results
cat when2tool_outputs/probe/results.json

# 5. See all examples
bash WHEN2TOOL_EXAMPLES.sh

════════════════════════════════════════════════════════════════
📋 FILE DESCRIPTIONS
════════════════════════════════════════════════════════════════

Documentation Files (In order of detail level):
─────────────────────────────────────────────────────────────────
README_WHEN2TOOL.md                [Beginner] Executive summary
START_HERE.md                       [Beginner] Detailed walkthrough
WHEN2TOOL_QUICKSTART.md            [User]     Copy-paste examples
WHEN2TOOL_README.md                [Dev]      Complete reference
WHEN2TOOL_EXAMPLES.sh              [User]     Command templates
WHEN2TOOL_IMPLEMENTATION_SUMMARY.md [Dev]     Technical deep-dive

Python Implementation (In pipeline order):
─────────────────────────────────────────────────────────────────
split_dataset.py                   [Step 1]   Dataset splitting
extract_hidden_states.py           [Step 2]   Feature extraction
train_linear_probe.py              [Step 3]   Model training
prefill_inference.py               [Step 4]   Optional inference
pipeline.py                        [Main]     Orchestrator

════════════════════════════════════════════════════════════════
🎯 COMMON WORKFLOWS
════════════════════════════════════════════════════════════════

WORKFLOW 1: First Time Running
────────────────────────────────
1. bash validate_when2tool.sh
2. python -m sragents.when2tool.pipeline data.json

WORKFLOW 2: Quick Testing
────────────────────────────
1. python -m sragents.when2tool.pipeline data.json --steps 1,2
2. ls when2tool_outputs/features/

WORKFLOW 3: Custom Configuration
──────────────────────────────────
1. Read WHEN2TOOL_QUICKSTART.md
2. Copy an example and modify
3. python -m sragents.when2tool.pipeline ...

WORKFLOW 4: Using Python API
──────────────────────────────
1. Read WHEN2TOOL_README.md (API section)
2. Create config.py with PipelineConfig
3. python config.py

WORKFLOW 5: Troubleshooting
─────────────────────────────
1. Check WHEN2TOOL_QUICKSTART.md (Troubleshooting)
2. Run with reduced batch size
3. Check validate_when2tool.sh output

════════════════════════════════════════════════════════════════
📊 PIPELINE OVERVIEW
════════════════════════════════════════════════════════════════

Input Dataset (JSON/JSONL)
         ↓
[Step 1] Split 30/70 using split_dataset.py
         ├─ 30% Training Set
         └─ 70% Test Set
         ↓
[Step 2] Extract Hidden States using extract_hidden_states.py
         ├─ Forward pass on model
         ├─ Last token position
         └─ All layers → (n, layers, dim)
         ↓
[Step 3] Train Linear Probe using train_linear_probe.py
         ├─ Concatenate features
         ├─ Logistic regression
         └─ Output: binary classifier
         ↓
[Step 4] Prefill Inference using prefill_inference.py (Optional)
         ├─ Predict tool necessity
         ├─ Apply threshold
         └─ Generate with prefilling
         ↓
Output: when2tool_outputs/
         ├─ splits/
         ├─ features/
         ├─ probe/
         └─ inference/ (optional)

════════════════════════════════════════════════════════════════
✅ VERIFICATION CHECKLIST
════════════════════════════════════════════════════════════════

Before running:
☐ Read START_HERE.md
☐ Run bash validate_when2tool.sh
☐ Check dataset exists
☐ Have GPU or set --device cpu

After running:
☐ Check when2tool_outputs/ created
☐ Review probe/results.json
☐ Check AUROC > 0.80
☐ Inspect splits/ size (30% and 70%)

════════════════════════════════════════════════════════════════
🔗 KEY RESOURCES
════════════════════════════════════════════════════════════════

Paper:      https://arxiv.org/abs/2605.09252
Repo:       https://github.com/Trustworthy-ML-Lab/when2tool
Dataset:    https://huggingface.co/datasets/cesun/When2Tool

════════════════════════════════════════════════════════════════
💡 TIPS & TRICKS
════════════════════════════════════════════════════════════════

💡 TIP 1: Run validation first
   bash validate_when2tool.sh
   
💡 TIP 2: Start with small batch size
   --batch-size 1 (if out of memory)
   
💡 TIP 3: Use --steps 1,2 for quick test
   (Skips probe training which takes longer)
   
💡 TIP 4: Save configurations
   Look at when2tool_outputs/config.json after run
   
💡 TIP 5: Check intermediate outputs
   ls -la when2tool_outputs/features/train/

════════════════════════════════════════════════════════════════
📞 GETTING HELP
════════════════════════════════════════════════════════════════

For general usage:
   → Read START_HERE.md

For specific commands:
   → python -m sragents.when2tool.pipeline --help
   → python -m sragents.when2tool.split_dataset --help
   → python -m sragents.when2tool.extract_hidden_states --help
   → python -m sragents.when2tool.train_linear_probe --help

For troubleshooting:
   → See WHEN2TOOL_QUICKSTART.md section "Troubleshooting"
   → Run: bash validate_when2tool.sh

For technical details:
   → Read WHEN2TOOL_IMPLEMENTATION_SUMMARY.md

════════════════════════════════════════════════════════════════
🎉 YOU'RE READY!
════════════════════════════════════════════════════════════════

Next Step: Read START_HERE.md

   cat START_HERE.md

Then run your first pipeline:

   python -m sragents.when2tool.pipeline \\
       data/bench/instances/bigcodebench.json

════════════════════════════════════════════════════════════════

EOF
