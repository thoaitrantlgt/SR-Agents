"""
Evaluate probe on TEST set and compare with no_skill eval results.

Analysis:
  - When no_skill model answered CORRECTLY  → does probe say "load_skill" or not?
  - When no_skill model answered INCORRECTLY → does probe say "load_skill" or not?

Label convention (from training):
  probe=1 → "load_skill" (tool necessary)
  probe=0 → "no_skill"   (tool not necessary)
"""
import sys
sys.path.insert(0, "src")

# Compatibility shim for numpy 1.x vs 2.x pickle compatibility
try:
    import numpy._core
except ImportError:
    try:
        import numpy.core
        sys.modules['numpy._core'] = numpy.core
    except ImportError:
        pass

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from sragents.when2tool.train_linear_probe import LinearProbe, prepare_hidden_states

# ── Paths ─────────────────────────────────────────────────────────────────────
PROBE_PATH      = "when2tool_outputs_final/probe/probe.pkl"
TEST_HS_PATH    = "when2tool_outputs_final/features/test/hidden_states.pt"
TEST_ITEMS_PATH = "when2tool_outputs_final/features/test/items.json"
EVAL_FILE       = "results/eval_noskill_theoremqa.json"
OUT_FILE        = "when2tool_outputs_final/probe_vs_noskill_analysis.json"

# ── Load probe ────────────────────────────────────────────────────────────────
print("Loading probe...")
probe = LinearProbe.load(PROBE_PATH)

# ── Load test hidden states ───────────────────────────────────────────────────
print("Loading test hidden states...")
test_hs = torch.load(TEST_HS_PATH).numpy()   # (523, 37, 2560)
print(f"  shape: {test_hs.shape}")

# ── Load test items (to get instance_ids) ─────────────────────────────────────
print("Loading test items...")
with open(TEST_ITEMS_PATH) as f:
    test_items = json.load(f)
    if isinstance(test_items, dict):
        test_items = list(test_items.values())

test_ids = [item.get("instance_id") for item in test_items]
print(f"  {len(test_ids)} test instances")

# ── Prepare features & predict ────────────────────────────────────────────────
print("Predicting with probe...")
X_test = prepare_hidden_states(test_hs, concatenate_layers=False)  # last layer only
pred_labels, pred_proba = probe.predict(X_test)
# pred_labels: 1 = load_skill, 0 = no_skill

# ── Load no_skill eval results ────────────────────────────────────────────────
print(f"Loading no_skill eval from {EVAL_FILE}...")
with open(EVAL_FILE) as f:
    eval_data = json.load(f)

eval_map = {}  # instance_id → {correct, ground_truth, extracted_answer, ...}
details = eval_data.get("details", [])
if isinstance(details, list):
    for d in details:
        eval_map[d["instance_id"]] = d
elif isinstance(details, dict):
    eval_map = details

print(f"  eval_map covers {len(eval_map)} instances")

# ── Cross-analysis ────────────────────────────────────────────────────────────
results = []
confusion = {
    # (noskill_correct, probe_load_skill)
    (True,  True):  [],  # model was correct BUT probe says load_skill   → False Alarm
    (True,  False): [],  # model was correct, probe says no_skill         → True Negative ✅
    (False, True):  [],  # model was wrong,   probe says load_skill       → True Positive ✅
    (False, False): [],  # model was wrong,   probe says no_skill         → Missed (False Neg) ❌
}

not_in_eval = 0
for i, (iid, pred, prob) in enumerate(zip(test_ids, pred_labels, pred_proba)):
    load_skill = bool(pred == 1)
    
    if iid not in eval_map:
        not_in_eval += 1
        entry = {
            "instance_id": iid,
            "probe_decision": "load_skill" if load_skill else "no_skill",
            "probe_probability": round(float(prob), 4),
            "noskill_correct": None,
            "noskill_answer": None,
            "ground_truth": None,
        }
    else:
        ev = eval_map[iid]
        correct = ev.get("correct", None)
        entry = {
            "instance_id": iid,
            "probe_decision": "load_skill" if load_skill else "no_skill",
            "probe_probability": round(float(prob), 4),
            "noskill_correct": correct,
            "noskill_answer": ev.get("extracted_answer", ""),
            "ground_truth": ev.get("ground_truth", ""),
        }
        if correct is not None:
            confusion[(correct, load_skill)].append(iid)
    
    results.append(entry)

# ── Print summary ──────────────────────────────────────────────────────────────
total_in_eval = len(test_ids) - not_in_eval
TP = len(confusion[(False, True)])   # model wrong  + probe says load_skill
TN = len(confusion[(True,  False)])  # model correct + probe says no_skill
FP = len(confusion[(True,  True)])   # model correct + probe says load_skill (false alarm)
FN = len(confusion[(False, False)])  # model wrong  + probe says no_skill (missed)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
accuracy  = (TP + TN) / total_in_eval if total_in_eval > 0 else 0

print("\n" + "="*60)
print("PROBE vs NO_SKILL ANALYSIS (Test Set)")
print("="*60)
print(f"\nTest instances total   : {len(test_ids)}")
print(f"Found in no_skill eval : {total_in_eval}")
print(f"Not in eval (skipped)  : {not_in_eval}")

print(f"\n{'':30s} {'Probe: load_skill':>18} {'Probe: no_skill':>15}")
print(f"{'─'*65}")
print(f"{'NoSkill CORRECT  (tool not needed)':30s} {FP:>15} (FP)  {TN:>10} (TN)")
print(f"{'NoSkill WRONG    (tool needed)':30s} {TP:>15} (TP)  {FN:>10} (FN)")

print(f"\n── Metrics ──────────────────────────────")
print(f"  Accuracy  : {accuracy:.4f}  ({TP+TN}/{total_in_eval})")
print(f"  Precision : {precision:.4f}  (when probe says load_skill, how often was model wrong?)")
print(f"  Recall    : {recall:.4f}  (of all model-wrong cases, how many did probe catch?)")
print(f"  F1        : {f1:.4f}")

noskill_correct_total = TP + TN + FP + FN
noskill_correct_count = TN + FP
noskill_wrong_count   = TP + FN
probe_load_skill_when_correct = FP / (TN + FP) if (TN + FP) > 0 else 0
probe_load_skill_when_wrong   = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"\n── Key Questions ────────────────────────")
print(f"  When model answered CORRECTLY ({noskill_correct_count} cases):")
print(f"    → Probe says 'load_skill': {FP} ({probe_load_skill_when_correct:.1%}) ← false alarms")
print(f"    → Probe says 'no_skill'  : {TN} ({1-probe_load_skill_when_correct:.1%}) ← correct decisions ✅")
print(f"\n  When model answered WRONGLY ({noskill_wrong_count} cases):")
print(f"    → Probe says 'load_skill': {TP} ({probe_load_skill_when_wrong:.1%}) ← correctly identified need ✅")
print(f"    → Probe says 'no_skill'  : {FN} ({1-probe_load_skill_when_wrong:.1%}) ← missed, would still fail ❌")

# ── Save full results ──────────────────────────────────────────────────────────
output = {
    "summary": {
        "total_test": len(test_ids),
        "found_in_eval": total_in_eval,
        "confusion_matrix": {
            "TP_wrong_and_load_skill": TP,
            "TN_correct_and_no_skill": TN,
            "FP_correct_but_load_skill": FP,
            "FN_wrong_but_no_skill": FN,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "probe_load_skill_rate_when_noskill_correct": round(probe_load_skill_when_correct, 4),
            "probe_load_skill_rate_when_noskill_wrong": round(probe_load_skill_when_wrong, 4),
        }
    },
    "per_instance": results,
}

with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Full per-instance results saved to: {OUT_FILE}")
