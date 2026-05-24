"""
Quick hyperparameter sweep for linear probe.
Tests combinations of regularization + PCA components using cross-validation.
"""
import sys
sys.path.insert(0, "src")

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from sragents.when2tool.train_linear_probe import prepare_hidden_states, load_labels_from_eval_file

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_HS   = "when2tool_outputs_final/features/train/hidden_states.pt"
TRAIN_ITEMS = "when2tool_outputs_final/features/train/items.json"
EVAL_FILE  = "results/eval_noskill_theoremqa.json"

# ── Grid ─────────────────────────────────────────────────────────────────────
REG_VALUES  = [10.0, 100.0, 1000.0]    # regularization (1/C)
PCA_VALUES  = [16, 32, 64, 128]             # always use PCA (no-PCA is too slow for 2560 dims)
N_FOLDS     = 5

print("Loading data...")
hs = torch.load(TRAIN_HS).numpy()           # (224, 37, 2560)
with open(TRAIN_ITEMS) as f:
    items = json.load(f)
    if isinstance(items, dict):
        items = list(items.values())

y = load_labels_from_eval_file(items, EVAL_FILE)
X = hs[:, -1, :]   # last layer only → (224, 2560)

print(f"X shape: {X.shape}, labels: {np.bincount(y)}")
print(f"\nRunning {N_FOLDS}-fold CV over {len(REG_VALUES)*len(PCA_VALUES)} combos...\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = []
header = f"{'PCA':>6} {'Reg':>8} │ {'AUC mean':>10} {'AUC std':>9} │ {'Acc mean':>10} {'Acc std':>9}"
print(header)
print("─" * len(header))

for pca_k in PCA_VALUES:
    for reg in REG_VALUES:
        aucs, accs = [], []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr  = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

            if pca_k is not None:
                k = min(pca_k, X_tr.shape[0], X_tr.shape[1])
                pca = PCA(n_components=k)
                X_tr  = pca.fit_transform(X_tr)
                X_val = pca.transform(X_val)

            clf = LogisticRegression(C=1.0/reg, max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)

            proba = clf.predict_proba(X_val)[:, 1]
            pred  = clf.predict(X_val)
            aucs.append(roc_auc_score(y_val, proba))
            accs.append(accuracy_score(y_val, pred))

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        pca_str  = str(pca_k) if pca_k else "None"
        print(f"{pca_str:>6} {reg:>8.1f} │ {mean_auc:>10.4f} {std_auc:>9.4f} │ {mean_acc:>10.4f} {std_acc:>9.4f}")
        results.append({
            "pca": pca_k, "reg": reg,
            "val_auc_mean": mean_auc, "val_auc_std": std_auc,
            "val_acc_mean": mean_acc, "val_acc_std": std_acc,
        })

# Best by AUC
best = max(results, key=lambda r: r["val_auc_mean"])
print(f"\n{'='*60}")
print(f"Best combo  →  PCA={best['pca']}  reg={best['reg']}")
print(f"  val_auc = {best['val_auc_mean']:.4f} ± {best['val_auc_std']:.4f}")
print(f"  val_acc = {best['val_acc_mean']:.4f} ± {best['val_acc_std']:.4f}")
print(f"\nRun pipeline with:")
pca_arg = f"--pca-components {best['pca']}" if best['pca'] else "--pca-components 0"
print(f"  --regularization {best['reg']} {pca_arg}")
