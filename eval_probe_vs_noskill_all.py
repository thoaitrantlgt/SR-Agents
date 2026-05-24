#!/usr/bin/env python3
"""
Evaluate trained probe across multiple datasets and print a summary table.

This script compares the probe's load_skill / no_skill decisions against
no-skill evaluation results for one or more datasets.

Example:
    python eval_probe_vs_noskill_all.py \
        --output-template when2tool_outputs_{dataset} \
        --eval-file-template results/eval_noskill_{dataset}.json

If your outputs are all under one directory without dataset placeholders,
pass --output-template when2tool_outputs_final.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from sragents.when2tool.train_linear_probe import LinearProbe, prepare_hidden_states

DEFAULT_DATASETS = [
    "medcalcbench",
    "theoremqa",
    "logicbench",
    "champ",
    "bigcodebench",
    "toolqa",
]


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_output_dir(template: str, dataset: str) -> Path:
    if "{dataset}" in template:
        return Path(template.format(dataset=dataset))
    return Path(template)


def build_eval_path(template: str, dataset: str) -> Path:
    return Path(template.format(dataset=dataset))


def safe_ratio(num: int, denom: int) -> float:
    return float(num) / denom if denom else 0.0


def format_pct(value: float) -> str:
    return f"{value:.1%}" if value is not None else "-"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate probe decisions against no-skill eval over multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate (default: medcalcbench theoremqa logicbench champ bigcodebench toolqa)",
    )
    parser.add_argument(
        "--output-template",
        default="when2tool_outputs_{dataset}",
        help="Path template for dataset outputs. Use {dataset} macro, e.g. when2tool_outputs_{dataset}.",
    )
    parser.add_argument(
        "--eval-file-template",
        default="results/eval_noskill_{dataset}.json",
        help="Evaluation JSON template with {dataset} macro.",
    )
    parser.add_argument(
        "--save-summary",
        default="eval_probe_vs_noskill_all_summary.json",
        help="Path to save the summary JSON output.",
    )
    parser.add_argument(
        "--concatenate-layers",
        action="store_true",
        help="Concatenate all layers when preparing features (default: last layer only).",
    )
    parser.add_argument(
        "--probe-subpath",
        default="probe/probe.pkl",
        help="Relative path to trained probe inside each output directory.",
    )
    parser.add_argument(
        "--hidden-states-subpath",
        default="features/test/hidden_states.pt",
        help="Relative path to test hidden states inside each output directory.",
    )
    parser.add_argument(
        "--items-subpath",
        default="features/test/items.json",
        help="Relative path to test items inside each output directory.",
    )

    args = parser.parse_args()

    rows = []
    all_datasets = []

    for ds in args.datasets:
        output_dir = build_output_dir(args.output_template, ds)
        probe_path = output_dir / args.probe_subpath
        hs_path = output_dir / args.hidden_states_subpath
        items_path = output_dir / args.items_subpath
        eval_path = build_eval_path(args.eval_file_template, ds)

        row = {
            "dataset": ds,
            "output_dir": str(output_dir),
            "probe_path": str(probe_path),
            "hidden_states_path": str(hs_path),
            "items_path": str(items_path),
            "eval_path": str(eval_path),
            "status": "ok",
        }

        if not output_dir.exists():
            row["status"] = "missing_output_dir"
            rows.append(row)
            print(f"WARN: output directory not found for {ds}: {output_dir}")
            continue

        if not probe_path.exists():
            row["status"] = "missing_probe"
            rows.append(row)
            print(f"WARN: probe not found for {ds}: {probe_path}")
            continue

        if not hs_path.exists():
            row["status"] = "missing_hidden_states"
            rows.append(row)
            print(f"WARN: hidden states not found for {ds}: {hs_path}")
            continue

        if not items_path.exists():
            row["status"] = "missing_items"
            rows.append(row)
            print(f"WARN: test items not found for {ds}: {items_path}")
            continue

        if not eval_path.exists():
            row["status"] = "missing_eval"
            rows.append(row)
            print(f"WARN: eval file not found for {ds}: {eval_path}")
            continue

        try:
            probe = LinearProbe.load(probe_path)
            test_hs = torch.load(hs_path).numpy()
            items = load_json(items_path)
            if isinstance(items, dict):
                items = list(items.values())

            test_ids = [item.get("instance_id") for item in items]
            if any(iid is None for iid in test_ids):
                print(f"WARN: missing instance_id in items for {ds}")

            X_test = prepare_hidden_states(test_hs, concatenate_layers=args.concatenate_layers)
            pred_labels, pred_proba = probe.predict(X_test)

            eval_data = load_json(eval_path)
            details = eval_data.get("details", [])
            if isinstance(details, dict):
                eval_map = details
            else:
                eval_map = {d["instance_id"]: d for d in details if "instance_id" in d}

            total = len(test_ids)
            found = 0
            TP = TN = FP = FN = 0
            total_probs = 0.0
            for iid, pred, proba in zip(test_ids, pred_labels, pred_proba):
                correct = None
                if iid in eval_map:
                    found += 1
                    correct = eval_map[iid].get("correct")

                load_skill = bool(pred == 1)
                if correct is True:
                    if load_skill:
                        FP += 1
                    else:
                        TN += 1
                elif correct is False:
                    if load_skill:
                        TP += 1
                    else:
                        FN += 1

                total_probs += float(proba)

            coverage = safe_ratio(found, total)
            accuracy = safe_ratio(TP + TN, found)
            precision = safe_ratio(TP, TP + FP)
            recall = safe_ratio(TP, TP + FN)
            f1 = safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
            avg_prob = total_probs / total if total else 0.0

            row.update({
                "status": "ok",
                "total_test": total,
                "eval_coverage": round(coverage, 4),
                "found_in_eval": found,
                "missing_in_eval": total - found,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "probe_load_skill_rate": round(safe_ratio(TP + FP, total), 4),
                "avg_probe_probability": round(avg_prob, 4),
            })

            rows.append(row)
            all_datasets.append(row)

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            rows.append(row)
            print(f"ERROR processing {ds}: {exc}")

    if not all_datasets:
        print("No dataset evaluations completed successfully.")
        return 1

    print("\nSummary table:\n")
    header = [
        "Dataset",
        "Total",
        "Covered",
        "Acc",
        "Prec",
        "Rec",
        "F1",
        "TP",
        "TN",
        "FP",
        "FN",
        "Load%",
    ]
    print("  " + " | ".join(f"{h:>8s}" for h in header))
    print("  " + " | ".join(["--------" for _ in header]))
    for row in all_datasets:
        print(
            f"  {row['dataset']:<10s} | "
            f"{row['total_test']:>5d} | "
            f"{format_pct(row['eval_coverage']):>6s} | "
            f"{row['accuracy']:.3f} | "
            f"{row['precision']:.3f} | "
            f"{row['recall']:.3f} | "
            f"{row['f1']:.3f} | "
            f"{row['TP']:>3d} | "
            f"{row['TN']:>3d} | "
            f"{row['FP']:>3d} | "
            f"{row['FN']:>3d} | "
            f"{format_pct(row['probe_load_skill_rate']):>6s}"
        )

    total_test = sum(r["total_test"] for r in all_datasets)
    total_found = sum(r["found_in_eval"] for r in all_datasets)
    total_TP = sum(r["TP"] for r in all_datasets)
    total_TN = sum(r["TN"] for r in all_datasets)
    total_FP = sum(r["FP"] for r in all_datasets)
    total_FN = sum(r["FN"] for r in all_datasets)
    total_accuracy = safe_ratio(total_TP + total_TN, total_found)
    total_precision = safe_ratio(total_TP, total_TP + total_FP)
    total_recall = safe_ratio(total_TP, total_TP + total_FN)
    total_f1 = safe_ratio(2 * total_precision * total_recall, total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0

    print("  " + "-" * 116)
    print(
        f"  {'TOTAL':<10s} | {total_test:>5d} | {format_pct(safe_ratio(total_found,total_test)):>6s} | "
        f"{total_accuracy:.3f} | {total_precision:.3f} | {total_recall:.3f} | {total_f1:.3f} | "
        f"{total_TP:>3d} | {total_TN:>3d} | {total_FP:>3d} | {total_FN:>3d} | {'-':>6s}"
    )

    with open(args.save_summary, "w") as out_f:
        json.dump({"datasets": rows}, out_f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary JSON to {args.save_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
