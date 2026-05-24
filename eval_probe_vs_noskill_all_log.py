#!/usr/bin/env python3
"""
Produce per-dataset logs showing whether the probe would choose "load_skill" for
cases where the no-skill model is wrong versus cases where the no-skill model is correct.

For each dataset, the script prints:
  - no-skill wrong cases: how many the probe says load_skill / no_skill
  - no-skill correct cases: how many the probe says load_skill / no_skill

Example:
  python3 eval_probe_vs_noskill_all_log.py \
      --output-template when2tool_outputs_{dataset} \
      --eval-file-template results/eval_noskill_{dataset}.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

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
    return Path(template.format(dataset=dataset) if "{dataset}" in template else template)


def build_eval_path(template: str, dataset: str) -> Path:
    return Path(template.format(dataset=dataset))


def log_dataset(dataset: str, output_dir: Path, eval_path: Path, probe_subpath: str, hs_subpath: str, items_subpath: str, concatenate_layers: bool):
    probe_path = output_dir / probe_subpath
    hs_path = output_dir / hs_subpath
    items_path = output_dir / items_subpath

    print(f"\n=== Dataset: {dataset} ===")
    print(f"Output dir: {output_dir}")
    print(f"Probe path: {probe_path}")
    print(f"Hidden states: {hs_path}")
    print(f"Test items: {items_path}")
    print(f"No-skill eval file: {eval_path}\n")

    if not output_dir.exists():
        print(f"SKIP: output directory not found: {output_dir}")
        return
    if not probe_path.exists():
        print(f"SKIP: probe file not found: {probe_path}")
        return
    if not hs_path.exists():
        print(f"SKIP: hidden states file not found: {hs_path}")
        return
    if not items_path.exists():
        print(f"SKIP: items file not found: {items_path}")
        return
    if not eval_path.exists():
        print(f"SKIP: eval file not found: {eval_path}")
        return

    probe = LinearProbe.load(probe_path)
    hidden_states = torch.load(hs_path).numpy()
    test_items = load_json(items_path)
    if isinstance(test_items, dict):
        test_items = list(test_items.values())
    test_ids = [item.get("instance_id") for item in test_items]

    X_test = prepare_hidden_states(hidden_states, concatenate_layers=concatenate_layers)
    pred_labels, pred_proba = probe.predict(X_test)

    eval_data = load_json(eval_path)
    details = eval_data.get("details", [])
    if isinstance(details, dict):
        eval_map = details
    else:
        eval_map = {d["instance_id"]: d for d in details if "instance_id" in d}

    wrong_load = wrong_skip = correct_load = correct_skip = 0
    wrong_total = correct_total = 0
    missing = 0

    for iid, pred in zip(test_ids, pred_labels):
        if iid not in eval_map:
            missing += 1
            continue
        correct = eval_map[iid].get("correct")
        load_skill = pred == 1
        if correct is True:
            correct_total += 1
            if load_skill:
                correct_load += 1
            else:
                correct_skip += 1
        elif correct is False:
            wrong_total += 1
            if load_skill:
                wrong_load += 1
            else:
                wrong_skip += 1

    def pct(value, total):
        return f"{value}/{total} ({value/total:.1%})" if total else "0/0 (0.0%)"

    print("No-skill WRONG cases:")
    print(f"  probe load_skill: {pct(wrong_load, wrong_total)}")
    print(f"  probe no_skill  : {pct(wrong_skip, wrong_total)}")

    print("\nNo-skill CORRECT cases:")
    print(f"  probe load_skill: {pct(correct_load, correct_total)}")
    print(f"  probe no_skill  : {pct(correct_skip, correct_total)}")

    print(f"\nTotal test instances: {len(test_ids)}")
    print(f"Found in eval: {wrong_total + correct_total}")
    print(f"Missing in eval: {missing}")

    if wrong_total or correct_total:
        print("\nSummary:")
        print(f"  When no-skill model is WRONG, probe wants skill in {wrong_load} cases and skips in {wrong_skip} cases.")
        print(f"  When no-skill model is CORRECT, probe wants skill in {correct_load} cases and skips in {correct_skip} cases.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log whether the probe chooses load_skill for no-skill wrong vs correct cases across datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--output-template",
        default="when2tool_outputs_{dataset}",
        help="Path template for dataset outputs.",
    )
    parser.add_argument(
        "--eval-file-template",
        default="results/eval_noskill_{dataset}.json",
        help="Path template for no-skill eval JSON files.",
    )
    parser.add_argument(
        "--probe-subpath",
        default="probe/probe.pkl",
        help="Probe relative path inside output directory.",
    )
    parser.add_argument(
        "--hidden-states-subpath",
        default="features/test/hidden_states.pt",
        help="Hidden states relative path inside output directory.",
    )
    parser.add_argument(
        "--items-subpath",
        default="features/test/items.json",
        help="Test items relative path inside output directory.",
    )
    parser.add_argument(
        "--concatenate-layers",
        action="store_true",
        help="Concatenate all layers when preparing features (default last layer only).",
    )

    args = parser.parse_args()

    for ds in args.datasets:
        output_dir = build_output_dir(args.output_template, ds)
        eval_path = build_eval_path(args.eval_file_template, ds)
        log_dataset(
            ds,
            output_dir,
            eval_path,
            args.probe_subpath,
            args.hidden_states_subpath,
            args.items_subpath,
            args.concatenate_layers,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
