#!/usr/bin/env python3
"""
Produce per-dataset logs showing whether the model (rejector) would choose "load_skill" for
cases where the no-skill model is wrong versus cases where the no-skill model is correct,
filtered strictly to the probe test set.

Example:
  python3 eval_model_loadskill_all_log_test.py
"""

import argparse
import json
import sys
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(
        description="Log whether the rejector chooses load_skill for no-skill wrong vs correct cases on the test set."
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
        "--items-subpath",
        default="features/test/items.json",
        help="Test items relative path inside output directory.",
    )
    parser.add_argument(
        "--eval-file-template",
        default="results/eval_noskill_{dataset}.json",
        help="Path template for no-skill eval JSON files.",
    )
    parser.add_argument(
        "--rejector-file-template",
        default="results/rejector_{dataset}.jsonl",
        help="Path template for rejector inference JSONL files.",
    )

    args = parser.parse_args()

    # Aggregate counts
    agg_wrong_load = agg_wrong_skip = agg_correct_load = agg_correct_skip = 0

    print("Model self-decision load_skill summary for probe test set")
    print("=========================================================")

    for ds in args.datasets:
        output_dir = build_output_dir(args.output_template, ds)
        items_path = output_dir / args.items_subpath
        eval_path = Path(args.eval_file_template.format(dataset=ds))
        rejector_path = Path(args.rejector_file_template.format(dataset=ds))

        if not items_path.exists():
            print(f"\nDataset: {ds} | SKIP (items file not found: {items_path})")
            continue
        if not eval_path.exists():
            print(f"\nDataset: {ds} | SKIP (eval file not found: {eval_path})")
            continue
        if not rejector_path.exists():
            print(f"\nDataset: {ds} | SKIP (rejector file not found: {rejector_path})")
            continue

        # Load test items to get test_ids
        test_items = load_json(items_path)
        if isinstance(test_items, dict):
            test_items = list(test_items.values())
        test_ids = {item.get("instance_id") for item in test_items if item.get("instance_id")}

        # Load no-skill eval mapping
        eval_data = load_json(eval_path)
        details = eval_data.get("details", [])
        if isinstance(details, dict):
            eval_map = details
        else:
            eval_map = {d["instance_id"]: d for d in details if "instance_id" in d}

        # Load rejector decisions
        rejector_decisions = {}
        with open(rejector_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                iid = rec.get("instance_id")
                if iid:
                    # load_skill if skill_ids_used has items
                    skills_used = rec.get("skill_ids_used", [])
                    rejector_decisions[iid] = len(skills_used) > 0

        wrong_load = wrong_skip = correct_load = correct_skip = 0
        missing_rejector = 0
        missing_eval = 0

        for iid in test_ids:
            if iid not in eval_map:
                missing_eval += 1
                continue
            if iid not in rejector_decisions:
                missing_rejector += 1
                continue

            correct = eval_map[iid].get("correct")
            load_skill = rejector_decisions[iid]

            if correct is True:
                if load_skill:
                    correct_load += 1
                else:
                    correct_skip += 1
            elif correct is False:
                if load_skill:
                    wrong_load += 1
                else:
                    wrong_skip += 1

        total_wrong = wrong_load + wrong_skip
        total_correct = correct_load + correct_skip

        wrong_load_rate = (wrong_load / total_wrong) if total_wrong > 0 else 0.0
        correct_load_rate = (correct_load / total_correct) if total_correct > 0 else 0.0

        print(f"\nDataset: {ds}")
        print(f"  no-skill wrong:  load_skill={wrong_load}, no_skill={wrong_skip}, total={total_wrong}")
        print(f"  no-skill correct: load_skill={correct_load}, no_skill={correct_skip}, total={total_correct}")
        print(f"    wrong load rate: {wrong_load_rate:.1%}")
        print(f"    correct load rate: {correct_load_rate:.1%}")
        if missing_eval > 0 or missing_rejector > 0:
            print(f"    (skipped test items: missing_eval={missing_eval}, missing_rejector={missing_rejector})")

        agg_wrong_load += wrong_load
        agg_wrong_skip += wrong_skip
        agg_correct_load += correct_load
        agg_correct_skip += correct_skip

    total_agg_wrong = agg_wrong_load + agg_wrong_skip
    total_agg_correct = agg_correct_load + agg_correct_skip

    agg_wrong_rate = (agg_wrong_load / total_agg_wrong) if total_agg_wrong > 0 else 0.0
    agg_correct_rate = (agg_correct_load / total_agg_correct) if total_agg_correct > 0 else 0.0

    print("\nAggregate across all datasets (Probe Test Set):")
    print(f"  no-skill wrong:  load_skill={agg_wrong_load}, no_skill={agg_wrong_skip}, total={total_agg_wrong}")
    print(f"  no-skill correct: load_skill={agg_correct_load}, no_skill={agg_correct_skip}, total={total_agg_correct}")
    print(f"    wrong load rate: {agg_wrong_rate:.1%}")
    print(f"    correct load rate: {agg_correct_rate:.1%}")

if __name__ == "__main__":
    main()
