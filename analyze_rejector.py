#!/usr/bin/env python3
"""Compare no-skill baseline with rejector results.

For each dataset, shows:
- How many questions the model got RIGHT without any skill
- How many questions the model got WRONG without any skill  
- Among WRONG ones: how many did the rejector decide to USE a skill?
- Among RIGHT ones: how many did the rejector decide to USE a skill (unnecessary)?
- Impact: did using skill actually help fix wrong answers?

Usage:
    python3 analyze_rejector.py
"""

import json
import sys
from pathlib import Path

DATASETS = ["medcalcbench", "theoremqa", "logicbench", "champ", "bigcodebench", "toolqa"]
RESULTS_DIR = Path("results")


def load_eval(prefix: str, dataset: str) -> dict[str, bool] | None:
    """Load eval results, return {instance_id: correct}."""
    path = RESULTS_DIR / f"eval_{prefix}_{dataset}.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    return {d["instance_id"]: d["correct"] for d in data["details"]}


def load_inference_skills(prefix: str, dataset: str) -> dict[str, list] | None:
    """Load inference results, return {instance_id: skill_ids_used}."""
    path = RESULTS_DIR / f"{prefix}_{dataset}.jsonl"
    if not path.exists():
        return None
    result = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            result[rec["instance_id"]] = rec.get("skill_ids_used", [])
    return result


def analyze_dataset(dataset: str):
    """Analyze one dataset."""
    noskill_eval = load_eval("noskill", dataset)
    rejector_eval = load_eval("rejector", dataset)
    rejector_skills = load_inference_skills("rejector", dataset)

    if noskill_eval is None:
        print(f"  ⚠ No-skill eval not found: eval_noskill_{dataset}.json")
        return None
    if rejector_eval is None:
        print(f"  ⚠ Rejector eval not found: eval_rejector_{dataset}.json")
        return None
    if rejector_skills is None:
        print(f"  ⚠ Rejector inference not found: rejector_{dataset}.jsonl")
        return None

    # Find common instance IDs
    common_ids = set(noskill_eval.keys()) & set(rejector_eval.keys()) & set(rejector_skills.keys())

    # Categorize
    noskill_correct = {iid for iid in common_ids if noskill_eval[iid]}
    noskill_wrong = {iid for iid in common_ids if not noskill_eval[iid]}
    used_skill = {iid for iid in common_ids if rejector_skills[iid]}
    no_used_skill = common_ids - used_skill

    # Cross analysis
    wrong_and_used_skill = noskill_wrong & used_skill      # GOOD: model was wrong, rejector gave skill
    wrong_and_no_skill = noskill_wrong & no_used_skill      # BAD: model was wrong, rejector rejected skill
    right_and_used_skill = noskill_correct & used_skill      # Unnecessary: model was right, rejector still gave skill
    right_and_no_skill = noskill_correct & no_used_skill     # GOOD: model was right, rejector skipped skill

    # Among wrong+used_skill, did the rejector actually help?
    wrong_used_skill_now_correct = {iid for iid in wrong_and_used_skill if rejector_eval[iid]}
    wrong_used_skill_still_wrong = wrong_and_used_skill - wrong_used_skill_now_correct

    # Among right+used_skill, did using skill break things?
    right_used_skill_still_correct = {iid for iid in right_and_used_skill if rejector_eval[iid]}
    right_used_skill_now_wrong = right_and_used_skill - right_used_skill_still_correct

    # Among wrong+no_skill, did not using skill keep them wrong?
    wrong_no_skill_still_wrong = {iid for iid in wrong_and_no_skill if not rejector_eval[iid]}
    wrong_no_skill_now_correct = wrong_and_no_skill - wrong_no_skill_still_wrong

    # Among right+no_skill, did skipping skill keep them right?
    right_no_skill_still_correct = {iid for iid in right_and_no_skill if rejector_eval[iid]}
    right_no_skill_now_wrong = right_and_no_skill - right_no_skill_still_correct

    # Rejector overall
    rejector_correct = sum(1 for iid in common_ids if rejector_eval[iid])

    return {
        "total": len(common_ids),
        "noskill_correct": len(noskill_correct),
        "noskill_wrong": len(noskill_wrong),
        "rejector_correct": rejector_correct,
        "used_skill": len(used_skill),
        "no_used_skill": len(no_used_skill),
        # Key cross-analysis
        "wrong_used_skill": len(wrong_and_used_skill),
        "wrong_no_skill": len(wrong_and_no_skill),
        "right_used_skill": len(right_and_used_skill),
        "right_no_skill": len(right_and_no_skill),
        # Deeper: did skill actually help?
        "wrong_used_skill_now_correct": len(wrong_used_skill_now_correct),
        "wrong_used_skill_still_wrong": len(wrong_used_skill_still_wrong),
        "right_used_skill_still_correct": len(right_used_skill_still_correct),
        "right_used_skill_now_wrong": len(right_used_skill_now_wrong),
        # Deeper: what happened when skill was skipped?
        "wrong_no_skill_still_wrong": len(wrong_no_skill_still_wrong),
        "wrong_no_skill_now_correct": len(wrong_no_skill_now_correct),
        "right_no_skill_still_correct": len(right_no_skill_still_correct),
        "right_no_skill_now_wrong": len(right_no_skill_now_wrong),
    }


def print_report(dataset: str, stats: dict):
    """Print detailed report for one dataset."""
    total = stats["total"]
    
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'='*70}")
    print(f"  Total instances: {total}")
    print()
    
    # Baseline
    nc = stats["noskill_correct"]
    nw = stats["noskill_wrong"]
    print(f"  📊 No-Skill Baseline:  {nc}/{total} correct ({nc/total:.1%})")
    print(f"     ✅ Correct without skill: {nc}")
    print(f"     ❌ Wrong without skill:   {nw}")
    print()
    
    # Rejector decisions
    us = stats["used_skill"]
    ns = stats["no_used_skill"]
    print(f"  🔧 Rejector Decisions:")
    print(f"     Used skill:    {us} ({us/total:.1%})")
    print(f"     Skipped skill: {ns} ({ns/total:.1%})")
    print()
    
    # Key cross-analysis
    print(f"  📋 Cross Analysis (No-Skill outcome × Rejector decision):")
    print(f"  ┌─────────────────────┬──────────────────┬──────────────────┐")
    print(f"  │                     │  Used Skill      │  Skipped Skill   │")
    print(f"  ├─────────────────────┼──────────────────┼──────────────────┤")
    wus = stats["wrong_used_skill"]
    wns = stats["wrong_no_skill"]
    rus = stats["right_used_skill"]
    rns = stats["right_no_skill"]
    print(f"  │ Baseline WRONG ({nw:3d}) │  {wus:4d} (helped?)  │  {wns:4d} (missed!)  │")
    print(f"  │ Baseline RIGHT ({nc:3d}) │  {rus:4d} (wasteful) │  {rns:4d} (correct!) │")
    print(f"  └─────────────────────┴──────────────────┴──────────────────┘")
    print()
    
    # Impact analysis
    print(f"  🎯 Impact Analysis:")
    wnc = stats["wrong_used_skill_now_correct"]
    wsw = stats["wrong_used_skill_still_wrong"]
    rsc = stats["right_used_skill_still_correct"]
    rnw = stats["right_used_skill_now_wrong"]
    wnsw = stats["wrong_no_skill_still_wrong"]
    wnsc = stats["wrong_no_skill_now_correct"]
    rnsc = stats["right_no_skill_still_correct"]
    rnsw = stats["right_no_skill_now_wrong"]

    print(f"     When skill WAS used:")
    if wus > 0:
        print(f"       Wrong → Used skill → Now CORRECT:    {wnc:4d}/{wus} ({wnc/wus:.1%}) ✅ SAVED")
        print(f"       Wrong → Used skill → Still WRONG:    {wsw:4d}/{wus} ({wsw/wus:.1%}) 😕 skill didn't help")
    else:
        print(f"       (No cases where baseline was wrong and skill was used)")
    if rus > 0:
        print(f"       Right → Used skill → Still CORRECT:  {rsc:4d}/{rus} ({rsc/rus:.1%}) ✅ OK (no harm)")
        print(f"       Right → Used skill → Now WRONG:      {rnw:4d}/{rus} ({rnw/rus:.1%}) 💥 BROKEN by skill")
    else:
        print(f"       (No cases where baseline was right and skill was used)")

    print(f"     When skill was SKIPPED:")
    if wns > 0:
        print(f"       Wrong → No skill  → Still WRONG:     {wnsw:4d}/{wns} ({wnsw/wns:.1%}) 😕 missed opportunity")
        print(f"       Wrong → No skill  → Now CORRECT:     {wnsc:4d}/{wns} ({wnsc/wns:.1%}) 🤔 lucky")
    else:
        print(f"       (No cases where baseline was wrong and skill was skipped)")
    if rns > 0:
        print(f"       Right → No skill  → Still CORRECT:   {rnsc:4d}/{rns} ({rnsc/rns:.1%}) ✅ PERFECT decision")
        print(f"       Right → No skill  → Now WRONG:       {rnsw:4d}/{rns} ({rnsw/rns:.1%}) 😕 inconsistent")
    else:
        print(f"       (No cases where baseline was right and skill was skipped)")
    print()
    
    # Overall comparison
    rc = stats["rejector_correct"]
    delta = rc - nc
    sign = "+" if delta >= 0 else ""
    print(f"  📈 Overall:")
    print(f"     No-Skill accuracy:  {nc/total:.1%} ({nc}/{total})")
    print(f"     Rejector accuracy:  {rc/total:.1%} ({rc}/{total})")
    print(f"     Delta:              {sign}{delta} ({sign}{delta/total:.1%})")


def main():
    print("=" * 70)
    print("  REJECTOR vs NO-SKILL BASELINE — FULL ANALYSIS")
    print("=" * 70)

    all_stats = {}
    for dataset in DATASETS:
        print(f"\n  Loading {dataset}...")
        stats = analyze_dataset(dataset)
        if stats:
            all_stats[dataset] = stats
            print_report(dataset, stats)

    # Summary table
    if all_stats:
        print(f"\n{'='*70}")
        print(f"  SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"  {'Dataset':<15} {'NoSkill':>8} {'Rejector':>9} {'Delta':>7} {'SkillUsed':>10} {'Saved':>6} {'Broken':>7}")
        print(f"  {'-'*15} {'-'*8} {'-'*9} {'-'*7} {'-'*10} {'-'*6} {'-'*7}")
        
        total_ns = total_rej = total_total = 0
        total_saved = total_broken = 0
        for ds, s in all_stats.items():
            nc = s["noskill_correct"]
            rc = s["rejector_correct"]
            t = s["total"]
            delta = rc - nc
            sign = "+" if delta >= 0 else ""
            us = s["used_skill"]
            saved = s["wrong_used_skill_now_correct"]
            broken = s["right_used_skill_now_wrong"]
            print(f"  {ds:<15} {nc/t:>7.1%} {rc/t:>8.1%} {sign+str(delta):>7} {us:>10} {saved:>6} {broken:>7}")
            total_ns += nc
            total_rej += rc
            total_total += t
            total_saved += saved
            total_broken += broken
        
        if total_total > 0:
            d = total_rej - total_ns
            sign = "+" if d >= 0 else ""
            print(f"  {'-'*15} {'-'*8} {'-'*9} {'-'*7} {'-'*10} {'-'*6} {'-'*7}")
            print(f"  {'TOTAL':<15} {total_ns/total_total:>7.1%} {total_rej/total_total:>8.1%} {sign+str(d):>7} {'':>10} {total_saved:>6} {total_broken:>7}")
        
        print()
        print(f"  💡 Legend:")
        print(f"     Saved  = baseline wrong → used skill → now correct")
        print(f"     Broken = baseline right → used skill → now wrong")
        print(f"     Delta  = Rejector correct − NoSkill correct")

    # Save results to JSON
    output_path = RESULTS_DIR / "analysis_rejector_vs_noskill.json"
    json.dump(all_stats, open(output_path, "w"), indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
