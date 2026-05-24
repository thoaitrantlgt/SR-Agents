#!/usr/bin/env python3
import json
from pathlib import Path

DATASETS = ['medcalcbench', 'theoremqa', 'logicbench', 'champ', 'bigcodebench', 'toolqa']
R = Path('results')

# Collect stats
all_stats = []
for ds in DATASETS:
    noskill = json.load(open(R / f'eval_noskill_{ds}.json'))
    rejector = json.load(open(R / f'eval_oracle_{ds}.json'))
    ns_map = {d['instance_id']: d['correct'] for d in noskill['details']}
    rj_map = {d['instance_id']: d['correct'] for d in rejector['details']}

    skills = {}
    with open(R / f'oracle_{ds}.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            skills[rec['instance_id']] = bool(rec.get('skill_ids_used'))

    common = set(ns_map) & set(rj_map) & set(skills)
    used = {iid for iid in common if skills[iid]}
    skipped = common - used

    wrong_total = sum(1 for i in common if not ns_map[i])
    wrong_used_right = sum(1 for i in used if not ns_map[i] and rj_map[i])
    wrong_used_wrong = sum(1 for i in used if not ns_map[i] and not rj_map[i])
    wrong_skip_right = sum(1 for i in skipped if not ns_map[i] and rj_map[i])
    wrong_skip_wrong = sum(1 for i in skipped if not ns_map[i] and not rj_map[i])

    right_total = sum(1 for i in common if ns_map[i])
    right_used_right = sum(1 for i in used if ns_map[i] and rj_map[i])
    right_used_wrong = sum(1 for i in used if ns_map[i] and not rj_map[i])
    right_skip_right = sum(1 for i in skipped if ns_map[i] and rj_map[i])
    right_skip_wrong = sum(1 for i in skipped if ns_map[i] and not rj_map[i])

    all_stats.append({
        'ds': ds,
        'wrong_total': wrong_total,
        'wrong_used_right': wrong_used_right,
        'wrong_used_wrong': wrong_used_wrong,
        'wrong_skip_right': wrong_skip_right,
        'wrong_skip_wrong': wrong_skip_wrong,
        'right_total': right_total,
        'right_used_right': right_used_right,
        'right_used_wrong': right_used_wrong,
        'right_skip_right': right_skip_right,
        'right_skip_wrong': right_skip_wrong,
    })

# ── PART 1: Originally WRONG ──
print()
print('='*120)
print('  PART 1: ORIGINALLY WRONG (without skill)')
print('='*120)
print(f"  {'Dataset':<15} {'Wrong':>5}  {'Used>Right':>12} {'Used>Wrong':>12} {'Skip>Right':>12} {'Skip>Wrong':>12}")
print(f"  {'-'*15} {'-'*5}  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

tw = tw_ur = tw_uw = tw_sr = tw_sw = 0
for s in all_stats:
    w = s['wrong_total']
    ur = s['wrong_used_right']
    uw = s['wrong_used_wrong']
    sr = s['wrong_skip_right']
    sw = s['wrong_skip_wrong']
    def pct(n, d): return f"{n/d:.1%}" if d else "-"
    print(f"  {s['ds']:<15} {w:>5}  {ur:>5} ({pct(ur,w):>5}) {uw:>5} ({pct(uw,w):>5}) {sr:>5} ({pct(sr,w):>5}) {sw:>5} ({pct(sw,w):>5})")
    tw += w; tw_ur += ur; tw_uw += uw; tw_sr += sr; tw_sw += sw

print(f"  {'-'*15} {'-'*5}  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'TOTAL':<15} {tw:>5}  {tw_ur:>5} ({tw_ur/tw:.1%}) {tw_uw:>5} ({tw_uw/tw:.1%}) {tw_sr:>5} ({tw_sr/tw:.1%}) {tw_sw:>5} ({tw_sw/tw:.1%})")
print(f"  Check: {tw_ur}+{tw_uw}+{tw_sr}+{tw_sw} = {tw_ur+tw_uw+tw_sr+tw_sw} (should be {tw})")

# ── PART 2: Originally RIGHT ──
print()
print('='*120)
print('  PART 2: ORIGINALLY RIGHT (without skill)')
print('='*120)
print(f"  {'Dataset':<15} {'Right':>5}  {'Used>Right':>12} {'Used>Wrong':>12} {'Skip>Right':>12} {'Skip>Wrong':>12}")
print(f"  {'-'*15} {'-'*5}  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

tr = tr_ur = tr_uw = tr_sr = tr_sw = 0
for s in all_stats:
    r = s['right_total']
    ur = s['right_used_right']
    uw = s['right_used_wrong']
    sr = s['right_skip_right']
    sw = s['right_skip_wrong']
    def pct(n, d): return f"{n/d:.1%}" if d else "-"
    print(f"  {s['ds']:<15} {r:>5}  {ur:>5} ({pct(ur,r):>5}) {uw:>5} ({pct(uw,r):>5}) {sr:>5} ({pct(sr,r):>5}) {sw:>5} ({pct(sw,r):>5})")
    tr += r; tr_ur += ur; tr_uw += uw; tr_sr += sr; tr_sw += sw

print(f"  {'-'*15} {'-'*5}  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'TOTAL':<15} {tr:>5}  {tr_ur:>5} ({tr_ur/tr:.1%}) {tr_uw:>5} ({tr_uw/tr:.1%}) {tr_sr:>5} ({tr_sr/tr:.1%}) {tr_sw:>5} ({tr_sw/tr:.1%})")
print(f"  Check: {tr_ur}+{tr_uw}+{tr_sr}+{tr_sw} = {tr_ur+tr_uw+tr_sr+tr_sw} (should be {tr})")

print()
print("  Legend:")
print("    Used>Right = gate accepted skill, answer became/stayed correct   (SAVED)")
print("    Used>Wrong = gate accepted skill, answer still/became wrong      (WASTED)")
print("    Skip>Right = gate rejected skill, answer correct without it      (GOOD REJECT)")
print("    Skip>Wrong = gate rejected skill, answer wrong without it        (BAD REJECT)")
print(f"    Net gain = Saved({tw_ur}) - Broken({tr_uw}) = +{tw_ur - tr_uw}")
