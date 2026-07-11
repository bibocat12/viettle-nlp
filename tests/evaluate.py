#!/usr/bin/env python3
"""Compare pipeline output against ground truth."""

import json
from pathlib import Path

GT_PATH = Path(__file__).parent / "ground_truth.json"


def load_gt():
    with open(GT_PATH) as f:
        return json.load(f)["files"]


def load_pipe(fid):
    p = Path(__file__).parent.parent / "outputs_new" / f"{fid}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        return data.get("mentions", [])
    return data


def evaluate(gt_list, pipe_list):
    tp = fp = fn = 0
    details = []
    used_gt = set()

    for pi, p in enumerate(pipe_list):
        best = None
        for gi, g in enumerate(gt_list):
            if gi in used_gt:
                continue
            s = max(p["position"][0], g["position"][0])
            e = min(p["position"][1], g["position"][1])
            if e - s > 0 and p["type"] == g["type"]:
                best = gi
                break
        if best is not None:
            tp += 1
            used_gt.add(best)
            details.append(f"  TP: {p['type']:20s} \"{p['text']}\"")
        else:
            fp += 1
            details.append(f"  FP: {p['type']:20s} \"{p['text']}\"")

    for gi, g in enumerate(gt_list):
        if gi not in used_gt:
            fn += 1
            details.append(f"  FN: {g['type']:20s} \"{g['text']}\"")

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"f1": round(f1,3), "p": round(p,3), "r": round(r,3),
            "tp": tp, "fp": fp, "fn": fn, "details": details}


def main():
    gt_all = load_gt()
    all_tp = all_fp = all_fn = 0

    for fid in sorted(gt_all):
        data = gt_all[fid]
        pipe = load_pipe(fid)
        s = evaluate(data["entities"], pipe)
        all_tp += s["tp"]; all_fp += s["fp"]; all_fn += s["fn"]
        print(f"File {fid}: P={s['p']:.3f} R={s['r']:.3f} F1={s['f1']:.3f}  GT={len(data['entities'])} Pipe={len(pipe)}")
        for d in s["details"]:
            print(d)
        print()

    # Recalculate overall
    p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0
    r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"OVERALL: P={p:.3f} R={r:.3f} F1={f1:.3f}  TP={all_tp} FP={all_fp} FN={all_fn}")


if __name__ == "__main__":
    main()
