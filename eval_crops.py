"""
Score model outputs against test.jsonl.
Metrics: JSON-valid rate, exact match per crop, field-level precision/recall/F1,
hallucination rate (= 1 - precision), per kind (dimension / gdt).

Input predictions file: JSONL with {"image": ..., "pred": "<raw model text>"}
Usage: python eval_crops.py --test out/test.jsonl --pred preds_zeroshot.jsonl
"""
import argparse, json, re
from collections import defaultdict


def extract_json(text):
    """Pull the first {...} block out of model text; return dict or None."""
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def norm(v):
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return f"{float(v):g}"
    s = str(v).strip().replace("Ø", "⌀").replace("ø", "⌀")
    try:
        return f"{float(s):g}"
    except ValueError:
        return s.lower()


def flatten(d, prefix=""):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            out |= flatten(v, f"{prefix}{k}.")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out |= flatten(v, f"{prefix}{i}.")
    else:
        out.add((prefix.rstrip("."), norm(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()

    gt = {r["image"]: r for r in map(json.loads, open(a.test, encoding="utf-8"))}
    preds = {r["image"]: r["pred"] for r in map(json.loads, open(a.pred, encoding="utf-8"))}

    stats = defaultdict(lambda: {"n": 0, "valid": 0, "em": 0, "tp": 0, "fp": 0, "fn": 0})
    for img, row in gt.items():
        kind = row["kind"]
        for key in (kind, "all"):
            s = stats[key]
            s["n"] += 1
            g = json.loads(row["answer"])
            p = extract_json(preds.get(img))
            if p is None:
                s["fn"] += len(flatten(g))
                continue
            s["valid"] += 1
            G, P = flatten(g), flatten(p)
            if G == P:
                s["em"] += 1
            s["tp"] += len(P & G); s["fp"] += len(P - G); s["fn"] += len(G - P)

    print(f"{'kind':10} {'n':>4} {'json_ok':>8} {'exact':>7} {'prec':>6} {'rec':>6} {'f1':>6} {'halluc':>7}")
    for key in sorted(stats):
        s = stats[key]
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{key:10} {s['n']:4d} {s['valid']/s['n']:8.2f} {s['em']/s['n']:7.2f} "
              f"{prec:6.2f} {rec:6.2f} {f1:6.2f} {1-prec:7.2f}")


if __name__ == "__main__":
    main()
