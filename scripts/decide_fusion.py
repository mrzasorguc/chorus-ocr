"""Decide between the legacy and current selector with a stated error bar.

Why this script exists: the selector in chorus.consensus was adopted after it
gained about three points on a 60-sample tuning split. On the held-out test
split it gained three points in one mode and lost three in another. Sixty
samples simply cannot resolve a three point difference -- three points is two
samples -- so the earlier decision was made on noise.

Every comparison here is therefore paired (both selectors judged on the exact
same crops) and reported with a bootstrap interval over a larger tuning split.
A change is only worth adopting when the interval for the difference stays on
one side of zero.

The test split is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from chorus import consensus, lang  # noqa: E402

def load_dump(dataset, n, skip=1000):
    path = os.path.join(ROOT, "out", "research_20260724",
                        f"hyps_{dataset}_skip{skip}_n{n}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["records"]

def per_sample(records, mode, selector):
    """Return one (hit, cer) pair per crop so comparisons stay paired."""
    cfg = MODES[mode]
    out = []
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        if not hyps:
            out.append((0, 1.0))
            continue
        route = rec["route"]
        short = max(len(h["text"].split()) for h in hyps) <= consensus.SPLICE_MIN_TOKENS
        if short:
            fused = selector(hyps, route)
        else:
            fused = consensus.fuse(hyps, mode=route)
        pred = lang.polish(fused["text"], mode=route)
        ref = rec["gt"]
        out.append((int(norm(pred).lower() == norm(ref).lower()), cer(ref, pred)))
    return out

SELECTORS = {
    "legacy": lambda hyps, route: consensus._mbr(hyps),
    "current": lambda hyps, route: consensus._select(hyps, route),
}

def bootstrap_diff(a, b, rounds=4000, seed=12345, field=0, sign=1):
    """Paired bootstrap over a per-sample difference.

    field=0 compares accuracy (higher is better). field=1 compares CER, where
    lower is better, so pass sign=-1 to keep "positive means current wins".
    """
    rng = random.Random(seed)
    n = len(a)
    diffs = [sign * (b[i][field] - a[i][field]) for i in range(n)]
    point = sum(diffs) / n
    samples = []
    idx = range(n)
    for _ in range(rounds):
        picks = [rng.choice(idx) for _ in range(n)]
        samples.append(sum(diffs[i] for i in picks) / n)
    samples.sort()
    lo = samples[int(0.025 * rounds)]
    hi = samples[int(0.975 * rounds) - 1]
    return point, lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    for dataset in args.datasets.split(","):
        try:
            records = load_dump(dataset, args.n, args.skip)
        except FileNotFoundError:
            print(f"[atlandi] {dataset}: n={args.n} dokumu henuz yok")
            continue
        print(f"===== {dataset.upper()}  (tuning split, n={len(records)})")
        for mode in ("standard", "max", "maxq"):
            res = {k: per_sample(records, mode, fn) for k, fn in SELECTORS.items()}
            acc = {k: sum(h for h, _ in v) / len(v) for k, v in res.items()}
            cers = {k: sum(c for _, c in v) / len(v) for k, v in res.items()}
            point, lo, hi = bootstrap_diff(res["legacy"], res["current"])
            if lo > 0:
                verdict = "CURRENT KAZANIYOR"
            elif hi < 0:
                verdict = "LEGACY KAZANIYOR"
            else:
                verdict = "AYIRT EDILEMIYOR"
            print(f"  {mode:9s} legacy acc={acc['legacy']:.4f} cer={cers['legacy']:.4f}"
                  f"   current acc={acc['current']:.4f} cer={cers['current']:.4f}")
            print(f"            acc  fark={point:+.4f}  %95 araligi=[{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
            cpoint, clo, chi = bootstrap_diff(res["legacy"], res["current"], field=1, sign=-1)
            if clo > 0:
                cverdict = "CURRENT KAZANIYOR"
            elif chi < 0:
                cverdict = "LEGACY KAZANIYOR"
            else:
                cverdict = "AYIRT EDILEMIYOR"
            print(f"            cer  fark={cpoint:+.4f}  %95 araligi=[{clo:+.4f}, {chi:+.4f}]  -> {cverdict}")
        print()

if __name__ == "__main__":
    main()
