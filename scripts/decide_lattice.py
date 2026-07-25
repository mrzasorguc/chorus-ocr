"""Does character-lattice fusion beat the current fuser? With an error bar.

Same discipline as scripts/decide_fusion.py: both methods are replayed over the
exact same cached hypotheses so every comparison is paired, and the difference
is reported with a 4,000-round paired bootstrap 95% interval. A change is only
worth adopting when the interval stays on one side of zero.

The candidate is fixed before measuring, with no per-mode constants:

  scene route     -> unchanged, keep the existing selector (a scene crop is one
                     word, so picking a whole hypothesis is the right move)
  document route  -> character lattice, one ballot per engine per column,
                     no deletion bias (the neutral setting)

Tuning split only (skip=1000). The test split is never loaded here.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from exp_charfuse import load_dump  # noqa: E402
from exp_lattice_rules import decide_engvote, lattice  # noqa: E402
from chorus import consensus, lang  # noqa: E402


def lattice_pred(hyps, route):
    low = [dict(h, text=norm(lang.polish(h["text"], mode=route)).lower())
           for h in hyps]
    cols, weights = lattice(low)
    text = "".join(decide_engvote(col, low, weights, 1.0) for col in cols)
    return norm(text).lower()


def baseline_pred(hyps, route):
    fused = consensus.fuse(hyps, mode=route)
    return norm(lang.polish(fused["text"], mode=route)).lower()


def per_sample(records, mode, use_lattice):
    cfg = MODES[mode]
    out = []
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        gl = norm(rec["gt"]).lower()
        if not hyps:
            out.append((0, 1.0))
            continue
        route = rec["route"]
        if use_lattice and route != "scene":
            pred = lattice_pred(hyps, route)
        else:
            pred = baseline_pred(hyps, route)
        out.append((int(pred == gl), cer(gl, pred)))
    return out


def bootstrap(a, b, field=0, sign=1, rounds=4000, seed=12345):
    rng = random.Random(seed)
    n = len(a)
    diffs = [sign * (b[i][field] - a[i][field]) for i in range(n)]
    point = sum(diffs) / n
    idx = range(n)
    samples = []
    for _ in range(rounds):
        picks = [rng.choice(idx) for _ in range(n)]
        samples.append(sum(diffs[i] for i in picks) / n)
    samples.sort()
    return point, samples[int(0.025 * rounds)], samples[int(0.975 * rounds) - 1]


def verdict(lo, hi):
    if lo > 0:
        return "KAFES KAZANIYOR"
    if hi < 0:
        return "MEVCUT KAZANIYOR"
    return "AYIRT EDILEMIYOR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    for dataset in args.datasets.split(","):
        records = load_dump(dataset, args.n, args.skip)
        print(f"===== {dataset.upper()}  (tuning split, n={len(records)})")
        for mode in ("standard", "max", "maxq"):
            base = per_sample(records, mode, False)
            cand = per_sample(records, mode, True)
            n = len(base)
            b_acc = sum(h for h, _ in base) / n
            c_acc = sum(h for h, _ in cand) / n
            b_cer = sum(c for _, c in base) / n
            c_cer = sum(c for _, c in cand) / n
            print(f"  {mode:9s} mevcut acc={b_acc:.4f} cer={b_cer:.4f}"
                  f"   kafes acc={c_acc:.4f} cer={c_cer:.4f}")
            p, lo, hi = bootstrap(base, cand, field=0, sign=1)
            print(f"            acc fark={p:+.4f}  %95=[{lo:+.4f}, {hi:+.4f}]  -> {verdict(lo, hi)}")
            p, lo, hi = bootstrap(base, cand, field=1, sign=-1)
            print(f"            cer fark={p:+.4f}  %95=[{lo:+.4f}, {hi:+.4f}]  -> {verdict(lo, hi)}")
        print()


if __name__ == "__main__":
    main()
