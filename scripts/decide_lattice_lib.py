"""Verify the lattice through the real library path, not the prototype path.

scripts/decide_lattice.py measured a prototype that polished each hypothesis
before aligning them and compared in lower case. The shipped library does
neither: it aligns raw engine output inside chorus.consensus and polishes once
afterwards in the pipeline. Those are different computations, so the prototype
result does not transfer for free.

This script toggles chorus.consensus.LATTICE_ENABLED and replays the exact
shipped code path both ways over the same cached hypotheses, paired per crop,
with the same 4,000-round bootstrap. Tuning split only.
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
from chorus import consensus, lang  # noqa: E402


def per_sample(records, mode, lattice_on):
    cfg = MODES[mode]
    previous = consensus.LATTICE_ENABLED
    consensus.LATTICE_ENABLED = lattice_on
    try:
        out = []
        for rec in records:
            hyps = weighted(rec, cfg["engines"], cfg["profile"])
            gl = norm(rec["gt"]).lower()
            if not hyps:
                out.append((0, 1.0))
                continue
            fused = consensus.fuse(hyps, mode=rec["route"])
            pred = norm(lang.polish(fused["text"], mode=rec["route"])).lower()
            out.append((int(pred == gl), cer(gl, pred)))
        return out
    finally:
        consensus.LATTICE_ENABLED = previous


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
        return "KAFESSIZ KAZANIYOR"
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
            off = per_sample(records, mode, False)
            on = per_sample(records, mode, True)
            n = len(off)
            print(f"  {mode:9s} kafessiz acc={sum(h for h, _ in off)/n:.4f}"
                  f" cer={sum(c for _, c in off)/n:.4f}"
                  f"   kafes acc={sum(h for h, _ in on)/n:.4f}"
                  f" cer={sum(c for _, c in on)/n:.4f}")
            p, lo, hi = bootstrap(off, on, field=0, sign=1)
            print(f"            acc fark={p:+.4f}  %95=[{lo:+.4f}, {hi:+.4f}]  -> {verdict(lo, hi)}")
            p, lo, hi = bootstrap(off, on, field=1, sign=-1)
            print(f"            cer fark={p:+.4f}  %95=[{lo:+.4f}, {hi:+.4f}]  -> {verdict(lo, hi)}")
        print()


if __name__ == "__main__":
    main()
