"""Paired bootstrap: lattice vote against lattice + beam + language prior.

One global lambda is used for every mode and both datasets. A per-mode lambda
scored higher on this split, and that is precisely the shape of result that
failed to transfer last time, so it is not on the table.

A change is adopted only when the whole 95% interval sits on one side of zero,
for word accuracy and character error rate separately.
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
from exp_beam import beam_candidates  # noqa: E402
from exp_prior import pick, polish  # noqa: E402
from chorus import consensus  # noqa: E402

LAM = 0.2


def per_sample(records, mode, use_prior):
    cfg = MODES[mode]
    out = []
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        if not hyps:
            continue
        gl = norm(rec["gt"]).lower()
        route = rec["route"]
        if use_prior and route != "scene":
            pred = polish(pick(beam_candidates(hyps), LAM), route)
        else:
            prev = consensus.LATTICE_ENABLED
            consensus.LATTICE_ENABLED = True
            try:
                pred = polish(consensus.fuse(hyps, mode=route)["text"], route)
            finally:
                consensus.LATTICE_ENABLED = prev
        out.append((int(pred == gl), cer(gl, pred)))
    return out


def bootstrap(a, b, field=0, sign=1, rounds=4000, seed=12345):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i][field] for i in idx) / n
        db = sum(b[i][field] for i in idx) / n
        diffs.append(sign * (db - da))
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]


def verdict(lo, hi):
    if lo > 0:
        return "ONSEL KAZANIYOR"
    if hi < 0:
        return "ONSELSIZ KAZANIYOR"
    return "AYIRT EDILEMIYOR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    print(f"lambda={LAM} (tum modlar ve veri kumeleri icin tek sabit)\n")
    for dataset in args.datasets.split(","):
        records = load_dump(dataset, args.n, args.skip)
        for mode in ("standard", "max", "maxq"):
            base = per_sample(records, mode, False)
            new = per_sample(records, mode, True)
            bacc = sum(x[0] for x in base) / len(base)
            nacc = sum(x[0] for x in new) / len(new)
            bcer = sum(x[1] for x in base) / len(base)
            ncer = sum(x[1] for x in new) / len(new)
            print(f"{dataset} {mode}")
            print(f"  onselsiz acc={bacc:.4f} cer={bcer:.4f}   onsel acc={nacc:.4f} cer={ncer:.4f}")
            lo, hi = bootstrap(base, new, field=0, sign=1)
            print(f"  acc fark={nacc - bacc:+.4f} %95=[{lo:+.4f}, {hi:+.4f}] -> {verdict(lo, hi)}")
            lo, hi = bootstrap(base, new, field=1, sign=-1)
            print(f"  cer fark={bcer - ncer:+.4f} %95=[{lo:+.4f}, {hi:+.4f}] -> {verdict(lo, hi)}")
            print()


if __name__ == "__main__":
    main()
