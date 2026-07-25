"""When should the character lattice run at all?

The v5 test-split measurement showed the lattice cutting character error hard
in the GOT-enabled modes while losing three points of word accuracy in the
standard profile. The hypothesis is mechanical rather than dataset-specific:
the lattice aligns every hypothesis to the heaviest one, so a strong pivot
turns the other engines into a spell-checker, while a weak pivot lets them
blend noise into a string nobody read.

This script tests that hypothesis on the tuning split with gates built only
from signals available at inference time. Which dataset a crop came from is
not such a signal and is never used as one.

Every gate is evaluated over the same cached predictions, so the sweep costs
one pass rather than one pass per threshold.
"""
from __future__ import annotations

import argparse
import os
import sys
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from exp_charfuse import load_dump  # noqa: E402
from chorus import consensus, lang  # noqa: E402


def engine_of(hyp):
    return (hyp.get("src", "") or "").split(":", 1)[0]


def features(hyps):
    """Signals describing how much the heaviest hypothesis can be trusted."""
    pivot = max(hyps, key=lambda h: h["weight"])
    total = sum(h["weight"] for h in hyps) or 1.0
    others = [h for h in hyps if h is not pivot]
    if others:
        sims = [SequenceMatcher(None, pivot["text"], h["text"]).ratio() for h in others]
        mean_sim = sum(sims) / len(sims)
    else:
        mean_sim = 1.0
    return {
        "pivot_share": pivot["weight"] / total,
        "pivot_weight": pivot["weight"],
        "pivot_conf": pivot.get("conf", 0.0),
        "mean_sim": mean_sim,
        "pivot_is_got": engine_of(pivot) == "got",
        "n_engines": len({engine_of(h) for h in hyps}),
    }


def predict(hyps, route, lattice_on):
    previous = consensus.LATTICE_ENABLED
    consensus.LATTICE_ENABLED = lattice_on
    try:
        fused = consensus.fuse(hyps, mode=route)
        return norm(lang.polish(fused["text"], mode=route)).lower()
    finally:
        consensus.LATTICE_ENABLED = previous


def collect(records, mode):
    """Per record: gold, both predictions, and the gate features."""
    cfg = MODES[mode]
    rows = []
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        gl = norm(rec["gt"]).lower()
        if not hyps:
            rows.append({"gl": gl, "scene": True, "base": (0, 1.0), "lat": (0, 1.0), "f": None})
            continue
        route = rec["route"]
        base = predict(hyps, route, False)
        if route == "scene":
            rows.append({"gl": gl, "scene": True,
                         "base": (int(base == gl), cer(gl, base)),
                         "lat": (int(base == gl), cer(gl, base)), "f": None})
            continue
        lat = predict(hyps, route, True)
        rows.append({"gl": gl, "scene": False,
                     "base": (int(base == gl), cer(gl, base)),
                     "lat": (int(lat == gl), cer(gl, lat)),
                     "f": features(hyps)})
    return rows


def score(rows, gate):
    n = len(rows)
    acc = num = 0.0
    used = 0
    for r in rows:
        on = (not r["scene"]) and gate(r["f"])
        used += int(on)
        hit, c = r["lat"] if on else r["base"]
        acc += hit
        num += c
    return acc / n, num / n, used


def gates():
    yield "kapali (v4)", lambda f: False
    yield "hep acik (v5)", lambda f: True
    yield "eksen=got", lambda f: f["pivot_is_got"]
    for t in (0.12, 0.16, 0.20, 0.25, 0.30):
        yield f"eksen_payi>={t:.2f}", (lambda t: lambda f: f["pivot_share"] >= t)(t)
    for t in (0.9, 1.1, 1.3, 1.5):
        yield f"eksen_agirlik>={t:.1f}", (lambda t: lambda f: f["pivot_weight"] >= t)(t)
    for t in (0.40, 0.55, 0.70):
        yield f"benzerlik>={t:.2f}", (lambda t: lambda f: f["mean_sim"] >= t)(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    cache = {ds: load_dump(ds, args.n, args.skip) for ds in datasets}

    for mode in ("standard", "max", "maxq"):
        print(f"===== {mode}  (tuning split, n={args.n})")
        rows = {ds: collect(cache[ds], mode) for ds in datasets}
        for name, gate in gates():
            parts = []
            for ds in datasets:
                acc, c, used = score(rows[ds], gate)
                parts.append(f"{ds}: acc={acc:.4f} cer={c:.4f} kafes={used:3d}")
            print(f"  {name:20s} | " + " | ".join(parts))
        print()


if __name__ == "__main__":
    main()
