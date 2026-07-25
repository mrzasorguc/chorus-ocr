"""Close the gap between the column vote and what the lattice can reach.

The column vote scores 0.7733 on the tuning split while the lattice provably
contains the right answer 0.9333 of the time. The answer is in there; the rule
cannot find it. The reason is structural: voting decides each column on its own,
but columns are not independent. Deleting a character in one column only makes
sense given what the neighbouring columns decided.

So instead of one greedy pass, keep a beam of whole candidate strings, then pick
among them with minimum Bayes risk: the candidate with the smallest expected
character distance to the engines' readings, weighted by how much each reading
is trusted. MBR already decides short readings in this library; the new part is
that it now ranks strings the lattice invented, not only strings an engine
produced.

No parameter here is fitted to a benchmark. Beam width and branching are
capacity limits, and the MBR objective has no free constants.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from exp_charfuse import load_dump  # noqa: E402
from chorus import consensus, lang  # noqa: E402


def column_scores(col, hyps):
    """Per column: one ballot per engine, spread over that engine's options."""
    by_engine = {}
    surfaces = defaultdict(lambda: defaultdict(float))
    for opt, h in zip(col, hyps):
        eng = consensus._engine(h.get("src", ""))
        key = opt.lower()
        slot = by_engine.setdefault(eng, {"opts": defaultdict(float), "ballot": 0.0})
        slot["opts"][key] += h["weight"]
        if h["weight"] > slot["ballot"]:
            slot["ballot"] = h["weight"]
        surfaces[key][opt] += h["weight"]

    tally = defaultdict(float)
    for slot in by_engine.values():
        total = sum(slot["opts"].values()) or 1.0
        for key, weight in slot["opts"].items():
            tally[key] += slot["ballot"] * (weight / total)

    out = []
    for key, sc in tally.items():
        form = max(surfaces[key].items(), key=lambda kv: (kv[1], kv[0]))[0]
        out.append((form, sc))
    out.sort(key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return out


def columns_of(hyps):
    pivot = max(hyps, key=lambda h: h["weight"])["text"]
    return consensus._align_columns(pivot, [h["text"] for h in hyps])


def greedy(hyps):
    cols = columns_of(hyps)
    return " ".join("".join(column_scores(c, hyps)[0][0] for c in cols).split())


def beam_candidates(hyps, width=12, branch=3):
    """Whole-string candidates, kept by cumulative column support."""
    cols = columns_of(hyps)
    beams = [("", 0.0)]
    for col in cols:
        opts = column_scores(col, hyps)[:branch]
        if not opts:
            continue
        nxt = {}
        for text, sc in beams:
            for form, osc in opts:
                cand = text + form
                val = sc + osc
                if cand not in nxt or val > nxt[cand]:
                    nxt[cand] = val
        beams = sorted(nxt.items(), key=lambda kv: -kv[1])[:width]
    return [(" ".join(t.split()), s) for t, s in beams]


def mbr_pick(cands, hyps):
    """Smallest expected character distance to the engines' readings."""
    best, best_val = None, None
    for text, _ in cands:
        val = 0.0
        for h in hyps:
            val += h["weight"] * SequenceMatcher(None, text, h["text"]).ratio()
        if best_val is None or val > best_val or (val == best_val and text < best):
            best, best_val = text, val
    return best


def polish(text, route):
    return norm(lang.polish(text, mode=route)).lower()


def evaluate(records, mode):
    cfg = MODES[mode]
    stats = {k: [0, 0.0] for k in ("current", "greedy", "beam_mbr", "beam_oracle")}
    n = 0
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        if not hyps:
            continue
        n += 1
        gl = norm(rec["gt"]).lower()
        route = rec["route"]

        prev = consensus.LATTICE_ENABLED
        consensus.LATTICE_ENABLED = True
        cur = polish(consensus.fuse(hyps, mode=route)["text"], route)
        consensus.LATTICE_ENABLED = prev

        if route == "scene":
            for key in stats:
                stats[key][0] += int(cur == gl)
                stats[key][1] += cer(gl, cur)
            continue

        cands = beam_candidates(hyps)
        preds = {
            "current": cur,
            "greedy": polish(greedy(hyps), route),
            "beam_mbr": polish(mbr_pick(cands, hyps), route),
        }
        for key, pred in preds.items():
            stats[key][0] += int(pred == gl)
            stats[key][1] += cer(gl, pred)

        polished = [polish(t, route) for t, _ in cands]
        if gl in polished:
            stats["beam_oracle"][0] += 1
        else:
            stats["beam_oracle"][1] += min(cer(gl, p) for p in polished)

    return {k: (v[0] / n, v[1] / n) for k, v in stats.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    for dataset in args.datasets.split(","):
        records = load_dump(dataset, args.n, args.skip)
        print(f"===== {dataset.upper()}  (tuning split)")
        for mode in ("standard", "max", "maxq"):
            res, n = evaluate(records, mode)
            print(f"  {mode:9s} n={n}")
            for key in ("current", "greedy", "beam_mbr", "beam_oracle"):
                acc, c = res[key]
                print(f"     {key:12s} acc={acc:.4f} cer={c:.4f}")
        print()


if __name__ == "__main__":
    main()
