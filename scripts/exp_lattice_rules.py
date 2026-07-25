"""Column decision rules for the character lattice.

exp_charfuse.py showed that aligning hypotheses into a column lattice raises
the reachable ceiling far above the pick-one oracle (FUNSD standard 0.69 ->
0.78, maxq 0.90 -> 0.93), and that even a naive per-column vote already cuts
CER close to half on the document route. The naive vote leaves most of that
new headroom unused, for a reason we already diagnosed once at the whole-string
level: test-time augmentation gives one engine many correlated variants, so a
single engine can outvote three engines that agree.

This script applies that fix per column and sweeps the two remaining knobs:

  raw        every hypothesis votes with its own weight (baseline)
  engvote    one ballot per engine, worth that engine's heaviest weight
  engvote+d  same, with a deletion bias on the empty option

A hybrid is also measured, since scene text is a single word where picking a
whole hypothesis is the right move, while document text is fragmented and
benefits from mixing.

Tuning split only (skip=1000).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from exp_charfuse import lattice, load_dump  # noqa: E402
from chorus import consensus, lang  # noqa: E402


def engine_of(src):
    return src.split(":", 1)[0]


def decide_raw(col, hyps, weights, del_bias=1.0):
    tally = {}
    for opt, w in zip(col, weights):
        tally[opt] = tally.get(opt, 0.0) + w
    if "" in tally:
        tally[""] *= del_bias
    return max(tally.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]


def decide_engvote(col, hyps, weights, del_bias=1.0):
    """One ballot per engine so correlated TTA variants cannot stuff the vote.

    Within an engine the variants still argue among themselves: the engine's
    ballot is split across the options its variants produced, in proportion to
    the weight behind each option.
    """
    by_engine = {}
    for opt, h, w in zip(col, hyps, weights):
        eng = engine_of(h["src"])
        slot = by_engine.setdefault(eng, {"opts": {}, "ballot": 0.0})
        slot["opts"][opt] = slot["opts"].get(opt, 0.0) + w
        slot["ballot"] = max(slot["ballot"], w)
    tally = {}
    for slot in by_engine.values():
        total = sum(slot["opts"].values()) or 1.0
        for opt, w in slot["opts"].items():
            tally[opt] = tally.get(opt, 0.0) + slot["ballot"] * (w / total)
    if "" in tally:
        tally[""] *= del_bias
    return max(tally.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]


RULES = {"raw": decide_raw, "engvote": decide_engvote}


def lattice_text(hyps, rule, del_bias):
    cols, weights = lattice(hyps)
    return "".join(rule(col, hyps, weights, del_bias) for col in cols)


def evaluate(records, mode, rule_name, del_bias, hybrid):
    cfg = MODES[mode]
    rule = RULES[rule_name]
    hit = 0
    cer_sum = 0.0
    n = 0
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        n += 1
        gl = norm(rec["gt"]).lower()
        if not hyps:
            cer_sum += 1.0
            continue
        route = rec["route"]
        if hybrid and route == "scene":
            fused = consensus.fuse(hyps, mode=route)
            pred = norm(lang.polish(fused["text"], mode=route)).lower()
        else:
            low = [dict(h, text=norm(lang.polish(h["text"], mode=route)).lower())
                   for h in hyps]
            pred = norm(lattice_text(low, rule, del_bias)).lower()
        hit += int(pred == gl)
        cer_sum += cer(gl, pred)
    return hit / n, cer_sum / n


def baseline(records, mode):
    cfg = MODES[mode]
    hit = 0
    cer_sum = 0.0
    n = 0
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        n += 1
        gl = norm(rec["gt"]).lower()
        if not hyps:
            cer_sum += 1.0
            continue
        fused = consensus.fuse(hyps, mode=rec["route"])
        pred = norm(lang.polish(fused["text"], mode=rec["route"])).lower()
        hit += int(pred == gl)
        cer_sum += cer(gl, pred)
    return hit / n, cer_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    biases = [0.7, 0.85, 1.0, 1.15]
    for dataset in args.datasets.split(","):
        records = load_dump(dataset, args.n, args.skip)
        print(f"===== {dataset.upper()}  (tuning split, n={len(records)})")
        for mode in ("standard", "max", "maxq"):
            acc, c = baseline(records, mode)
            print(f"  {mode}  MEVCUT acc={acc:.4f} cer={c:.4f}")
            for rule_name in ("raw", "engvote"):
                for hybrid in (False, True):
                    tag = "hibrit" if hybrid else "tam   "
                    line = []
                    for b in biases:
                        a, cc = evaluate(records, mode, rule_name, b, hybrid)
                        line.append(f"d={b:<4} acc={a:.4f} cer={cc:.4f}")
                    print(f"     {rule_name:8s} {tag} | " + " | ".join(line))
        print()


if __name__ == "__main__":
    main()
