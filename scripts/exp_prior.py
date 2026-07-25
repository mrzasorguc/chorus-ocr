"""Rank lattice candidates with a general-purpose language prior.

The beam already contains the correct string far more often than the column
vote picks it: 0.8600 reachable against 0.7667 chosen on FUNSD Maximum. Ranking
those candidates by agreement with the engines (MBR) made things worse, because
agreement pulls the answer back toward what the engines already said, which is
exactly what the lattice was built to escape.

What separates a real reading from a plausible-looking blend is not engine
agreement but whether the result is a word. This ranks candidates by column
support plus the Zipf frequency of their tokens, taken from `wordfreq`, a
general English and Turkish frequency list. The prior never sees the benchmark:
no ground truth, no dataset vocabulary, nothing conditioned on which corpus a
crop came from. Tokens that are not words -- numbers, punctuation, codes -- are
scored neutrally so the prior cannot punish a correct form ID.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from exp_charfuse import load_dump  # noqa: E402
from exp_beam import beam_candidates  # noqa: E402
from chorus import consensus, lang  # noqa: E402

from wordfreq import zipf_frequency  # noqa: E402

_CACHE = {}


def token_prior(token):
    """Zipf score for a token; neutral for anything that is not a word."""
    key = token.lower()
    if key in _CACHE:
        return _CACHE[key]
    letters = [ch for ch in key if ch.isalpha()]
    if not letters or len(letters) < len(key) / 2:
        value = None  # numbers, punctuation, codes: no opinion
    else:
        word = "".join(ch for ch in key if ch.isalpha() or ch in "'-")
        value = max(zipf_frequency(word, "en"), zipf_frequency(word, "tr"))
    _CACHE[key] = value
    return value


def text_prior(text):
    scores = [s for s in (token_prior(t) for t in text.split()) if s is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def polish(text, route):
    return norm(lang.polish(text, mode=route)).lower()


def pick(cands, lam):
    best, best_val = None, None
    for text, support in cands:
        val = support + lam * text_prior(text)
        if best_val is None or val > best_val or (val == best_val and text < best):
            best, best_val = text, val
    return best


def evaluate(records, mode, lams):
    cfg = MODES[mode]
    keys = ["current"] + [f"lam={l:g}" for l in lams]
    stats = {k: [0, 0.0] for k in keys}
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

        stats["current"][0] += int(cur == gl)
        stats["current"][1] += cer(gl, cur)

        if route == "scene":
            for l in lams:
                stats[f"lam={l:g}"][0] += int(cur == gl)
                stats[f"lam={l:g}"][1] += cer(gl, cur)
            continue

        cands = beam_candidates(hyps)
        for l in lams:
            pred = polish(pick(cands, l), route)
            stats[f"lam={l:g}"][0] += int(pred == gl)
            stats[f"lam={l:g}"][1] += cer(gl, pred)

    return {k: (v[0] / n, v[1] / n) for k, v in stats.items()}, n, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--datasets", default="funsd,iiit5k")
    args = ap.parse_args()

    lams = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    for dataset in args.datasets.split(","):
        records = load_dump(dataset, args.n, args.skip)
        print(f"===== {dataset.upper()}  (tuning split)")
        for mode in ("standard", "max", "maxq"):
            res, n, keys = evaluate(records, mode, lams)
            print(f"  {mode:9s} n={n}")
            for key in keys:
                acc, c = res[key]
                print(f"     {key:10s} acc={acc:.4f} cer={c:.4f}")
        print()


if __name__ == "__main__":
    main()
