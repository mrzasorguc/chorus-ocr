"""Character-lattice fusion: can we build a reading no engine produced?

Every selector shipped so far PICKS one whole hypothesis, so its accuracy is
capped by the pick-one oracle (about 0.90 on the FUNSD quality profile). That
ceiling is an artifact of the architecture, not of the engines: when one engine
reads 'Elcvcn' and another reads 'Eleven ='  the correct string may exist only
as a mixture of the two.

This script aligns all hypotheses against the heaviest one, turns them into a
column lattice, and votes per column. The output can therefore be a string that
appears in no hypothesis at all.

Two things are measured:

  vote     what a weighted per-column vote actually produces today
  ceiling  whether the ground truth is reachable through the lattice at all,
           i.e. the ceiling of ANY per-column decision rule

If the ceiling sits far above the pick-one oracle, the architecture is worth
changing. Tuning split only (skip=1000); the test split is never loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from replay_fusion import MODES, cer, norm, weighted  # noqa: E402
from chorus import consensus, lang  # noqa: E402


def load_dump(dataset, n, skip=1000):
    path = os.path.join(ROOT, "out", "research_20260724",
                        f"hyps_{dataset}_skip{skip}_n{n}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["records"]


# ------------------------------------------------------------------ lattice

def columns_for(pivot, texts):
    """Align every text to the pivot and return one option list per column.

    The column layout is  ins[0], piv[0], ins[1], piv[1], ... ins[L].
    'ins' columns absorb characters a hypothesis has but the pivot lacks;
    'piv' columns hold the character each hypothesis aligns to that position.
    Every hypothesis contributes exactly one option to every column, so the
    columns stay comparable and a vote is well defined.
    """
    L = len(pivot)
    per_text = []
    for text in texts:
        ins = [""] * (L + 1)
        piv = [""] * L
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, pivot, text).get_opcodes():
            if tag == "equal":
                for k in range(i1, i2):
                    piv[k] = text[j1 + (k - i1)]
            elif tag == "replace":
                span = j2 - j1
                for k in range(i1, i2):
                    off = k - i1
                    piv[k] = text[j1 + off] if off < span else ""
                if span > (i2 - i1):
                    ins[i2] += text[j1 + (i2 - i1):j2]
            elif tag == "delete":
                for k in range(i1, i2):
                    piv[k] = ""
            elif tag == "insert":
                ins[i1] += text[j1:j2]
        per_text.append((ins, piv))

    cols = []
    for i in range(L + 1):
        cols.append([opts[0][i] for opts in per_text])
        if i < L:
            cols.append([opts[1][i] for opts in per_text])
    return cols


def lattice(hyps):
    """Return (columns, weights) where columns[c][h] is hypothesis h's option."""
    pivot_h = max(hyps, key=lambda h: h["weight"])
    texts = [h["text"] for h in hyps]
    weights = [h["weight"] for h in hyps]
    return columns_for(pivot_h["text"], texts), weights


def vote(cols, weights):
    """Weighted per-column vote. Can emit a string no hypothesis contained."""
    out = []
    for col in cols:
        tally = {}
        for opt, w in zip(col, weights):
            tally[opt] = tally.get(opt, 0.0) + w
        # Deterministic: weight first, then the longer option, then codepoint.
        best = max(tally.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))
        out.append(best[0])
    return "".join(out)


def reachable(cols, target):
    """Is `target` expressible by choosing one observed option per column?"""
    states = {0}
    n = len(target)
    for col in cols:
        nxt = set()
        opts = set(col)
        for p in states:
            for opt in opts:
                if not opt:
                    nxt.add(p)
                elif target.startswith(opt, p):
                    nxt.add(p + len(opt))
        if not nxt:
            return False
        states = nxt
        if max(states) > n:
            states = {p for p in states if p <= n}
            if not states:
                return False
    return n in states


# ------------------------------------------------------------------ report

def evaluate(records, mode):
    cfg = MODES[mode]
    stats = {k: 0 for k in ("current", "pick_oracle", "vote", "ceiling", "n")}
    cer_sum = {"current": 0.0, "vote": 0.0}
    novel = 0          # vote produced a string no hypothesis contained
    novel_right = 0    # ... and it was correct

    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        if not hyps:
            stats["n"] += 1
            cer_sum["current"] += 1.0
            cer_sum["vote"] += 1.0
            continue
        route = rec["route"]
        gl = norm(rec["gt"]).lower()
        stats["n"] += 1

        fused = consensus.fuse(hyps, mode=route)
        cur = norm(lang.polish(fused["text"], mode=route)).lower()
        stats["current"] += int(cur == gl)
        cer_sum["current"] += cer(gl, cur)

        texts = [norm(lang.polish(h["text"], mode=route)).lower() for h in hyps]
        stats["pick_oracle"] += int(gl in texts)

        low = [dict(h, text=t) for h, t in zip(hyps, texts)]
        cols, weights = lattice(low)
        voted = norm(vote(cols, weights)).lower()
        stats["vote"] += int(voted == gl)
        cer_sum["vote"] += cer(gl, voted)
        if voted not in texts:
            novel += 1
            novel_right += int(voted == gl)

        stats["ceiling"] += int(reachable(cols, gl))

    n = stats["n"] or 1
    return {
        "n": stats["n"],
        "current": stats["current"] / n,
        "pick_oracle": stats["pick_oracle"] / n,
        "vote": stats["vote"] / n,
        "ceiling": stats["ceiling"] / n,
        "cer_current": cer_sum["current"] / n,
        "cer_vote": cer_sum["vote"] / n,
        "novel": novel,
        "novel_right": novel_right,
    }


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
            r = evaluate(records, mode)
            print(f"  {mode:9s} mevcut={r['current']:.4f} (cer {r['cer_current']:.4f})"
                  f"   secim-oracle={r['pick_oracle']:.4f}")
            print(f"            kafes-oy={r['vote']:.4f} (cer {r['cer_vote']:.4f})"
                  f"   KAFES TAVANI={r['ceiling']:.4f}")
            print(f"            yeni dizge uretildi: {r['novel']}  bunlardan dogru: {r['novel_right']}")
        print()


if __name__ == "__main__":
    main()
