"""Sweep the two free parameters of the new selector on the tuning split.

The selector introduced in chorus.consensus has exactly two arbitrary choices
left in it:

* ballot: how much a single engine's vote is worth. "max" uses the engine's
  strongest variant, "mean" averages its variants.
* power: how sharply partial credit falls off with character similarity on the
  document route. power=0 collapses to all-or-nothing voting, higher powers
  make near misses count for less.

Both are swept on the tuning split (skip=1000) only. Whatever wins here is
measured afterwards on the untouched test split.
"""
from __future__ import annotations

from collections import defaultdict

from replay_fusion import load, replay, score  # noqa: E402
from chorus import consensus  # noqa: E402

NORM = consensus._norm
ENG = consensus._engine

def build(ballot="max", power=1.0):
    def fuse(hyps, mode="auto"):
        hyps = [h for h in hyps if h.get("text", "").strip()]
        if not hyps:
            return {"text": "", "confidence": 0.0, "words": []}
        if max(len(h["text"].split()) for h in hyps) > consensus.SPLICE_MIN_TOKENS:
            return consensus.fuse(hyps, mode=mode)

        per_engine = defaultdict(list)
        for h in hyps:
            per_engine[ENG(h.get("src", ""))].append(h)

        scores = defaultdict(float)
        for items in per_engine.values():
            total = sum(h["weight"] for h in items) or 1.0
            if ballot == "mean":
                unit = total / len(items)
            else:
                unit = max(h["weight"] for h in items)
            share = defaultdict(float)
            for h in items:
                share[NORM(h["text"])] += h["weight"]
            for key, got in share.items():
                if key:
                    scores[key] += unit * (got / total)

        if not scores:
            return consensus.fuse(hyps, mode=mode)

        if mode == "document" and power > 0:
            soft = defaultdict(float)
            for cand in scores:
                for other, weight in scores.items():
                    soft[cand] += weight * (consensus._sim(cand, other) ** power)
            scores = soft

        key = max(scores.items(), key=lambda kv: kv[1])[0]
        cands = [h for h in hyps if NORM(h["text"]) == key]
        text = consensus._surface_pick(cands)
        return {"text": text, "confidence": 0.9,
                "words": [(w, 0.9) for w in text.split()], "mode": "sweep"}
    return fuse

def main():
    data = {d: load(d) for d in ("funsd", "iiit5k")}
    print(f"{'ballot':7s} {'power':6s}  " + "  ".join(
        f"{d[:6]}/{m[:5]}" for d in data for m in ("standard", "max", "maxq")) + "   toplam")
    for ballot in ("max", "mean"):
        for power in (0.0, 0.5, 1.0, 2.0, 3.0):
            fn = build(ballot, power)
            cells, total = [], 0.0
            for d, recs in data.items():
                for m in ("standard", "max", "maxq"):
                    a = score(replay(recs, m, fuse_fn=fn))["word_acc"]
                    total += a
                    cells.append(f"{a:.4f}     ")
            print(f"{ballot:7s} {power:<6.1f}  " + "  ".join(cells) + f"   {total:.4f}")

if __name__ == "__main__":
    main()
