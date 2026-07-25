"""Fusion strategy experiments on the tuning (dev) split.

Diagnosis that motivated this file: in the quality profile each engine
contributes one hypothesis per TTA variant, and every hypothesis votes
independently. EasyOCR and PaddleOCR run seven variants each while GOT-OCR
runs two, so the weakest engines cast the most votes. Those votes are also
highly correlated, because they come from the same engine looking at lightly
modified copies of one image. Classic ensemble double counting.

The strategies below test fixes for that. Every number here is measured on the
tuning split (skip=1000), which shares no samples with any reported benchmark.
"""
from __future__ import annotations

import sys
from collections import defaultdict

from replay_fusion import MODES, load, replay, score, weighted  # noqa: E402
from chorus import consensus, lang  # noqa: E402

NORM = consensus._norm

def _engine_of(h):
    return h["src"].split(":", 1)[0]

def _punct_preference(best, hyps):
    """Same tie-break the library uses: agree on letters, vote on punctuation."""
    bkey = NORM(best["text"])
    same = [h for h in hyps if NORM(h["text"]) == bkey]
    if len(same) < 2:
        return best
    groups = defaultdict(list)
    for h in same:
        sig = "".join(c for c in h["text"] if not c.isalnum() and not c.isspace())
        groups[sig].append(h)
    win = max(groups.values(), key=lambda xs: (len(xs), sum(h["weight"] for h in xs)))
    return max(win, key=lambda h: h["weight"])

# ------------------------------------------------------------------ strategies

def fuse_baseline(hyps, mode="auto"):
    return consensus.fuse(hyps, mode=mode)

def _engine_vote_scores(hyps):
    """Each engine gets exactly one unit of vote, split across its variants.

    Prevents an engine from gaining influence just by running more test-time
    augmentations. Intra-engine agreement still matters, because a reading that
    survives several variants of the same engine takes a larger share of that
    engine's single vote.
    """
    per_engine = defaultdict(list)
    for h in hyps:
        per_engine[_engine_of(h)].append(h)

    scores = defaultdict(float)
    best_by_key = {}
    for eng, items in per_engine.items():
        total = sum(h["weight"] for h in items) or 1.0
        # rel is already folded into each weight; recover the engine's scale as
        # the strongest single variant it produced.
        eng_scale = max(h["weight"] for h in items)
        share = defaultdict(float)
        for h in items:
            share[NORM(h["text"])] += h["weight"]
        for key, got in share.items():
            if not key:
                continue
            scores[key] += eng_scale * (got / total)
        for h in items:
            key = NORM(h["text"])
            prev = best_by_key.get(key)
            if prev is None or h["weight"] > prev["weight"]:
                best_by_key[key] = h
    return scores, best_by_key

def fuse_engine_vote(hyps, mode="auto"):
    hyps = [h for h in hyps if h.get("text", "").strip()]
    if not hyps:
        return {"text": "", "confidence": 0.0, "words": []}
    max_tok = max(len(h["text"].split()) for h in hyps)
    if max_tok > consensus.SPLICE_MIN_TOKENS:
        return consensus.fuse(hyps, mode=mode)

    scores, best_by_key = _engine_vote_scores(hyps)
    if not scores:
        return consensus.fuse(hyps, mode=mode)
    key = max(scores.items(), key=lambda kv: kv[1])[0]
    best = _punct_preference(best_by_key[key], hyps)
    return {"text": best["text"], "confidence": round(float(best.get("conf", 0.9)), 4),
            "words": [(w, 0.9) for w in best["text"].split()], "mode": "engine_vote"}

def fuse_engine_vote_soft(hyps, mode="auto"):
    """Engine voting plus partial credit for near-miss readings.

    All-or-nothing agreement wastes the signal in readings that differ by one
    character. Here every engine's vote is spread over candidates in proportion
    to character similarity, so three engines that nearly agree can outvote one
    engine that is confidently alone.
    """
    hyps = [h for h in hyps if h.get("text", "").strip()]
    if not hyps:
        return {"text": "", "confidence": 0.0, "words": []}
    max_tok = max(len(h["text"].split()) for h in hyps)
    if max_tok > consensus.SPLICE_MIN_TOKENS:
        return consensus.fuse(hyps, mode=mode)

    base, best_by_key = _engine_vote_scores(hyps)
    if not base:
        return consensus.fuse(hyps, mode=mode)

    keys = list(base.keys())
    soft = defaultdict(float)
    for cand in keys:
        for other, w in base.items():
            soft[cand] += w * consensus._sim(cand, other)
    key = max(soft.items(), key=lambda kv: kv[1])[0]
    best = _punct_preference(best_by_key[key], hyps)
    return {"text": best["text"], "confidence": round(float(best.get("conf", 0.9)), 4),
            "words": [(w, 0.9) for w in best["text"].split()], "mode": "engine_vote_soft"}

STRATEGIES = {
    "baseline": fuse_baseline,
    "engine_vote": fuse_engine_vote,
    "engine_vote_soft": fuse_engine_vote_soft,
}

# ------------------------------------------------------------------ diagnosis

def variant_counts(records, mode):
    cfg = MODES[mode]
    tally = defaultdict(int)
    for rec in records:
        for h in weighted(rec, cfg["engines"], cfg["profile"]):
            tally[_engine_of(h)] += 1
    return dict(tally)

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for dataset in ("funsd", "iiit5k"):
        if only and only != dataset:
            continue
        records = load(dataset)
        print(f"===== {dataset.upper()} (dev, n={len(records)})")
        print(f"  maxq oy dagilimi (hipotez sayisi): {variant_counts(records, 'maxq')}")
        for mode in ("standard", "max", "maxq"):
            line = [f"  {mode:9s}"]
            for name, fn in STRATEGIES.items():
                s = score(replay(records, mode, fuse_fn=fn))
                line.append(f"{name}={s['word_acc']:.4f}/{s['avg_cer']:.4f}")
            print("  ".join(line))
        print()

if __name__ == "__main__":
    main()
