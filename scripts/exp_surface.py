"""Surface-form selection experiments on the tuning (dev) split.

Diagnosis: on the missed cases, fusion usually already agrees on WHAT the text
says. It loses on HOW the text is spelled out -- a stray period, a dropped
hyphen, a space inserted before punctuation. The normalized reading is right
and still scores as a miss.

Two causes, both real quality bugs rather than benchmark artifacts:

1. Correlated variants. One engine running seven augmentations can outvote
   three engines that agree, because every variant votes on its own.
2. Surface tie-breaks by summed weight. Weight sums inherit the same
   correlation problem, so the surface form of the loudest engine wins even
   when every other engine spells it differently.

Fix under test: count DISTINCT ENGINES, not hypotheses, at both stages.
"""
from __future__ import annotations

from collections import defaultdict

from replay_fusion import load, replay, score  # noqa: E402
from exp_fusion import STRATEGIES, _engine_of, _engine_vote_scores  # noqa: E402
from chorus import consensus  # noqa: E402

NORM = consensus._norm

def _surface_by_engine_agreement(cands):
    """Pick the exact spelling that the most distinct engines produced.

    Falls back to total weight, then to the shorter string. Shortness is a
    deliberate tie-break: spurious trailing punctuation is a far more common
    OCR artifact than a genuinely omitted mark.
    """
    groups = defaultdict(list)
    for h in cands:
        groups[h["text"]].append(h)

    def key(item):
        text, items = item
        engines = {_engine_of(h) for h in items}
        return (len(engines), sum(h["weight"] for h in items), -len(text))

    return max(groups.items(), key=key)[0]

def _build(select_surface, soft=False):
    def fuse(hyps, mode="auto"):
        hyps = [h for h in hyps if h.get("text", "").strip()]
        if not hyps:
            return {"text": "", "confidence": 0.0, "words": []}
        max_tok = max(len(h["text"].split()) for h in hyps)
        if max_tok > consensus.SPLICE_MIN_TOKENS:
            return consensus.fuse(hyps, mode=mode)

        base, _ = _engine_vote_scores(hyps)
        if not base:
            return consensus.fuse(hyps, mode=mode)

        if soft:
            scores = defaultdict(float)
            for cand in base:
                for other, w in base.items():
                    scores[cand] += w * consensus._sim(cand, other)
        else:
            scores = base

        key = max(scores.items(), key=lambda kv: kv[1])[0]
        cands = [h for h in hyps if NORM(h["text"]) == key]
        text = select_surface(cands)
        return {"text": text, "confidence": 0.9,
                "words": [(w, 0.9) for w in text.split()], "mode": "surface"}
    return fuse

EXTRA = {
    "surface_eng": _build(_surface_by_engine_agreement),
    "surface_eng_soft": _build(_surface_by_engine_agreement, soft=True),
}

def main():
    allstrats = dict(STRATEGIES)
    allstrats.update(EXTRA)
    for dataset in ("funsd", "iiit5k"):
        records = load(dataset)
        print(f"===== {dataset.upper()} (dev, n={len(records)})")
        for mode in ("standard", "max", "maxq"):
            print(f"  -- {mode}")
            for name, fn in allstrats.items():
                s = score(replay(records, mode, fuse_fn=fn))
                print(f"       {name:18s} acc={s['word_acc']:.4f}  cer={s['avg_cer']:.4f}")
        print()

if __name__ == "__main__":
    main()
