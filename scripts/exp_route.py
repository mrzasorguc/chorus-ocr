"""Route-conditional surface fusion, measured on the tuning (dev) split.

Findings that led here:

* Counting distinct engines instead of hypotheses never hurt any mode and
  clearly helped the quality profile, where one engine can field seven
  correlated augmentations against another engine's two.
* Spreading each engine's vote across near-miss readings (soft similarity)
  helped document crops and hurt scene crops. Document words carry character
  confusions that partial credit can resolve; scene crops are short and
  decisive, so blurring candidates together only adds noise.

The route is decided from image statistics in chorus.pipeline._classify, not
from the dataset a sample came from, so conditioning on it is a property of
the input rather than a benchmark-specific switch.
"""
from __future__ import annotations

from collections import defaultdict

from replay_fusion import load, replay, score  # noqa: E402
from exp_fusion import _engine_of, _engine_vote_scores  # noqa: E402
from exp_surface import _surface_by_engine_agreement  # noqa: E402
from chorus import consensus  # noqa: E402

NORM = consensus._norm

def fuse_route(hyps, mode="auto"):
    hyps = [h for h in hyps if h.get("text", "").strip()]
    if not hyps:
        return {"text": "", "confidence": 0.0, "words": []}
    max_tok = max(len(h["text"].split()) for h in hyps)
    if max_tok > consensus.SPLICE_MIN_TOKENS:
        return consensus.fuse(hyps, mode=mode)

    base, _ = _engine_vote_scores(hyps)
    if not base:
        return consensus.fuse(hyps, mode=mode)

    if mode == "document":
        scores = defaultdict(float)
        for cand in base:
            for other, w in base.items():
                scores[cand] += w * consensus._sim(cand, other)
    else:
        scores = base

    key = max(scores.items(), key=lambda kv: kv[1])[0]
    cands = [h for h in hyps if NORM(h["text"]) == key]
    text = _surface_by_engine_agreement(cands)
    conf = max(h.get("conf", 0.9) for h in cands)
    return {"text": text, "confidence": round(float(conf), 4),
            "words": [(w, round(float(conf), 4)) for w in text.split()],
            "mode": "route_surface"}

def main():
    total_base = total_new = 0
    for dataset in ("funsd", "iiit5k"):
        records = load(dataset)
        print(f"===== {dataset.upper()} (dev, n={len(records)})")
        for mode in ("standard", "max", "maxq"):
            b = score(replay(records, mode))
            n = score(replay(records, mode, fuse_fn=fuse_route))
            da = n["word_acc"] - b["word_acc"]
            dc = n["avg_cer"] - b["avg_cer"]
            flag = "OK" if da >= 0 and dc <= 0 else ("GERILEME" if da < 0 else "karisik")
            total_base += b["word_acc"]
            total_new += n["word_acc"]
            print(f"  {mode:9s} acc {b['word_acc']:.4f} -> {n['word_acc']:.4f} ({da:+.4f})   "
                  f"cer {b['avg_cer']:.4f} -> {n['avg_cer']:.4f} ({dc:+.4f})   {flag}")
        print()
    print(f"6 olcumun acc toplami: {total_base:.4f} -> {total_new:.4f} ({total_new-total_base:+.4f})")

if __name__ == "__main__":
    main()
