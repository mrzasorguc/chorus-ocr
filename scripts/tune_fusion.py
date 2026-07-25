"""Replay cached engine hypotheses to compare fusion strategies offline.

The cache produced by dump_hyps.py holds raw engine output for a tuning split
that is disjoint from the test split. Everything here reads that cache only, so
no strategy is ever selected by looking at test-split scores.

Strategies compared
-------------------
baseline   Current shipped consensus.fuse (word-level ROVER / champion).
mbr_cer    Minimum Bayes Risk selection under character edit distance.
mbr_word   Minimum Bayes Risk selection under exact-match loss.
mbr_hybrid MBR under a blended character/word risk.

MBR only ever returns a string some engine actually produced. Word-level ROVER
can splice tokens from different engines into a reading nobody proposed, which
is useful on long documents but risky on short crops.
"""

import os, sys, json, argparse
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chorus import consensus, lang
from chorus.pipeline import MODE_REL


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def norm(s):
    return " ".join(str(s).split())


def _key(s):
    return "".join(ch.lower() for ch in s if ch.isalnum())


INTERACTIVE_VARIANTS = ["orig"]


def build_hyps(record, engines, rel_map, variant_mode, got_orig_bonus=1.12):
    """Rebuild the weighted hypothesis list exactly as pipeline.read would."""
    route = record["route"]
    hyps = []
    for h in record["hyps"]:
        eng = h["eng"]
        if eng not in engines:
            continue
        if variant_mode == "interactive" and h["variant"] not in INTERACTIVE_VARIANTS:
            continue
        if not h["text"].strip():
            continue
        rel = rel_map[route].get(eng, 0.8)
        q = h["quality"]
        q2 = q * (got_orig_bonus if eng == "got" and h["variant"] == "orig" else 1.0)
        hyps.append({"text": h["text"], "conf": h["conf"], "eng": eng,
                     "weight": rel * q2 * (0.5 + 0.5 * h["conf"]),
                     "src": f"{eng}:{h['variant']}"})
    return hyps, route


# ---------------------------------------------------------------- strategies

def baseline_fuse(hyps, route):
    return consensus.fuse(hyps, mode=route)


def _mbr(hyps, risk):
    """Pick the candidate with the lowest weighted risk against all others.

    Engines that agree reinforce each other, so a reading supported by several
    engines wins even when a single confident engine disagrees.
    """
    best, best_risk = None, None
    for cand in hyps:
        total = 0.0
        for other in hyps:
            if other is cand:
                continue
            total += other["weight"] * risk(cand["text"], other["text"])
        total -= 1e-6 * cand["weight"]  # stable tie-break toward heavier engines
        if best_risk is None or total < best_risk:
            best, best_risk = cand, total
    return best


def _risk_cer(a, b):
    return cer(_key(b), _key(a))


def _risk_word(a, b):
    return 0.0 if _key(a) == _key(b) else 1.0


def _risk_hybrid(a, b):
    return 0.5 * _risk_cer(a, b) + 0.5 * _risk_word(a, b)


def _wrap(best):
    if best is None:
        return {"text": "", "confidence": 0.0, "words": []}
    words = [(w, round(min(0.99, best.get("conf", 0.9)), 4)) for w in best["text"].split()]
    conf = sum(c for _, c in words) / len(words) if words else float(best.get("conf", 0.0))
    return {"text": best["text"], "confidence": round(conf, 4), "words": words, "mode": "mbr"}


def mbr_cer(hyps, route):
    return _wrap(_mbr(hyps, _risk_cer))


def mbr_word(hyps, route):
    return _wrap(_mbr(hyps, _risk_word))


def mbr_hybrid(hyps, route):
    return _wrap(_mbr(hyps, _risk_hybrid))


def mbr_or_rover(hyps, route):
    """MBR on short readings, ROVER splicing on long ones."""
    max_tok = max((len(h["text"].split()) for h in hyps), default=0)
    if max_tok <= 4:
        return _wrap(_mbr(hyps, _risk_hybrid))
    return consensus.fuse(hyps, mode=route)


STRATEGIES = {
    "baseline": baseline_fuse,
    "mbr_cer": mbr_cer,
    "mbr_word": mbr_word,
    "mbr_hybrid": mbr_hybrid,
    "mbr_or_rover": mbr_or_rover,
}


# ---------------------------------------------------------------- evaluation

def evaluate(records, engines, rel_map, variant_mode, fuse_fn, polish=True):
    exact, cer_sum = 0, 0.0
    for rec in records:
        hyps, route = build_hyps(rec, engines, rel_map, variant_mode)
        text = ""
        if hyps:
            text = fuse_fn(hyps, route).get("text", "")
            if polish:
                text = lang.polish(text, mode=route)
        gt = norm(rec["gt"]).lower()
        hyp = norm(text).lower()
        exact += int(hyp == gt)
        cer_sum += cer(gt, hyp)
    n = max(1, len(records))
    return {"word_acc": round(exact / n, 4), "avg_cer": round(cer_sum / n, 4), "n": len(records)}


def single_engine(records, eng, polish=True):
    exact, cer_sum = 0, 0.0
    for rec in records:
        text = ""
        for h in rec["hyps"]:
            if h["eng"] == eng and h["variant"] == "orig":
                text = h["text"]
                break
        if polish:
            text = lang.polish(text, mode=rec["route"])
        gt = norm(rec["gt"]).lower()
        hyp = norm(text).lower()
        exact += int(hyp == gt)
        cer_sum += cer(gt, hyp)
    n = max(1, len(records))
    return {"word_acc": round(exact / n, 4), "avg_cer": round(cer_sum / n, 4), "n": len(records)}


def fmt(name, res):
    return f"  {name:22s} acc={res['word_acc']:.4f}  cer={res['avg_cer']:.4f}  n={res['n']}"


def search_weights(records, engines, variant_mode, fuse_fn, route, grid=None):
    """Coordinate search for per-engine reliability on the tuning split.

    Shipped weights were hand-picked and rank Tesseract lowest on documents even
    though it is the strongest document engine here. Searching on a split that
    is disjoint from the test data keeps the fix honest.
    """
    grid = grid or [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    engs = sorted(engines)
    current = {e: 1.0 for e in engs}

    def score(weights):
        rel = {"scene": dict(weights), "document": dict(weights)}
        r = evaluate(records, engines, rel, variant_mode, fuse_fn)
        return (r["word_acc"], -r["avg_cer"]), r

    best_key, best_res = score(current)
    improved = True
    rounds = 0
    while improved and rounds < 4:
        improved = False
        rounds += 1
        for e in engs:
            for v in grid:
                if v == current[e]:
                    continue
                trial = dict(current)
                trial[e] = v
                key, res = score(trial)
                if key > best_key:
                    best_key, best_res, current = key, res, trial
                    improved = True
    return current, best_res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--engines", default="easyocr,paddle,tesseract")
    ap.add_argument("--variant-mode", default="interactive",
                    choices=["interactive", "quality"])
    ap.add_argument("--search", action="store_true",
                    help="Search reliability weights on the tuning split.")
    ap.add_argument("--strategy", default="mbr_word", choices=sorted(STRATEGIES))
    ap.add_argument("--holdout", action="store_true",
                    help="Two-fold check that tuned weights generalize.")
    a = ap.parse_args()

    engines = set(a.engines.split(","))
    for path in a.caches:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        records = data["records"]
        print(f"\n=== {os.path.basename(path)}  n={len(records)}  "
              f"engines={sorted(engines)}  mode={a.variant_mode} ===")
        for eng in sorted(engines):
            print(fmt(f"single:{eng}", single_engine(records, eng)))
        for name, fn in STRATEGIES.items():
            print(fmt(name, evaluate(records, engines, MODE_REL, a.variant_mode, fn)))
        if a.search:
            route = records[0]["route"] if records else "document"
            weights, res = search_weights(records, engines, a.variant_mode,
                                          STRATEGIES[a.strategy], route)
            print(f"  -- tuned weights ({a.strategy}) --")
            print(f"     {weights}")
            print(fmt(f"tuned:{a.strategy}", res))

        if a.holdout:
            # Guard against fitting noise: tune on one half, score the other.
            half = len(records) // 2
            folds = [(records[:half], records[half:]), (records[half:], records[:half])]
            for i, (fit, held) in enumerate(folds, 1):
                w, _ = search_weights(fit, engines, a.variant_mode,
                                      STRATEGIES[a.strategy], "document")
                rel = {"scene": dict(w), "document": dict(w)}
                got = evaluate(held, engines, rel, a.variant_mode, STRATEGIES[a.strategy])
                base = evaluate(held, engines, MODE_REL, a.variant_mode, baseline_fuse)
                print(f"  fold{i} fit={w}")
                print(fmt(f"  fold{i} tuned(held)", got))
                print(fmt(f"  fold{i} baseline(held)", base))
