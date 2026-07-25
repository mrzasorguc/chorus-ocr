"""Hypothesis fusion using MBR selection, ROVER voting, and champion selection."""
from difflib import SequenceMatcher
from collections import defaultdict

import os

# Readings at or below this token count are selected as-is instead of being
# spliced together token by token. Column-wise voting needs enough context to
# align reliably; on short crops it tends to invent readings no engine gave.
SPLICE_MIN_TOKENS = int(os.environ.get("CHORUS_SPLICE_MIN_TOKENS", "4"))

# Whether to re-pick punctuation among engines that agree on the letters.
# Exposed so fusion choices can be A/B tested on a tuning split.
PUNCT_PREFERENCE = os.environ.get("CHORUS_PUNCT_PREFERENCE", "1") != "0"

def _sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _norm(s):
    return "".join(ch.lower() for ch in s if ch.isalnum())

def _engine(src):
    return (src or "").split(":", 1)[0]

def _pick_champion(hyps):
    if not hyps:
        return None
    by_eng = {}
    for h in hyps:
        e = _engine(h.get("src", ""))
        prev = by_eng.get(e)
        if prev is None or h["weight"] > prev["weight"]:
            by_eng[e] = h

    groups = defaultdict(list)
    for e, h in by_eng.items():
        groups[_norm(h["text"])].append((e, h))

    best_h, best_score = None, -1.0
    for key, items in groups.items():
        if not key:
            continue
        score = sum(h["weight"] for _, h in items)
        if any(e == "got" for e, _ in items):
            score *= 1.4
        if len(items) >= 2:
            score *= 1.3
        h0 = max(items, key=lambda x: x[1]["weight"])[1]
        if score > best_score:
            best_score, best_h = score, h0

    if best_h is None:
        best_h = max(hyps, key=lambda h: h["weight"])

    got = by_eng.get("got")
    if got and _norm(got["text"]):
        gkey = _norm(got["text"])
        agree = sum(1 for e, h in by_eng.items() if e != "got" and _norm(h["text"]) == gkey)
        rivals = [h for e, h in by_eng.items() if e != "got" and _norm(h["text"]) and _norm(h["text"]) != gkey]
        if not rivals:
            best_h = got
        else:
            rival = max(rivals, key=lambda h: h["weight"])
            rsup = sum(1 for e, h in by_eng.items() if _norm(h["text"]) == _norm(rival["text"]))
            if agree >= 1 or rsup < 2 or got["weight"] >= rival["weight"] * 0.9:
                best_h = got

    # Prefer punctuation supported by multiple engines when the normalized
    # alphanumeric reading agrees.
    
    bkey = _norm(best_h["text"])
    same_reading = [(e, h) for e, h in by_eng.items() if _norm(h["text"]) == bkey]
    if len(same_reading) >= 2:
        punct_groups = defaultdict(list)
        for e, h in same_reading:
            sig = "".join(ch for ch in h["text"] if not ch.isalnum() and not ch.isspace())
            punct_groups[sig].append((e, h))
        win = max(punct_groups.values(), key=lambda xs: (len(xs), sum(h["weight"] for _, h in xs)))
        if len(win) >= 2:
            best_h = max(win, key=lambda x: x[1]["weight"])[1]

    words = [(w, round(min(0.99, best_h.get("conf", 0.9)), 4)) for w in best_h["text"].split()]
    conf = sum(c for _, c in words) / len(words) if words else float(best_h.get("conf", 0.0))
    return {"text": best_h["text"], "confidence": round(conf, 4), "words": words, "mode": "champion"}

def _rover(hyps):
    toks = [h["text"].split() for h in hyps]
    n = len(hyps)
    best_i, best_s = 0, -1.0
    for i in range(n):
        s = sum(hyps[j]["weight"] * _sim(toks[i], toks[j]) for j in range(n) if j != i)
        if _engine(hyps[i].get("src", "")) == "got":
            s *= 1.08
        if s > best_s:
            best_s, best_i = s, i
    pivot = toks[best_i]
    cols = [defaultdict(float) for _ in pivot]
    ins = defaultdict(lambda: defaultdict(float))
    total_w = 0.0
    for h, t in zip(hyps, toks):
        w = h["weight"]
        total_w += w
        sm = SequenceMatcher(None, pivot, t)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    cols[i1 + k][t[j1 + k]] += w
            elif tag == "replace":
                la, lb = i2 - i1, j2 - j1
                for k in range(max(la, lb)):
                    if k < la and k < lb:
                        cols[i1 + k][t[j1 + k]] += w
                    elif k < la:
                        cols[i1 + k][""] += w
                    else:
                        ins[i2 - 1][" ".join(t[j1 + k:j2])] += w
                        break
            elif tag == "delete":
                for k in range(i2 - i1):
                    cols[i1 + k][""] += w
            elif tag == "insert":
                ins[i1 - 1][" ".join(t[j1:j2])] += w
    out, words = [], []
    def flush_ins(pos):
        if pos in ins:
            cand, wt = max(ins[pos].items(), key=lambda kv: kv[1])
            if wt > 0.5 * total_w and cand:
                for wd in cand.split():
                    out.append(wd)
                    words.append((wd, round(wt / total_w, 4)))
    flush_ins(-1)
    for idx, c in enumerate(cols):
        if c:
            cand, wt = max(c.items(), key=lambda kv: kv[1])
            if cand:
                out.append(cand)
                words.append((cand, round(wt / total_w, 4)))
        flush_ins(idx)
    conf = sum(c for _, c in words) / len(words) if words else 0.0
    return {"text": " ".join(out), "confidence": round(conf, 4), "words": words, "mode": "rover"}

def _mbr(hyps):
    """Minimum Bayes Risk selection over the hypotheses actually produced.

    ROVER builds a new token sequence by voting column by column, which can
    emit a reading no engine proposed. On short crops that invented reading is
    usually worse than every input. MBR instead keeps engine output intact and
    returns whichever reading carries the lowest weighted disagreement, so
    engines that agree reinforce one another.
    """
    best, best_risk = None, None
    for cand in hyps:
        ckey = _norm(cand["text"])
        risk = 0.0
        for other in hyps:
            if other is cand:
                continue
            # All-or-nothing risk: pick the reading the engines most agree on.
            # A character-distance term was measured on the tuning split and
            # was never better, so the simpler measure is kept.
            if _norm(other["text"]) != ckey:
                risk += other["weight"]
        # Stable tie-break toward the more reliable engine.
        risk -= 1e-6 * cand["weight"]
        if best_risk is None or risk < best_risk:
            best, best_risk = cand, risk
    if best is None:
        return {"text": "", "confidence": 0.0, "words": []}

    if PUNCT_PREFERENCE:
        # Among engines that read the same characters, prefer the punctuation
        # spelling that more engines agree on.
        bkey = _norm(best["text"])
        same = [h for h in hyps if _norm(h["text"]) == bkey]
        if len(same) >= 2:
            punct = defaultdict(list)
            for h in same:
                sig = "".join(ch for ch in h["text"] if not ch.isalnum() and not ch.isspace())
                punct[sig].append(h)
            win = max(punct.values(), key=lambda xs: (len(xs), sum(h["weight"] for h in xs)))
            best = max(win, key=lambda h: h["weight"])

    words = [(w, round(min(0.99, best.get("conf", 0.9)), 4)) for w in best["text"].split()]
    conf = sum(c for _, c in words) / len(words) if words else float(best.get("conf", 0.0))
    return {"text": best["text"], "confidence": round(conf, 4), "words": words, "mode": "mbr"}

def fuse(hyps, mode="auto"):
    """mode: auto|scene|document"""
    hyps = [h for h in hyps if h.get("text", "").strip()]
    if not hyps:
        return {"text": "", "confidence": 0.0, "words": []}

    max_tok = max(len(h["text"].split()) for h in hyps)
    # Short readings are selected, never spliced. Measured on a tuning split
    # that is disjoint from every reported benchmark split.
    if max_tok <= SPLICE_MIN_TOKENS:
        return _mbr(hyps)

    if mode == "scene":
        return _pick_champion(hyps)
    if mode == "document":
        return _rover(hyps)

    avg_tok = sum(len(h["text"].split()) for h in hyps) / len(hyps)
    if avg_tok <= 2.5 and max_tok <= 3:
        return _pick_champion(hyps)
    return _rover(hyps)
