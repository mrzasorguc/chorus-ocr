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

# Set to "legacy" to restore the older per-hypothesis MBR selection.
FUSION_LEGACY = os.environ.get("CHORUS_FUSION", "engine_vote") == "legacy"

def _engine_vote_scores(hyps):
    """Give every engine one vote, split across the variants it produced.

    Test-time augmentation means an engine can return many readings of the
    same crop. Those readings are not independent opinions, they are one
    engine looking at lightly modified copies of one image. Summing them per
    hypothesis lets an engine buy influence simply by running more variants:
    in the quality profile EasyOCR and PaddleOCR field seven variants each
    while GOT-OCR fields two, so the weakest engines cast the most votes.

    Here each engine's ballot sums to the strength of its own best reading,
    and is divided among its readings in proportion to their weight. Engines
    still differ in influence, but only through reliability and confidence,
    never through variant count. Agreement across variants of one engine is
    still rewarded, because a reading that survives several augmentations
    claims a larger share of that engine's single ballot.
    """
    per_engine = defaultdict(list)
    for h in hyps:
        per_engine[_engine(h.get("src", ""))].append(h)

    scores = defaultdict(float)
    for items in per_engine.values():
        total = sum(h["weight"] for h in items) or 1.0
        ballot = max(h["weight"] for h in items)
        share = defaultdict(float)
        for h in items:
            share[_norm(h["text"])] += h["weight"]
        for key, got in share.items():
            if key:
                scores[key] += ballot * (got / total)
    return scores

def _surface_pick(cands):
    """Choose how the winning reading is spelled out.

    Once the letters are settled, the remaining question is punctuation and
    spacing. Weight sums answer it badly for the same reason as above, so the
    exact spelling produced by the most distinct engines wins. The final
    tie-break prefers the shorter string, because a stray trailing mark is a
    far more common artifact than a genuinely omitted one.
    """
    groups = defaultdict(list)
    for h in cands:
        groups[h["text"]].append(h)

    def rank(item):
        text, items = item
        engines = {_engine(h.get("src", "")) for h in items}
        return (len(engines), sum(h["weight"] for h in items), -len(text))

    return max(groups.items(), key=rank)[0]

def _select(hyps, mode="auto"):
    """Select a short reading by engine vote, then settle its surface form.

    On document crops each engine's ballot is spread over the candidates in
    proportion to character similarity. Document words fail through character
    confusions, so several engines that nearly agree carry real evidence that
    all-or-nothing voting throws away. Scene crops are short and decisive, and
    partial credit only blurred them on the tuning split, so their votes stay
    exact. The route comes from image statistics, not from which dataset a
    crop belongs to.
    """
    scores = _engine_vote_scores(hyps)
    if not scores:
        return _mbr(hyps)

    if mode == "document":
        soft = defaultdict(float)
        for cand in scores:
            for other, weight in scores.items():
                soft[cand] += weight * _sim(cand, other)
        scores = soft

    key = max(scores.items(), key=lambda kv: kv[1])[0]
    cands = [h for h in hyps if _norm(h["text"]) == key]
    text = _surface_pick(cands)
    conf = min(0.99, max(h.get("conf", 0.9) for h in cands))
    words = [(w, round(conf, 4)) for w in text.split()]
    return {"text": text, "confidence": round(conf, 4), "words": words, "mode": "select"}

# Character-lattice fusion. Every selector before this one had to return a
# reading some engine had already produced, so its accuracy was capped by the
# best available hypothesis. On the tuning split that cap sat at 0.69 for the
# standard document profile while the truth was reachable 0.78 of the time by
# mixing engines character by character: one engine reads "Elcvcn", another
# reads "Eleven =", and the correct string exists only as a blend of the two.
# Aligning the hypotheses into columns and voting per column lifts that cap,
# because the result need not exist in any single hypothesis.
LATTICE_ENABLED = os.environ.get("CHORUS_LATTICE", "1") != "0"

def _align_columns(pivot, texts):
    """Align every text to the pivot and return one option per text per column.

    Columns interleave as ins[0], piv[0], ins[1], piv[1], ... ins[L]. The 'piv'
    columns hold whatever each text aligns to that pivot character, and the
    'ins' columns absorb characters a text has where the pivot has none. Every
    text contributes exactly one option to every column, which is what makes a
    per-column vote well defined.
    """
    span_len = len(pivot)
    per_text = []
    for text in texts:
        ins = [""] * (span_len + 1)
        piv = [""] * span_len
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
    for i in range(span_len + 1):
        cols.append([item[0][i] for item in per_text])
        if i < span_len:
            cols.append([item[1][i] for item in per_text])
    return cols

def _column_pick(col, hyps):
    """Decide one column: one ballot per engine, then restore the surface form.

    Test-time augmentation hands a single engine many correlated variants, so
    weighing each variant separately lets one engine outvote three that agree.
    Each engine therefore casts a single ballot worth its heaviest weight,
    split across the options its own variants produced. Votes are counted
    without case so that 'B' and 'b' are not treated as rival readings, and
    the winning surface form is the one the heaviest evidence actually wrote.
    """
    by_engine = {}
    surfaces = defaultdict(lambda: defaultdict(float))
    for opt, h in zip(col, hyps):
        eng = _engine(h.get("src", ""))
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

    key = max(tally.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]
    forms = surfaces[key]
    return max(forms.items(), key=lambda kv: (kv[1], kv[0]))[0]

def _lattice(hyps):
    """Fuse document readings character by character through a column lattice."""
    pivot = max(hyps, key=lambda h: h["weight"])["text"]
    cols = _align_columns(pivot, [h["text"] for h in hyps])
    text = " ".join("".join(_column_pick(col, hyps) for col in cols).split())
    if not text:
        return _rover(hyps)
    conf = min(0.99, max(h.get("conf", 0.9) for h in hyps))
    words = [(w, round(conf, 4)) for w in text.split()]
    return {"text": text, "confidence": round(conf, 4), "words": words, "mode": "lattice"}

def fuse(hyps, mode="auto"):
    """mode: auto|scene|document"""
    hyps = [h for h in hyps if h.get("text", "").strip()]
    if not hyps:
        return {"text": "", "confidence": 0.0, "words": []}

    # Document crops are fused character by character. Scene crops are not: a
    # scene crop is a single word photographed once, where picking the whole
    # reading an engine actually saw beats assembling a new one. Mixing hurt
    # scene text on the tuning split and helped document text there, so the
    # split follows the route, which comes from image statistics rather than
    # from which dataset a crop belongs to.
    if mode == "document" and LATTICE_ENABLED:
        return _lattice(hyps)

    max_tok = max(len(h["text"].split()) for h in hyps)
    # Short readings are selected, never spliced. Measured on a tuning split
    # that is disjoint from every reported benchmark split.
    if max_tok <= SPLICE_MIN_TOKENS:
        return _mbr(hyps) if FUSION_LEGACY else _select(hyps, mode)

    if mode == "scene":
        return _pick_champion(hyps)
    if mode == "document":
        return _rover(hyps)

    avg_tok = sum(len(h["text"].split()) for h in hyps) / len(hyps)
    if avg_tok <= 2.5 and max_tok <= 3:
        return _pick_champion(hyps)
    return _rover(hyps)
