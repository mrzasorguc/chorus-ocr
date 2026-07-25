"""Offline fusion replay and oracle analysis on the tuning (dev) split.

The engines are expensive, so their raw hypotheses were dumped once with
scripts/dump_hyps.py. This module replays fusion over those recorded
hypotheses, reproducing the exact weights the pipeline would assign. That
makes it possible to evaluate a fusion change in seconds instead of minutes,
and to measure how much accuracy is theoretically reachable at all.

Everything here runs on the tuning split (skip=1000). It shares no samples
with any reported benchmark split.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chorus import consensus, lang  # noqa: E402
from chorus.pipeline import DOC_PLAN, MODE_REL, SCENE_PLAN  # noqa: E402

RESEARCH = os.path.join(ROOT, "out", "research_20260724")
DEV_FILES = {
    "funsd": os.path.join(RESEARCH, "hyps_funsd_skip1000_n60.json"),
    "iiit5k": os.path.join(RESEARCH, "hyps_iiit5k_skip1000_n60.json"),
}

# Mode definitions mirror chorus/web.py.
MODES = {
    "standard": {"engines": ("easyocr", "paddle", "tesseract"), "profile": "interactive"},
    "max": {"engines": ("easyocr", "paddle", "tesseract", "got"), "profile": "interactive"},
    "maxq": {"engines": ("easyocr", "paddle", "tesseract", "got"), "profile": "quality"},
}

# ---------------------------------------------------------------- metrics

def cer(ref: str, hyp: str) -> float:
    """Character error rate, identical to scripts/benchmark_datasets.py."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)

def norm(s: str) -> str:
    return " ".join((s or "").split())

def score(pairs):
    """pairs: iterable of (ground_truth, prediction)."""
    exact = 0
    cer_sum = 0.0
    n = 0
    for gt, pred in pairs:
        rl = norm(gt).lower()
        hl = norm(pred).lower()
        exact += int(hl == rl)
        cer_sum += cer(rl, hl)
        n += 1
    if not n:
        return {"word_acc": 0.0, "avg_cer": 0.0, "n": 0}
    return {"word_acc": round(exact / n, 4), "avg_cer": round(cer_sum / n, 4), "n": n}

# ---------------------------------------------------------------- replay

def weighted(record, engines, profile):
    """Rebuild the hypothesis list exactly as chorus.pipeline.read would."""
    route = record["route"]
    relmap = MODE_REL[route]
    plan = SCENE_PLAN if route == "scene" else DOC_PLAN
    out = []
    for eng in engines:
        if profile == "interactive":
            allowed = ["orig"]
        else:
            allowed = list(plan.get(eng, ["orig"]))
        rel = relmap.get(eng, 0.8)
        for vn in allowed:
            h = next((x for x in record["hyps"]
                      if x["eng"] == eng and x["variant"] == vn), None)
            if h is None or not h["text"].strip():
                continue
            q2 = h["quality"] * (1.12 if eng == "got" and vn == "orig" else 1.0)
            out.append({
                "text": h["text"],
                "conf": h["conf"],
                "weight": rel * q2 * (0.5 + 0.5 * h["conf"]),
                "src": f"{eng}:{vn}",
            })
            # GOT stops after the first non-orig variant that produced text.
            if eng == "got" and vn != "orig":
                break
    return out

def replay(records, mode, fuse_fn=None):
    cfg = MODES[mode]
    fuse_fn = fuse_fn or consensus.fuse
    pairs = []
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        if not hyps:
            pairs.append((rec["gt"], ""))
            continue
        fused = fuse_fn(hyps, mode=rec["route"])
        pairs.append((rec["gt"], lang.polish(fused["text"], mode=rec["route"])))
    return pairs

# ---------------------------------------------------------------- oracles

def oracle_select(records, mode):
    """Ceiling for any method that PICKS one existing hypothesis."""
    cfg = MODES[mode]
    pairs = []
    reachable = 0
    for rec in records:
        hyps = weighted(rec, cfg["engines"], cfg["profile"])
        gl = norm(rec["gt"]).lower()
        best, best_cer = "", 1e9
        hit = False
        for h in hyps:
            t = lang.polish(h["text"], mode=rec["route"])
            if norm(t).lower() == gl:
                hit = True
                best, best_cer = t, -1.0
                break
            c = cer(gl, norm(t).lower())
            if c < best_cer:
                best, best_cer = t, c
        reachable += int(hit)
        pairs.append((rec["gt"], best))
    return pairs, reachable

def single_engine(records, engine, variant="orig"):
    pairs = []
    for rec in records:
        h = next((x for x in rec["hyps"]
                  if x["eng"] == engine and x["variant"] == variant), None)
        pairs.append((rec["gt"], h["text"] if h else ""))
    return pairs

# ---------------------------------------------------------------- report

def load(dataset):
    with open(DEV_FILES[dataset], encoding="utf-8") as fh:
        return json.load(fh)["records"]

def main():
    for dataset in ("funsd", "iiit5k"):
        records = load(dataset)
        print(f"===== {dataset.upper()}  (dev split, n={len(records)})")
        routes = {}
        for rec in records:
            routes[rec["route"]] = routes.get(rec["route"], 0) + 1
        print(f"  rota dagilimi: {routes}")

        print("  -- tekil motorlar (orig)")
        for eng in ("easyocr", "paddle", "tesseract", "got"):
            s = score(single_engine(records, eng))
            print(f"     {eng:10s} acc={s['word_acc']:.4f}  cer={s['avg_cer']:.4f}")

        print("  -- mevcut fusion / oracle tavani")
        for mode in ("standard", "max", "maxq"):
            cur = score(replay(records, mode))
            orc_pairs, reachable = oracle_select(records, mode)
            orc = score(orc_pairs)
            gap = orc["word_acc"] - cur["word_acc"]
            print(f"     {mode:9s} mevcut={cur['word_acc']:.4f}  "
                  f"oracle={orc['word_acc']:.4f}  bosluk={gap:+.4f}  "
                  f"| dogru cevap adaylarda: {reachable}/{len(records)}")
        print()

if __name__ == "__main__":
    main()
