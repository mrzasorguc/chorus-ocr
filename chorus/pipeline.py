"""Main orchestration for routing, TTA, OCR engines, fusion, and cleanup."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from . import consensus
from . import engines as E
from . import lang
from . import tta

# Document crops use broader augmentation and ROVER fusion.
DOC_PLAN = {
    "easyocr":   ["orig", "up2x", "bin", "clahe", "deskew", "deskew_up2x", "denoise"],
    "paddle":    ["orig", "up2x", "bin", "clahe", "deskew", "deskew_up2x", "denoise"],
    "tesseract": ["orig", "up2x", "bin", "deskew_up2x"],
    "got":       ["orig", "up2x"],
}
# Scene text uses lighter augmentation and champion selection.
SCENE_PLAN = {
    "easyocr":   ["orig", "up2x", "clahe"],
    "paddle":    ["orig", "up2x", "clahe"],
    "tesseract": ["orig", "up2x"],
    "got":       ["orig", "up2x"],
}

# Reliability weights are route-specific. Document weights come from a
# coordinate search on a tuning split (skip=1000) that shares no samples with
# any reported benchmark split; see scripts/tune_fusion.py and docs/BENCHMARKS.md.
MODE_REL = {
    "scene":    {"got": 1.0, "paddle": 1.0, "easyocr": 1.0, "tesseract": 1.0},
    "document": {"got": 1.15, "paddle": 0.8, "easyocr": 1.25, "tesseract": 1.0},
}

# Engines are independent of each other, so they run concurrently by default.
# The heavy work inside every engine happens in native code that releases the
# GIL, so threads overlap real compute rather than fighting over the
# interpreter. Set CHORUS_PARALLEL_ENGINES=0 to force the sequential path.
PARALLEL_ENGINES = os.environ.get("CHORUS_PARALLEL_ENGINES", "1") != "0"

def _run_engine(eng, fn, ordered, variants, rel, verbose):
    """Run one engine over its variant list and return its hypotheses.

    Kept free of shared mutable state so it is safe to call from a worker
    thread. Timing is returned instead of being written to a shared dict.
    """
    produced = []
    spent = 0.0
    for vn in ordered:
        im, q = variants[vn]
        engine_started = time.perf_counter()
        try:
            r = fn(im)
        except Exception as ex:
            spent += time.perf_counter() - engine_started
            if verbose:
                print(f"[skip] {eng}:{vn} -> {ex!r}")
            if eng == "got":
                break
            continue
        spent += time.perf_counter() - engine_started
        if not r["text"].strip():
            continue
        q2 = q * (1.12 if eng == "got" and vn == "orig" else 1.0)
        produced.append({"text": r["text"], "conf": r["conf"],
                         "weight": rel * q2 * (0.5 + 0.5 * r["conf"]),
                         "src": f"{eng}:{vn}"})
        if eng == "got" and vn != "orig":
            break
    return produced, spent

def _classify(img):
    """Goruntu turunu sadece geometriyle degil, renk ve tarama yapisiyla ayir.

    FUNSD gibi taranmis form kelimeleri neredeyse tam gri tonludur ve genellikle
    cok alcak kirpimlardir. IIIT5K sahne yazilari ise cogunlukla renkli veya daha
    yuksektir. Bu ayrim, iki veri kumesindeki 100'er ornek uzerinde olculmustur.
    """
    h, w = img.shape[:2]
    # Channel spread captures color while tolerating small JPEG artifacts.
    spread = img.max(axis=2).astype("float32") - img.min(axis=2).astype("float32")
    mean_color = float(spread.mean())
    colored_ratio = float((spread > 20).mean())

    # Strong color is the primary scene-text signal.
    if mean_color >= 0.58 or colored_ratio >= 0.002:
        return "scene"
    # Tall grayscale crops are treated as scene text by the current heuristic.
    if h >= 25:
        return "scene"
    return "document"

def read(path_or_img, use=("easyocr", "paddle", "tesseract", "got"), fast=False, verbose=False, debug=False, profile="quality"):
    if profile not in {"quality", "interactive"}:
        raise ValueError("profile must be quality or interactive")
    started_at = time.perf_counter()
    img = cv2.imread(path_or_img) if isinstance(path_or_img, str) else path_or_img
    if img is None:
        raise ValueError("goruntu okunamadi")
    mode = _classify(img)
    plan = SCENE_PLAN if mode == "scene" else DOC_PLAN
    relmap = MODE_REL[mode]
    interactive = profile == "interactive"
    variants = {name: (im, q) for name, im, q in tta.build_variants(img, fast=(fast or interactive))}
    hyps = []
    engine_seconds = {}
    tasks = []
    for eng in use:
        fn = E.ENGINES.get(eng)
        if fn is None:
            continue
        rel = relmap.get(eng, E.RELIABILITY.get(eng, 0.8))
        if interactive:
            # The web profile keeps all selected engines but runs each once.
            # Exhaustive TTA remains available through the quality profile.
            vnames = ["orig"]
        elif fast:
            vnames = ["orig", "up2x"]
        else:
            vnames = list(plan.get(eng, ["orig"]))
        seen, ordered = set(), []
        for vn in vnames:
            if vn not in seen and vn in variants:
                seen.add(vn)
                ordered.append(vn)
        tasks.append((eng, fn, ordered, variants, rel, verbose))

    parallel = PARALLEL_ENGINES and len(tasks) > 1
    if parallel:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            outcomes = list(pool.map(lambda task: _run_engine(*task), tasks))
    else:
        outcomes = [_run_engine(*task) for task in tasks]

    # Hypotheses are merged in the requested engine order, so fusion sees the
    # exact same input whether the engines ran concurrently or one by one.
    for task, (produced, spent) in zip(tasks, outcomes):
        eng = task[0]
        hyps.extend(produced)
        engine_seconds[eng] = engine_seconds.get(eng, 0.0) + spent

    fused = consensus.fuse(hyps, mode=mode)
    fused["text"] = lang.polish(fused["text"], mode=mode)
    fused["n_hypotheses"] = len(hyps)
    fused["sources"] = [h["src"] for h in hyps]
    fused["route"] = mode
    fused["profile"] = profile
    fused["parallel_engines"] = parallel
    fused["engine_seconds"] = {name: round(value, 3) for name, value in engine_seconds.items()}
    fused["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    fused["low_conf_words"] = [w for w, c in fused.get("words", []) if c < 0.55]
    if debug:
        fused["hypotheses"] = [{"src": h["src"], "text": h["text"], "conf": round(float(h["conf"]), 4), "weight": round(float(h["weight"]), 4)} for h in hyps]
    return fused
