"""Cache raw engine hypotheses for a data split.

Fusion research needs to compare many voting strategies. Re-running OCR for
each idea costs minutes per attempt, so this script runs every engine on every
planned TTA variant once and writes the raw output to disk. Tuning scripts then
replay the cache in milliseconds.

Use a non-zero --skip so the tuning split never overlaps the test split.
"""

import os, sys, json, time, argparse
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from benchmark_datasets import load_funsd, load_iiit5k, norm
from chorus import engines as E
from chorus import pipeline, tta


def planned_variants(route, engine):
    plan = pipeline.SCENE_PLAN if route == "scene" else pipeline.DOC_PLAN
    return list(plan.get(engine, ["orig"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["funsd", "iiit5k"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--engines", default="easyocr,paddle,tesseract,got")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    use = a.engines.split(",")
    loader = load_funsd if a.dataset == "funsd" else load_iiit5k
    data = loader(a.n, skip=a.skip)
    print(f"{a.dataset}: {len(data)} ornek (skip={a.skip})", flush=True)

    # Warm every engine so cached timings reflect steady state, not model load.
    if data:
        for eng in use:
            try:
                E.ENGINES[eng](data[0][0])
            except Exception as ex:
                print(f"warm-up {eng} basarisiz: {ex!r}", flush=True)
        print("warm-up tamam", flush=True)

    records = []
    t_start = time.time()
    for idx, (img, ref) in enumerate(data):
        route = pipeline._classify(img)
        variants = {name: (im, q) for name, im, q in tta.build_variants(img, fast=False)}
        hyps = []
        for eng in use:
            fn = E.ENGINES.get(eng)
            if fn is None:
                continue
            for vn in planned_variants(route, eng):
                if vn not in variants:
                    continue
                im, q = variants[vn]
                t0 = time.time()
                try:
                    r = fn(im)
                    text, conf = r["text"], float(r["conf"])
                except Exception:
                    text, conf = "", 0.0
                hyps.append({"eng": eng, "variant": vn, "quality": q,
                             "text": text, "conf": conf,
                             "sec": round(time.time() - t0, 4)})
        records.append({"gt": ref, "route": route,
                        "height": int(img.shape[0]), "width": int(img.shape[1]),
                        "hyps": hyps})
        if (idx + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"[{idx+1}/{len(data)}] {el:.0f}s", flush=True)

    outp = a.out or os.path.join(ROOT, "out", "research_20260724",
                                 f"hyps_{a.dataset}_skip{a.skip}_n{a.n}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"dataset": a.dataset, "skip": a.skip, "n": len(records),
                   "engines": use, "records": records}, f, ensure_ascii=False)
    print("saved:", outp, flush=True)


if __name__ == "__main__":
    main()
