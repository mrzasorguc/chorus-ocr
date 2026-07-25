"""Compare OCR engines and Chorus on the local test set."""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cv2
import numpy as np
from chorus import engines as E
from chorus import pipeline

TS = os.path.join(ROOT, "testset")
OUT = os.path.join(ROOT, "out")

with open(os.path.join(TS, "ground_truth.json"), encoding="utf-8") as f:
    gt = json.load(f)

def cer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return float(d[-1, -1]) / max(1, len(ref))

def norm(s):
    return " ".join(s.split())

use = sys.argv[1].split(",") if len(sys.argv) > 1 else ["easyocr", "paddle", "tesseract", "got"]
res = {}
for fname, ref in gt.items():
    img = cv2.imread(os.path.join(TS, fname))
    row = {}
    for eng in use:
        try:
            t0 = time.time()
            r = E.ENGINES[eng](img)
            row[eng] = {"cer": round(cer(norm(ref), norm(r["text"])), 4),
                        "sec": round(time.time() - t0, 1), "text": r["text"]}
        except Exception as ex:
            row[eng] = {"error": repr(ex)[:200]}
    t0 = time.time()
    h = pipeline.read(img, use=tuple(use))
    row["CHORUS"] = {"cer": round(cer(norm(ref), norm(h["text"])), 4),
                        "sec": round(time.time() - t0, 1),
                        "conf": h["confidence"], "text": h["text"]}
    res[fname] = row
    print(f"\n== {fname} ==", flush=True)
    for k, v in row.items():
        print(f"  {k:10s} CER={v.get('cer', 'ERR')} ({v.get('sec', '-')}s)", flush=True)

with open(os.path.join(OUT, "eval_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\nDONE")
