"""Measure single-engine OCR baselines on the local test set."""
import os, json, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(ROOT, "testset")
OUT = os.path.join(ROOT, "out")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

with open(os.path.join(TS, "ground_truth.json"), encoding="utf-8") as f:
    gt = json.load(f)

def cer(ref, hyp):
    import numpy as np
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    d[:, 0] = range(len(ref)+1)
    d[0, :] = range(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return float(d[-1, -1]) / max(1, len(ref))

def norm(s):
    return " ".join(s.split())

results = {}

def record(fname, engine, txt, t0):
    results.setdefault(fname, {})[engine] = {
        "text": txt,
        "cer": round(cer(norm(gt[fname]), norm(txt)), 4),
        "sec": round(time.time()-t0, 1),
    }

print("=== EasyOCR ===", flush=True)
try:
    import easyocr, torch
    rd = easyocr.Reader(["tr", "en"], gpu=torch.cuda.is_available())
    for fname in gt:
        t0 = time.time()
        parts = rd.readtext(os.path.join(TS, fname), detail=0, paragraph=True)
        record(fname, "easyocr", " ".join(parts), t0)
        print(fname, "ok", flush=True)
except Exception:
    results["_easyocr_error"] = traceback.format_exc()

print("=== PaddleOCR ===", flush=True)
try:
    from paddleocr import PaddleOCR
    po = PaddleOCR(lang="tr", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    for fname in gt:
        t0 = time.time()
        out = po.predict(os.path.join(TS, fname))
        txt = " ".join(out[0]["rec_texts"]) if out else ""
        record(fname, "paddleocr", txt, t0)
        print(fname, "ok", flush=True)
except Exception:
    results["_paddleocr_error"] = traceback.format_exc()

print("=== Tesseract ===", flush=True)
try:
    import pytesseract
    from PIL import Image
    ver = str(pytesseract.get_tesseract_version())
    for fname in gt:
        t0 = time.time()
        txt = pytesseract.image_to_string(Image.open(os.path.join(TS, fname)), lang="tur+eng")
        record(fname, "tesseract", txt, t0)
        print(fname, "ok", flush=True)
except Exception:
    results["_tesseract_error"] = traceback.format_exc()

with open(os.path.join(OUT, "baseline_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)

print("\n=== OZET (CER: 0 = mukemmel) ===")
for fname in gt:
    if fname in results:
        for eng, r in results[fname].items():
            print(f"{fname:16s} {eng:10s} CER={r['cer']:.4f} ({r['sec']}s)")
print("DONE")
