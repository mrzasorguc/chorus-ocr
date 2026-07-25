"""Benchmark OCR engines and Chorus on FUNSD and IIIT5K."""

import os, sys, json, time, random, argparse
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cv2
import numpy as np
from chorus import engines as E
from chorus import pipeline

def cer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return float(d[-1, -1]) / max(1, len(ref))

def norm(s):
    return " ".join(str(s).split())

def load_funsd(n, seed=42, skip=0):
    base = os.path.join(ROOT, "bench", "funsd", "dataset", "testing_data")
    ann_dir = os.path.join(base, "annotations")
    img_dir = os.path.join(base, "images")
    samples = []
    for f in sorted(os.listdir(ann_dir)):
        with open(os.path.join(ann_dir, f), encoding="utf-8") as fh:
            j = json.load(fh)
        img_path = os.path.join(img_dir, f.replace(".json", ".png"))
        for item in j["form"]:
            for w in item.get("words", []):
                t = norm(w.get("text", ""))
                if len(t) >= 2:
                    samples.append((img_path, w["box"], t))
    random.Random(seed).shuffle(samples)
    out = []
    valid = 0
    for img_path, box, t in samples:
        if len(out) >= n:
            break
        img = cv2.imread(img_path)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = 3
        crop = img[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
        if crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 8:
            valid += 1
            if valid <= skip:
                continue
            out.append((crop, t))
    return out

def load_iiit5k(n, seed=42, skip=0):
    from scipy.io import loadmat
    base = os.path.join(ROOT, "bench", "iiit5k", "IIIT5K")
    md = loadmat(os.path.join(base, "testdata.mat"))
    arr = md["testdata"][0]
    samples = [(str(a["ImgName"][0]), str(a["GroundTruth"][0])) for a in arr]
    random.Random(seed).shuffle(samples)
    out = []
    valid = 0
    for name, gtw in samples:
        if len(out) >= n:
            break
        img = cv2.imread(os.path.join(base, name))
        if img is not None:
            valid += 1
            if valid <= skip:
                continue
            out.append((img, norm(gtw)))
    return out

def run(dataset, n, use, profile="quality", tag=None, skip=0):
    data = load_funsd(n, skip=skip) if dataset == "funsd" else load_iiit5k(n, skip=skip)
    print(f"{dataset}: {len(data)} ornek yuklendi (profile={profile})", flush=True)
    if data:
        warm = data[0][0]
        for eng in use:
            try:
                E.ENGINES[eng](warm)
            except Exception:
                pass
        try:
            pipeline.read(warm, use=tuple(use), profile=profile)
        except Exception:
            pass
        print("warm-up tamam", flush=True)
    keys = list(use) + ["CHORUS"]
    stats = {k: {"cer_sum": 0.0, "exact": 0, "n": 0, "sec": 0.0} for k in keys}
    for idx, (img, ref) in enumerate(data):
        rl = ref.lower()
        for eng in use:
            hyp = ""
            dt = 0.0
            try:
                t0 = time.time()
                r = E.ENGINES[eng](img)
                dt = time.time() - t0
                hyp = norm(r["text"])
            except Exception:
                pass
            s = stats[eng]
            s["cer_sum"] += cer(rl, hyp.lower())
            s["exact"] += int(hyp.lower() == rl)
            s["n"] += 1
            s["sec"] += dt
        hyp = ""
        t0 = time.time()
        try:
            h = pipeline.read(img, use=tuple(use), profile=profile)
            hyp = norm(h["text"])
        except Exception:
            pass
        dt = time.time() - t0
        s = stats["CHORUS"]
        s["cer_sum"] += cer(rl, hyp.lower())
        s["exact"] += int(hyp.lower() == rl)
        s["n"] += 1
        s["sec"] += dt
        if (idx + 1) % 25 == 0:
            print(f"[{idx+1}/{len(data)}]", flush=True)
    summary = {}
    for k, s in stats.items():
        if s["n"]:
            summary[k] = {"word_acc": round(s["exact"] / s["n"], 4),
                          "avg_cer": round(s["cer_sum"] / s["n"], 4),
                          "avg_sec": round(s["sec"] / s["n"], 2), "n": s["n"]}
    print(json.dumps(summary, indent=1), flush=True)
    name = tag or dataset
    outp = os.path.join(ROOT, "out", f"bench_{name}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("saved:", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["funsd", "iiit5k"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--engines", default="easyocr,paddle,tesseract,got")
    ap.add_argument("--profile", default="quality", choices=["quality", "interactive"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip this many samples before collecting; use a non-zero "
                         "value to build a tuning split disjoint from the test split.")
    a = ap.parse_args()
    run(a.dataset, a.n, a.engines.split(","), profile=a.profile, tag=a.tag, skip=a.skip)
