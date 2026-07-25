"""Single-sample GOT-OCR health probe with explicit timing."""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cv2
from chorus import engines as E

img = cv2.imread(os.path.join(ROOT, "testset", "tr_clean.png"))
print("image loaded:", img is not None, flush=True)
fn = E.ENGINES.get("got")
print("got engine registered:", fn is not None, flush=True)
if fn is None:
    sys.exit(2)
t0 = time.time()
try:
    r = fn(img)
    print("cold run sec:", round(time.time() - t0, 2), flush=True)
    print("text:", repr(r["text"])[:200], flush=True)
except Exception as ex:
    print("FAILED after", round(time.time() - t0, 2), "sec:", repr(ex), flush=True)
    sys.exit(3)
t1 = time.time()
r2 = fn(img)
print("warm run sec:", round(time.time() - t1, 2), flush=True)
print("GOT HEALTHY", flush=True)
