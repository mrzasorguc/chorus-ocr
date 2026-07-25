"""Verify that concurrent engine execution is identical and faster.

Runs the same images through the pipeline twice, once with threads and once
without, and reports whether the fused text matches and how long each took.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chorus import pipeline

IMAGES = [
    "testset/en_lowres.png",
    "testset/tr_clean.png",
    "testset/tr_hard.png",
]
ENGINES = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("easyocr", "paddle", "tesseract", "got")
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "quality"


def run(image, parallel):
    pipeline.PARALLEL_ENGINES = parallel
    started = time.perf_counter()
    out = pipeline.read(image, use=ENGINES, profile=PROFILE)
    return out, time.perf_counter() - started


print(f"engines={','.join(ENGINES)} profile={PROFILE}")
print("warm-up...")
run(IMAGES[0], True)
run(IMAGES[0], False)
print("warm-up tamam", flush=True)

seq_total = par_total = 0.0
mismatches = 0
for image in IMAGES:
    seq_out, seq_sec = run(image, False)
    par_out, par_sec = run(image, True)
    seq_total += seq_sec
    par_total += par_sec
    same = seq_out["text"] == par_out["text"]
    same_src = seq_out["sources"] == par_out["sources"]
    if not (same and same_src):
        mismatches += 1
    print(f"{image}: seri {seq_sec:.2f}s | paralel {par_sec:.2f}s | "
          f"ayni_metin={same} ayni_kaynaklar={same_src}", flush=True)
    if not same:
        print(f"   seri    -> {seq_out['text']!r}")
        print(f"   paralel -> {par_out['text']!r}")

speedup = (seq_total / par_total) if par_total else 0.0
print(f"\nTOPLAM seri {seq_total:.2f}s | paralel {par_total:.2f}s | hizlanma {speedup:.2f}x")
print(f"UYUSMAZLIK={mismatches}")
print("PARALLEL_CHECK_DONE")
