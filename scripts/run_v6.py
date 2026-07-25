"""Test-split measurement for the beam + language prior release.

Driven from Python rather than a shell: the background runner uses PowerShell,
where `&&` and cmd's `set` are both invalid, and a failed chain there dies
silently after the first command.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "scripts", "benchmark_datasets.py")

FAST = "easyocr,paddle,tesseract"
FULL = "easyocr,paddle,tesseract,got"

RUNS = [
    ("funsd", FAST, "interactive", "v6_standard_funsd"),
    ("funsd", FULL, "interactive", "v6_max_funsd"),
    ("funsd", FULL, "quality", "v6_maxq_funsd"),
    ("iiit5k", FAST, "interactive", "v6_standard_iiit5k"),
    ("iiit5k", FULL, "interactive", "v6_max_iiit5k"),
    ("iiit5k", FULL, "quality", "v6_maxq_iiit5k"),
]

env = dict(os.environ)
env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
env["PYTHONIOENCODING"] = "utf-8"

for i, (dataset, engines, profile, tag) in enumerate(RUNS, 1):
    print(f"=== [{i}/6] {tag} basliyor {time.strftime('%H:%M:%S')}", flush=True)
    result = subprocess.run(
        [sys.executable, BENCH, dataset, "--n", "100",
         "--engines", engines, "--profile", profile, "--tag", tag],
        cwd=ROOT, env=env,
    )
    if result.returncode != 0:
        print(f"=== {tag} BASARISIZ kod={result.returncode}", flush=True)
        break
    print(f"=== [{i}/6] {tag} bitti {time.strftime('%H:%M:%S')}", flush=True)

print("TUM_V6_BITTI", flush=True)
