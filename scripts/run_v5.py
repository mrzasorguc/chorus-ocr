"""Run the six test-split benchmarks for the v5 release, one after another.

Shell chaining with && is not portable across the shells this project gets
launched from, and quoting environment assignments through a shell has bitten
us before. Driving the runs from Python passes arguments directly to each
process and keeps the sequence readable.
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
    ("funsd", FAST, "interactive", "v5_standard_funsd"),
    ("funsd", FULL, "interactive", "v5_max_funsd"),
    ("funsd", FULL, "quality", "v5_maxq_funsd"),
    ("iiit5k", FAST, "interactive", "v5_standard_iiit5k"),
    ("iiit5k", FULL, "interactive", "v5_max_iiit5k"),
    ("iiit5k", FULL, "quality", "v5_maxq_iiit5k"),
]


def main():
    env = dict(os.environ)
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    env["PYTHONIOENCODING"] = "utf-8"
    for i, (dataset, engines, profile, tag) in enumerate(RUNS, 1):
        print(f"=== [{i}/{len(RUNS)}] {tag} basliyor {time.strftime('%H:%M:%S')}", flush=True)
        cmd = [sys.executable, BENCH, dataset, "--n", "100",
               "--engines", engines, "--profile", profile, "--tag", tag]
        res = subprocess.run(cmd, cwd=ROOT, env=env)
        if res.returncode != 0:
            print(f"=== {tag} BASARISIZ kod={res.returncode}", flush=True)
            return res.returncode
        print(f"=== [{i}/{len(RUNS)}] {tag} bitti {time.strftime('%H:%M:%S')}", flush=True)
    print("TUM_V5_BITTI", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
