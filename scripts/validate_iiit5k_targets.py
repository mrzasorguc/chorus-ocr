"""Validate selected IIIT5K rows after routing or consensus changes."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.benchmark_datasets import cer, load_iiit5k
from chorus import pipeline

TARGET_ROWS = (10, 15, 43, 44, 55, 64, 84)


def main() -> None:
    samples = load_iiit5k(100)
    exact_matches = 0

    for row_number in TARGET_ROWS:
        image, reference = samples[row_number - 1]
        result = pipeline.read(image)
        hypothesis = result["text"]
        is_exact = hypothesis.casefold() == reference.casefold()
        exact_matches += int(is_exact)
        row = {
            "row": row_number,
            "reference": reference,
            "hypothesis": hypothesis,
            "exact": is_exact,
            "cer": round(cer(reference.casefold(), hypothesis.casefold()), 4),
            "route": result.get("route"),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {"target_accuracy": exact_matches / len(TARGET_ROWS), "n": len(TARGET_ROWS)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
