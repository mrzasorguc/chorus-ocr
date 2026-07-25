"""Print detailed OCR hypotheses for selected IIIT5K errors."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.benchmark_datasets import load_iiit5k
from chorus import pipeline

TARGET_ROWS = (10, 15, 43, 44, 55, 64, 84)


def main() -> None:
    samples = load_iiit5k(100)
    for row_number in TARGET_ROWS:
        image, reference = samples[row_number - 1]
        result = pipeline.read(image, debug=True)
        print(
            f"\nROW {row_number} REF={reference!r} "
            f"HYP={result['text']!r} ROUTE={result['route']}"
        )
        hypotheses = sorted(
            result.get("hypotheses", []), key=lambda item: item["src"]
        )
        for hypothesis in hypotheses:
            print(
                f"  {hypothesis['src']} => {hypothesis['text']!r} "
                f"conf={hypothesis['conf']} weight={hypothesis['weight']}"
            )


if __name__ == "__main__":
    main()
