"""Command-line interface for Chorus."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import read


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chorus",
        description="Run multi-engine OCR on an image.",
    )
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use only the original and 2x-upscaled image variants",
    )
    parser.add_argument(
        "--engines",
        default="easyocr,paddle,tesseract",
        help="Comma-separated engines: easyocr,paddle,tesseract,got",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    parser.add_argument("--debug", action="store_true", help="Include engine hypotheses")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engines = tuple(item.strip() for item in args.engines.split(",") if item.strip())
    result = read(
        args.image,
        use=engines,
        fast=args.fast,
        verbose=True,
        debug=args.debug,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(result["text"])
    print(
        f"\n[confidence={result['confidence']:.3f} "
        f"hypotheses={result['n_hypotheses']} "
        f"route={result.get('route', 'unknown')}]"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
