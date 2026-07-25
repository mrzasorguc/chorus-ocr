"""Run a smoke test for the GOT-OCR engine."""

from pathlib import Path
import sys
import traceback

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from chorus import engines as engine_registry


def main() -> None:
    image_path = ROOT / "testset" / "en_lowres.png"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Test image could not be read: {image_path}")

    try:
        result = engine_registry.got_read(image)
        print("GOT OK:", result["text"][:300])
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
