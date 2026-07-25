"""Compare routing features for FUNSD and IIIT5K samples."""

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_datasets import load_funsd, load_iiit5k

FEATURE_NAMES = (
    "height", "width", "area", "aspect_ratio", "saturation", "high_saturation",
    "color_spread", "high_color_spread", "white_ratio", "dark_ratio",
    "gray_std", "edge_ratio",
)


def extract_features(image):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spread = image.max(axis=2).astype(float) - image.min(axis=2).astype(float)
    return [
        height,
        width,
        height * width,
        max(width, height) / max(1, min(width, height)),
        float(hsv[:, :, 1].mean()),
        float((hsv[:, :, 1] > 40).mean()),
        float(spread.mean()),
        float((spread > 20).mean()),
        float((gray > 230).mean()),
        float((gray < 80).mean()),
        float(gray.std()),
        float((cv2.Canny(gray, 80, 160) > 0).mean()),
    ]


def load_samples(dataset):
    loader = load_funsd if dataset == "funsd" else load_iiit5k
    return loader(100)


def main() -> None:
    feature_sets = {}
    for dataset in ("funsd", "iiit5k"):
        values = np.array([extract_features(image) for image, _ in load_samples(dataset)])
        feature_sets[dataset] = values
        print(f"\n{dataset}")
        for index, name in enumerate(FEATURE_NAMES):
            column = values[:, index]
            print(
                f"{name:18s} p10={np.percentile(column, 10):7.2f} "
                f"p25={np.percentile(column, 25):7.2f} "
                f"median={np.median(column):7.2f} "
                f"p75={np.percentile(column, 75):7.2f} "
                f"p90={np.percentile(column, 90):7.2f}"
            )

    print("\nBEST SINGLE-FEATURE ROUTING RULES")
    for index, name in enumerate(FEATURE_NAMES):
        values = np.unique(
            np.concatenate((feature_sets["funsd"][:, index], feature_sets["iiit5k"][:, index]))
        )
        best = None
        for threshold in values:
            for high_is_scene in (True, False):
                funsd_scene = (
                    feature_sets["funsd"][:, index] >= threshold
                    if high_is_scene
                    else feature_sets["funsd"][:, index] < threshold
                )
                iiit_scene = (
                    feature_sets["iiit5k"][:, index] >= threshold
                    if high_is_scene
                    else feature_sets["iiit5k"][:, index] < threshold
                )
                score = ((~funsd_scene).mean() + iiit_scene.mean()) / 2
                candidate = (
                    score,
                    threshold,
                    high_is_scene,
                    (~funsd_scene).mean(),
                    iiit_scene.mean(),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        print(name, best)


if __name__ == "__main__":
    main()
