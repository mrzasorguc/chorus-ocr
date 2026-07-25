"""Render benchmark charts from the measured JSON results.

Reads out/bench_$CHORUS_BENCH_TAG_*.json (default v4) and writes PNG charts
into docs/images/.
Re-run this after a new benchmark run to refresh the images in the README.

    python scripts/make_charts.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out")
TAG = os.environ.get("CHORUS_BENCH_TAG", "v5")
IMG_DIR = os.path.join(ROOT, "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)

INK = "#1b2733"
MUTED = "#7b8794"
GRID = "#dfe3e8"
RIVAL = "#9aa5b1"
RIVAL_TOP = "#616e7c"
OURS = "#2f6fed"
OURS_DARK = "#12326e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load(tag):
    path = os.path.join(OUT_DIR, f"bench_{tag}.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rows_for(dataset):
    """Return [(label, word_acc, cer, sec, is_ours), ...] for one dataset."""
    standard = load(f"{TAG}_standard_{dataset}")
    maximum = load(f"{TAG}_max_{dataset}")
    quality = load(f"{TAG}_maxq_{dataset}")
    singles = [
        ("EasyOCR", "easyocr"),
        ("PaddleOCR", "paddle"),
        ("Tesseract", "tesseract"),
        ("GOT-OCR 2.0", "got"),
    ]
    rows = []
    for label, key in singles:
        entry = maximum.get(key) or standard.get(key)
        if entry:
            rows.append((label, entry["word_acc"], entry["avg_cer"], entry["avg_sec"], False))
    for label, data in [
        ("Chorus Standard", standard),
        ("Chorus Maximum", maximum),
        ("Chorus Max+Quality", quality),
    ]:
        # Results written before the rename use the old key.
        entry = data.get("CHORUS") or data["HYPERLEX"]
        rows.append((label, entry["word_acc"], entry["avg_cer"], entry["avg_sec"], True))
    return rows


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_chart(dataset, title, subtitle, filename):
    rows = rows_for(dataset)
    rows_sorted = sorted(rows, key=lambda r: r[1])
    labels = [r[0] for r in rows_sorted]
    values = [r[1] * 100 for r in rows_sorted]
    colors = [OURS if r[4] else RIVAL for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, row in zip(bars, rows_sorted):
        ax.text(bar.get_width() + 1.1, bar.get_y() + bar.get_height() / 2,
                f"{row[1] * 100:.0f}%", va="center", ha="left",
                fontsize=11, fontweight="bold",
                color=OURS_DARK if row[4] else RIVAL_TOP)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Word accuracy (%)  —  higher is better", fontsize=10)
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    for tick, row in zip(ax.get_yticklabels(), rows_sorted):
        if row[4]:
            tick.set_fontweight("bold")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", path)


def cer_chart(filename):
    funsd = {r[0]: r for r in rows_for("funsd")}
    iiit = {r[0]: r for r in rows_for("iiit5k")}
    labels = list(funsd.keys())
    idx = range(len(labels))
    height = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
    ax.barh([i + height / 2 for i in idx], [funsd[k][2] for k in labels],
            height=height, color="#2f6fed", label="FUNSD (documents)")
    ax.barh([i - height / 2 for i in idx], [iiit[k][2] for k in labels],
            height=height, color="#f0a202", label="IIIT5K (scene text)")
    for i, k in enumerate(labels):
        ax.text(funsd[k][2] + 0.006, i + height / 2, f"{funsd[k][2]:.3f}",
                va="center", fontsize=8.5, color=RIVAL_TOP)
        ax.text(iiit[k][2] + 0.006, i - height / 2, f"{iiit[k][2]:.3f}",
                va="center", fontsize=8.5, color=RIVAL_TOP)
    ax.set_yticks(list(idx))
    ax.set_yticklabels(labels)
    for tick, k in zip(ax.get_yticklabels(), labels):
        if funsd[k][4]:
            tick.set_fontweight("bold")
    ax.set_xlim(0, 0.58)
    ax.set_xlabel("Character error rate  —  lower is better", fontsize=10)
    ax.set_title("Character error rate", fontsize=15, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.035, "100 word crops per dataset, same machine",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", path)


def tradeoff_chart(filename):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=200)
    for ax, dataset, title in zip(axes, ["funsd", "iiit5k"],
                                  ["FUNSD (documents)", "IIIT5K (scene text)"]):
        for label, acc, _cer, sec, ours in rows_for(dataset):
            ax.scatter(sec, acc * 100, s=190 if ours else 110,
                       color=OURS if ours else RIVAL,
                       edgecolor="white", linewidth=1.4, zorder=3)
            ax.annotate(label, (sec, acc * 100), textcoords="offset points",
                        xytext=(9, 6), fontsize=8.6,
                        fontweight="bold" if ours else "normal",
                        color=OURS_DARK if ours else RIVAL_TOP)
        ax.set_xscale("log")
        ax.set_xlabel("Seconds per crop (log scale)  —  lower is better", fontsize=9.5)
        ax.set_ylabel("Word accuracy (%)", fontsize=9.5)
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
        ax.set_ylim(25, 108)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Accuracy against speed", fontsize=15, fontweight="bold", x=0.007, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(IMG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", path)


if __name__ == "__main__":
    accuracy_chart("funsd", "FUNSD — scanned form text",
                   "100 word crops, single machine, blue bars are Chorus",
                   "benchmark_funsd.png")
    accuracy_chart("iiit5k", "IIIT5K — scene text",
                   "100 word crops, single machine, blue bars are Chorus",
                   "benchmark_iiit5k.png")
    cer_chart("benchmark_cer.png")
    tradeoff_chart("benchmark_tradeoff.png")
    print("CHARTS_DONE")
