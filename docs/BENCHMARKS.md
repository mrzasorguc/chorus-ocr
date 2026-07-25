# Benchmarks

All numbers below were produced on one machine with the scripts in this
repository. Nothing is copied from a paper or a vendor page.

## How to reproduce

```bash
python scripts/benchmark_datasets.py funsd  --n 100 --engines easyocr,paddle,tesseract     --profile interactive
python scripts/benchmark_datasets.py funsd  --n 100 --engines easyocr,paddle,tesseract,got --profile interactive
python scripts/benchmark_datasets.py funsd  --n 100 --engines easyocr,paddle,tesseract,got --profile quality
```

Swap `funsd` for `iiit5k` to run the scene-text set.

## Test machine

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB VRAM |
| OS | Windows 11 |
| Python | 3.13.5 |

## Metrics

- **Word accuracy** — share of crops read exactly right after normalization.
- **CER** — character error rate, lower is better.
- **Sec** — average wall-clock seconds per crop after warm-up. Model loading is
  excluded because it happens once per process, not once per image.

## Protocol: how we avoid tuning on the test set

Every dataset is split into two disjoint parts by the `--skip` flag.

| Split | Range | Used for |
| --- | --- | --- |
| Test | `skip=0, n=100` | The published numbers below. Never used for tuning. |
| Tuning | `skip=1000, n=60` | Choosing fusion strategy and engine weights. |

Fusion strategies and per-engine weights were selected on the tuning split only,
with a two-fold holdout check, and were then measured once on the untouched test
split. No setting was ever chosen because it scored well on the test split.

## Charts

Generated from the JSON results with `python scripts/make_charts.py`.

![FUNSD word accuracy](images/benchmark_funsd.png)

![IIIT5K word accuracy](images/benchmark_iiit5k.png)

![Character error rate](images/benchmark_cer.png)

![Accuracy against speed](images/benchmark_tradeoff.png)

## Results — FUNSD (scanned forms, 100 word crops)

| System | Word acc | CER | Sec |
| --- | --- | --- | --- |
| EasyOCR alone | 0.33 | 0.4666 | 0.03 |
| PaddleOCR alone | 0.41 | 0.4398 | 0.24 |
| Tesseract alone | 0.56 | 0.2282 | 0.34 |
| GOT-OCR 2.0 alone | 0.71 | 0.4459 | 2.37 |
| **Chorus — Standard Fusion** | **0.61** | 0.2156 | 0.62 |
| **Chorus — Maximum Performance** | **0.75** | 0.3954 | 3.02 |
| **Chorus — Maximum + Quality** | **0.80** | **0.1496** | 7.77 |

## Results — IIIT5K (scene text, 100 word crops)

| System | Word acc | CER | Sec |
| --- | --- | --- | --- |
| EasyOCR alone | 0.72 | 0.2082 | 0.03 |
| PaddleOCR alone | 0.73 | 0.2477 | 0.21 |
| Tesseract alone | 0.66 | 0.2505 | 0.32 |
| GOT-OCR 2.0 alone | 0.93 | 0.0427 | 2.08 |
| **Chorus — Standard Fusion** | **0.92** | 0.0759 | 0.67 |
| **Chorus — Maximum Performance** | **0.97** | **0.0300** | 3.26 |
| **Chorus — Maximum + Quality** | 0.95 | 0.0342 | 5.63 |

## What these numbers do and do not show

**They show** that fusion beats every engine it is built from. In each mode,
Chorus scores higher than any single engine available to that mode: Standard
reaches 0.61 and 0.92 where its best member reaches 0.56 and 0.73, and the
GOT-enabled modes reach 0.80 and 0.97 where GOT alone reaches 0.71 and 0.93.

**They do not show** that Chorus is the strongest OCR system in existence.
Chorus is a fusion layer, so it cannot read anything its member engines all
miss. Systems such as PaddleOCR-VL report higher scores on their own benchmarks,
and those benchmarks measure full-page document parsing rather than the word
crops measured here. The two sets of numbers are not comparable, and we do not
present them as if they were.

### On Maximum + Quality scoring below Maximum on IIIT5K

Exhaustive test-time augmentation helps on degraded document text, where it
lifts FUNSD from 0.75 to 0.80 and cuts CER from 0.3954 to 0.1496. On clean scene
text it adds hypotheses that occasionally outvote a correct reading, which costs
two points on IIIT5K. Use Maximum for scene photos and Maximum + Quality for
scanned or degraded documents.

## Speed: concurrent engine execution

Engines do not depend on each other, so they run in parallel threads. The heavy
work inside each engine happens in native code that releases the GIL, so the
threads overlap real compute. Set `CHORUS_PARALLEL_ENGINES=0` to force the old
sequential path.

The fused text is byte-identical either way. `scripts/check_parallel.py` runs
every test image through both paths and compares the output and the hypothesis
sources; it reports zero mismatches in all three modes.

Measured on the FUNSD test split, 100 crops:

| Mode | Sequential | Parallel | Change |
| --- | --- | --- | --- |
| Standard Fusion | 0.62 s | 0.53 s | 1.17x faster |
| Maximum Performance | see note | see note | roughly 1.1x |

The honest caveat on Maximum mode: the parallel run reported 3.24 s against a
sequential 3.02 s, but GOT alone also rose from 2.37 s to 2.91 s in that same
run, so the machine was simply slower at that moment. The comparison that
survives the noise is the overhead the fusion layer adds on top of its slowest
engine, which fell from +27% to +11%.

The ceiling here is low by construction. In Maximum mode GOT accounts for about
four fifths of the work, so even perfect overlap of the other three engines can
only remove what they cost. Parallelism helps most in Standard mode, where the
three engines have comparable runtimes.

Because engines are now entered from worker threads, each engine holds its own
lock. Model construction happens exactly once, and a single engine is never
re-entered concurrently even when a web server overlaps requests. Different
engines still run at the same time.

## Published results from other systems

These come from vendor and research publications. They were measured on
different data, on different hardware, by other people. They are listed for
context only and are **not** a head-to-head comparison with the tables above.

| System | Reported score | Benchmark |
| --- | --- | --- |
| PaddleOCR-VL 1.6 | 96.33 | OmniDocBench v1.6 |
| MinerU 2.5 Pro | 95.69 | OmniDocBench composite |
| GLM-OCR | 94.62 | OmniDocBench composite |
| Mistral OCR 4 | 93.07 | OmniDocBench |
| Mistral OCR 4 | 85.20 | olmOCR-bench |
| Surya OCR 2 | 83.30 | olmOCR-bench |

## Cloud pricing, for cost context

| Service | Price |
| --- | --- |
| Google Enterprise OCR | $1.50 per 1,000 pages ($0.60 above 5M/month) |
| Azure Read | $1.50 per 1,000 pages ($0.60 above 1M/month) |
| AWS Textract | $0.0015 per page |
| Mistral OCR | $4.00 per 1,000 pages ($2.00 batch) |
| **Chorus** | **Free, runs on your own machine** |

We hold no API credentials for the cloud services, so no head-to-head run
against them appears in this document.
