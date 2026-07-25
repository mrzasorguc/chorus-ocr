"""Adapters for EasyOCR, PaddleOCR, Tesseract, and GOT-OCR 2.0."""
import os
import threading
import numpy as np
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_easy = None
_paddle = None
_got = None
_got_failed = False
_tess_lang = None

# The pipeline runs engines in parallel threads, and a web server may overlap
# several requests on top of that. Each engine owns a lock so its lazily built
# model is created exactly once and never re-entered concurrently. Separate
# locks keep different engines running at the same time, which is the point of
# the parallel path.
_easy_lock = threading.Lock()
_paddle_lock = threading.Lock()
_got_lock = threading.Lock()
_tess_lock = threading.Lock()

def easyocr_read(img):
    global _easy
    import easyocr, torch
    with _easy_lock:
        if _easy is None:
            _easy = easyocr.Reader(["tr", "en"], gpu=torch.cuda.is_available(), verbose=False)
        res = _easy.readtext(img)
    res.sort(key=lambda r: (min(p[1] for p in r[0]) // 20, min(p[0] for p in r[0])))
    texts = [t for _, t, _ in res]
    confs = [c for _, _, c in res]
    return {"text": " ".join(texts), "conf": float(np.mean(confs)) if confs else 0.0}

def paddle_read(img):
    global _paddle
    from paddleocr import PaddleOCR
    with _paddle_lock:
        if _paddle is None:
            _paddle = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        out = _paddle.predict(img)
    if not out:
        return {"text": "", "conf": 0.0}
    r = out[0]
    texts = list(r["rec_texts"])
    scores = list(r["rec_scores"])
    return {"text": " ".join(texts), "conf": float(np.mean(scores)) if scores else 0.0}

def _find_tesseract():
    global _tess_lang
    if _tess_lang is not None:
        return _tess_lang
    with _tess_lock:
        # Another thread may have finished the lookup while this one waited.
        if _tess_lang is not None:
            return _tess_lang
        return _locate_tesseract()

def _locate_tesseract():
    global _tess_lang
    import pytesseract, shutil
    package_dir = os.path.dirname(os.path.abspath(__file__))
    tessdata_candidates = [
        os.path.join(package_dir, "tessdata"),
        os.path.join(os.path.dirname(package_dir), "tessdata"),
    ]
    for local_td in tessdata_candidates:
        if os.path.exists(os.path.join(local_td, "tur.traineddata")):
            os.environ["TESSDATA_PREFIX"] = local_td
            break
    cands = [shutil.which("tesseract"),
             r"C:\Program Files\Tesseract-OCR\tesseract.exe",
             r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for c in cands:
        if c and os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            try:
                langs = pytesseract.get_languages(config="")
            except Exception:
                langs = ["eng"]
            _tess_lang = "tur+eng" if "tur" in langs else "eng"
            return _tess_lang
    _tess_lang = ""
    return ""

def tesseract_read(img):
    import pytesseract, cv2
    lang = _find_tesseract()
    if not lang:
        raise RuntimeError("tesseract binary yok")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    psm = 8 if max(h, w) < 220 else 7
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(rgb, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for word, c in zip(data["text"], data["conf"]):
        if str(word).strip() and float(c) >= 0:
            words.append(str(word))
            confs.append(float(c) / 100.0)
    if not words:
        data = pytesseract.image_to_data(rgb, lang=lang, output_type=pytesseract.Output.DICT)
        for word, c in zip(data["text"], data["conf"]):
            if str(word).strip() and float(c) >= 0:
                words.append(str(word))
                confs.append(float(c) / 100.0)
    return {"text": " ".join(words), "conf": float(np.mean(confs)) if confs else 0.0}

def got_read(img):
    global _got, _got_failed
    if _got_failed:
        raise RuntimeError("GOT-OCR kullanilamiyor")
    import torch, cv2
    from PIL import Image
    import torch as T
    with _got_lock:
        if _got_failed:
            raise RuntimeError("GOT-OCR kullanilamiyor")
        try:
            if _got is None:
                from transformers import AutoProcessor, AutoModelForImageTextToText
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                model = AutoModelForImageTextToText.from_pretrained(
                    "stepfun-ai/GOT-OCR-2.0-hf", torch_dtype=dtype).to(device).eval()
                proc = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
                _got = (model, proc, device, dtype)
        except Exception:
            _got_failed = True
            raise
        model, proc, device, dtype = _got
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = proc(pil, return_tensors="pt")
        inputs = {k: (v.to(device, dtype) if v.dtype == T.float32 else v.to(device)) for k, v in inputs.items()}
        with T.no_grad():
            gen = model.generate(**inputs, do_sample=False, tokenizer=proc.tokenizer,
                                 stop_strings="<|im_end|>", max_new_tokens=1024)
        text = proc.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"text": text.strip(), "conf": 0.9}

ENGINES = {"easyocr": easyocr_read, "paddle": paddle_read,
           "tesseract": tesseract_read, "got": got_read}
# Base reliability values; route-specific multipliers live in pipeline.py.
RELIABILITY = {"got": 1.8, "paddle": 1.2, "easyocr": 1.0, "tesseract": 0.65}
