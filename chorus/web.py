"""Browser-based demo for non-technical Chorus users."""

from __future__ import annotations

import json
from typing import Any

import cv2
import gradio as gr

from .pipeline import read

STANDARD_MODE = "Standard Fusion — Fast multi-engine"
MAXIMUM_MODE = "Maximum Performance — Standard Fusion + GOT-OCR 2.0"
MAXIMUM_QUALITY_MODE = "Maximum + Quality — GOT-OCR + exhaustive TTA"
MODE_CONFIG = {
    STANDARD_MODE: {
        "engines": ("easyocr", "paddle", "tesseract"),
        "profile": "interactive",
    },
    MAXIMUM_MODE: {
        "engines": ("easyocr", "paddle", "tesseract", "got"),
        "profile": "interactive",
    },
    MAXIMUM_QUALITY_MODE: {
        "engines": ("easyocr", "paddle", "tesseract", "got"),
        "profile": "quality",
    },
}
MODE_ENGINES = {name: config["engines"] for name, config in MODE_CONFIG.items()}


def recognize(image: Any, mode: str):
    if image is None:
        raise gr.Error("Please upload an image first / Önce bir görsel yükleyin.")
    if mode not in MODE_CONFIG:
        raise gr.Error("Select an OCR mode / Bir OCR modu seçin.")

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    config = MODE_CONFIG[mode]
    engines = config["engines"]
    profile = config["profile"]
    result = read(
        bgr_image,
        use=engines,
        fast=False,
        verbose=True,
        debug=True,
        profile=profile,
    )

    if mode in {MAXIMUM_MODE, MAXIMUM_QUALITY_MODE} and not any(
        source.startswith("got:") for source in result.get("sources", [])
    ):
        raise gr.Error(
            "GOT-OCR could not be started. Run the installer again and choose "
            "GOT-OCR. It is required for Maximum Performance mode. / "
            "GOT-OCR başlatılamadı. Kurulumu yeniden çalıştırıp GOT-OCR seçeneğini "
            "kurun. Maksimum Performans modu için gereklidir."
        )

    confidence = f"{result['confidence'] * 100:.1f}%"
    route = (
        "Scene text / Sahne yazısı"
        if result.get("route") == "scene"
        else "Document / Belge"
    )
    details = {
        "mode": mode,
        "engines": list(engines),
        "route": result.get("route"),
        "fusion_mode": result.get("mode"),
        "confidence_score": result.get("confidence"),
        "hypothesis_count": result.get("n_hypotheses"),
        "profile": result.get("profile"),
        "processing_seconds": result.get("elapsed_seconds"),
        "engine_seconds": result.get("engine_seconds", {}),
        "low_confidence_words": result.get("low_conf_words", []),
        "sources": result.get("sources", []),
    }
    return (
        result["text"],
        confidence,
        route,
        f"{result.get('elapsed_seconds', 0):.1f} s",
        json.dumps(details, ensure_ascii=False, indent=2),
    )


def create_demo() -> gr.Blocks:
    css = """
    .gradio-container {max-width: 1120px !important; margin: 0 auto !important;}
    .hero {padding: 24px 0 8px;}
    .hero h1 {font-size: 2.25rem !important; margin-bottom: 8px !important;}
    .hero p {font-size: 1.05rem; color: #5f6368; max-width: 760px;}
    .primary-button {min-height: 48px;}
    """
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="md",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    )

    with gr.Blocks(title="Chorus OCR", theme=theme, css=css) as demo:
        gr.HTML(
            """
            <section class="hero">
              <h1>Chorus OCR</h1>
              <p>One image. Multiple OCR engines. One stronger result.<br>
              Tek görsel. Birden fazla OCR motoru. Daha güçlü tek sonuç.</p>
            </section>
            """
        )

        mode = gr.Radio(
            choices=[STANDARD_MODE, MAXIMUM_MODE, MAXIMUM_QUALITY_MODE],
            value=STANDARD_MODE,
            label="OCR mode / OCR modu",
            info=(
                "Maximum Performance adds GOT-OCR. Maximum + Quality also runs exhaustive "
                "image variants and can take several minutes. / Maksimum Performans GOT-OCR "
                "ekler. Maximum + Quality kapsamlı görüntü varyantları kullanır ve birkaç dakika sürebilir."
            ),
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image = gr.Image(
                    label="Image / Görsel",
                    type="numpy",
                    sources=["upload", "clipboard", "webcam"],
                    height=360,
                )
                run_button = gr.Button(
                    "Run Chorus / Chorus'i çalıştır",
                    variant="primary",
                    elem_classes=["primary-button"],
                )
            with gr.Column(scale=5):
                text = gr.Textbox(
                    label="Recognized text / Okunan metin",
                    lines=11,
                    show_copy_button=True,
                )
                with gr.Row():
                    confidence = gr.Textbox(label="Fusion score / Birleşim skoru")
                    route = gr.Textbox(label="Detected type / Algılanan tür")
                    elapsed = gr.Textbox(label="Processing time / İşlem süresi")

        with gr.Accordion("Technical details / Teknik ayrıntılar", open=False):
            details = gr.Code(label="Result details / Sonuç ayrıntıları", language="json")

        gr.Markdown(
            "**Privacy:** Images are processed only for the current request. "
            "Do not upload confidential documents to a public demo.  \n"
            "**Gizlilik:** Herkese açık demoya gizli veya kişisel belge yüklemeyin."
        )
        run_button.click(
            fn=recognize,
            inputs=[image, mode],
            outputs=[text, confidence, route, elapsed, details],
        )
    return demo


def main() -> None:
    create_demo().launch(inbrowser=True)


if __name__ == "__main__":
    main()
