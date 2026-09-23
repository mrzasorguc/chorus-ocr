"""Product-mode tests for the public browser demo."""

import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np

from chorus.web import (
    MAXIMUM_MODE,
    MAXIMUM_QUALITY_MODE,
    MODE_CONFIG,
    MODE_ENGINES,
    STANDARD_MODE,
    create_demo,
    recognize,
)


class ProductModeTests(unittest.TestCase):
    def test_exactly_three_modes_exist(self):
        self.assertEqual(
            set(MODE_CONFIG),
            {STANDARD_MODE, MAXIMUM_MODE, MAXIMUM_QUALITY_MODE},
        )

    def test_standard_mode_is_fast_multi_engine(self):
        self.assertEqual(
            MODE_CONFIG[STANDARD_MODE],
            {
                "engines": ("easyocr", "paddle", "tesseract"),
                "profile": "interactive",
            },
        )

    def test_maximum_mode_adds_got(self):
        self.assertEqual(
            MODE_CONFIG[MAXIMUM_MODE],
            {
                "engines": ("easyocr", "paddle", "tesseract", "got"),
                "profile": "interactive",
            },
        )

    def test_maximum_quality_uses_got_and_quality_profile(self):
        self.assertEqual(
            MODE_CONFIG[MAXIMUM_QUALITY_MODE],
            {
                "engines": ("easyocr", "paddle", "tesseract", "got"),
                "profile": "quality",
            },
        )

    def test_easyocr_only_mode_does_not_exist(self):
        self.assertNotIn(("easyocr",), MODE_ENGINES.values())

    @patch("chorus.web.read")
    def test_no_recognized_text_reports_actionable_error(self, read):
        read.return_value = {"text": "", "sources": [], "confidence": 0.0}
        with self.assertRaisesRegex(gr.Error, "No text was recognized"):
            recognize(np.zeros((2, 2, 3), dtype=np.uint8), STANDARD_MODE)

    @patch("chorus.web.read")
    def test_missing_got_keeps_specific_error(self, read):
        read.return_value = {"text": "", "sources": [], "confidence": 0.0}
        with self.assertRaisesRegex(gr.Error, "GOT-OCR could not be started"):
            recognize(np.zeros((2, 2, 3), dtype=np.uint8), MAXIMUM_MODE)

    def test_image_and_mode_changes_clear_every_result_field(self):
        demo = create_demo()
        events = demo.config["dependencies"]
        run = next(event for event in events if event["targets"][0][1] == "click")
        changes = [event for event in events if event["targets"][0][1] == "change"]
        self.assertEqual(len(changes), 2)
        self.assertEqual({event["targets"][0][0] for event in changes}, set(run["inputs"]))
        for event in changes:
            self.assertEqual(event["outputs"], run["outputs"])
            self.assertFalse(event["queue"])


if __name__ == "__main__":
    unittest.main()
