"""Product-mode tests for the public browser demo."""

import unittest

from chorus.web import (
    MAXIMUM_MODE,
    MAXIMUM_QUALITY_MODE,
    MODE_CONFIG,
    MODE_ENGINES,
    STANDARD_MODE,
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


if __name__ == "__main__":
    unittest.main()
