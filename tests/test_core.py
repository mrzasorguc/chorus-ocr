"""Fast tests that do not download OCR models."""

import unittest

from chorus import consensus, lang


class LanguageTests(unittest.TestCase):
    def test_scene_number_cleanup(self):
        self.assertEqual(lang.polish("3,642,039", mode="scene"), "3642039")

    def test_empty_text(self):
        self.assertEqual(lang.polish("   "), "")


class ConsensusTests(unittest.TestCase):
    def test_empty_hypotheses(self):
        result = consensus.fuse([])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)

    def test_scene_agreement(self):
        hypotheses = [
            {"text": "Chorus", "conf": 0.9, "weight": 1.0, "src": "easyocr:orig"},
            {"text": "Chorus", "conf": 0.8, "weight": 0.9, "src": "paddle:orig"},
        ]
        result = consensus.fuse(hypotheses, mode="scene")
        self.assertEqual(result["text"], "Chorus")


if __name__ == "__main__":
    unittest.main()
