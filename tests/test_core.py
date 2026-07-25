"""Fast tests that do not download OCR models."""

import unittest

from chorus import consensus, lang


class LanguageTests(unittest.TestCase):
    def test_scene_number_cleanup(self):
        self.assertEqual(lang.polish("3,642,039", mode="scene"), "3642039")

    def test_empty_text(self):
        self.assertEqual(lang.polish("   "), "")


class LatticeTests(unittest.TestCase):
    """The document route must fuse below the word, not pick a whole reading."""

    def _hypotheses(self):
        # No engine reads the phrase correctly, but every character is somewhere.
        return [
            {"text": "Total amount duc", "conf": 0.9, "weight": 1.4, "src": "got:orig"},
            {"text": "Total arnount due", "conf": 0.7, "weight": 1.0, "src": "paddle:orig"},
            {"text": "Tota1 amount due", "conf": 0.6, "weight": 0.9, "src": "easyocr:orig"},
        ]

    def test_lattice_can_emit_a_reading_no_engine_produced(self):
        result = consensus.fuse(self._hypotheses(), mode="document")
        readings = {h["text"] for h in self._hypotheses()}
        self.assertEqual(result["text"], "Total amount due")
        self.assertNotIn(result["text"], readings)

    def test_falls_back_to_the_column_vote_without_the_prior(self):
        previous = consensus.PRIOR_ENABLED
        consensus.PRIOR_ENABLED = False
        try:
            result = consensus.fuse(self._hypotheses(), mode="document")
        finally:
            consensus.PRIOR_ENABLED = previous
        self.assertTrue(result["text"])
        self.assertEqual(result["mode"], "lattice")

    def test_prior_stays_out_of_non_words(self):
        # A form code must not be penalised for being absent from a word list.
        self.assertIsNone(consensus._token_prior("12/31/1999"))


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
