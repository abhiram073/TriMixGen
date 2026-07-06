import unittest
import pandas as pd
from src.features.unicode_utils import UnicodeUtils

class TestUnicodeUtils(unittest.TestCase):

    def test_script_detection_telugu_only(self):
        text = "ఇది తెలుగు వాక్యం"
        self.assertEqual(UnicodeUtils.detect_script(text), "Telugu")

    def test_script_detection_english_only(self):
        text = "This is pure English."
        self.assertEqual(UnicodeUtils.detect_script(text), "Latin")

    def test_script_detection_mixed(self):
        text = "movie చాలా బాగుంది bro!"
        self.assertEqual(UnicodeUtils.detect_script(text), "Mixed")

    def test_script_detection_numbers_and_punctuation(self):
        text = "12345 !@#$%"
        self.assertEqual(UnicodeUtils.detect_script(text), "Unknown")

    def test_script_detection_emojis(self):
        text = "😂❤️"
        self.assertEqual(UnicodeUtils.detect_script(text), "Unknown")
        
    def test_script_detection_empty(self):
        self.assertEqual(UnicodeUtils.detect_script(""), "Unknown")
        self.assertEqual(UnicodeUtils.detect_script(None), "Unknown")

    def test_validate_encoding(self):
        valid = "Valid string"
        malformed = "Malformed string \ufffd with replacement char"
        self.assertTrue(UnicodeUtils.validate_encoding(valid))
        self.assertFalse(UnicodeUtils.validate_encoding(malformed))

    def test_analyze_proportions(self):
        text = "Aa బా 12 ! 😂"
        # Latin: 'A', 'a' = 2
        # Telugu: 'బ', 'ా' = 2 (Consonant + Vowel sign)
        # Digits: '1', '2' = 2
        # Punctuation: '!' = 1
        # Other: '😂' = 1
        # Total = 8 non-whitespace characters
        props = UnicodeUtils.analyze_proportions(text)
        self.assertAlmostEqual(props['latin'], 2/8)
        self.assertAlmostEqual(props['telugu'], 2/8)
        self.assertAlmostEqual(props['digits'], 2/8)
        self.assertAlmostEqual(props['punctuation'], 1/8)
        self.assertAlmostEqual(props['other'], 1/8)

    def test_generate_report(self):
        df = pd.DataFrame({
            "text": [
                "pure english",
                "పూర్తి తెలుగు",
                "mixed భాష",
                "just \ufffd malformed",
                "123"
            ]
        })
        report = UnicodeUtils.generate_report(df, "text")
        
        self.assertEqual(report["total_samples"], 5)
        self.assertEqual(report["malformed_samples"], 1)
        self.assertEqual(report["script_distribution"]["Latin"], 0.4) # 'pure english', 'just malformed'
        self.assertEqual(report["script_distribution"]["Telugu"], 0.2)
        self.assertEqual(report["script_distribution"]["Mixed"], 0.2)
        self.assertEqual(report["script_distribution"]["Unknown"], 0.2) # '123'

if __name__ == '__main__':
    unittest.main()
