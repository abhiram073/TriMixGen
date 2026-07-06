import unittest
from src.features.language_annotator import LanguageAnnotator

class TestLanguageAnnotator(unittest.TestCase):

    def setUp(self):
        # Provide a small test dictionary to mimic NLTK/Custom dict
        self.test_dict = {"movie", "is", "very", "good", "a", "i"}
        self.annotator = LanguageAnnotator(english_dict=self.test_dict)

    def test_tier1_universal(self):
        tests = ["123", "!!!", "<URL>", "<USER>", "<emoji_laughter>"]
        for t in tests:
            tag, conf, trace = self.annotator.annotate_token(t, is_first=True)
            self.assertEqual(tag, 'Univ')
            self.assertEqual(trace, 'Tier_1_Univ')

    def test_tier2_unicode(self):
        tag, conf, trace = self.annotator.annotate_token("బాగుంది", is_first=True)
        self.assertEqual(tag, 'Te')
        self.assertEqual(trace, 'Tier_2_Unicode')

    def test_tier3_ne(self):
        # Mahesh inside sentence -> NE
        tag, conf, trace = self.annotator.annotate_token("Mahesh", is_first=False)
        self.assertEqual(tag, 'NE')
        self.assertEqual(trace, 'Tier_3_NE')

        # Mahesh at start of sentence -> might just be a capitalized word, falls back to Romanized Te
        tag2, conf2, trace2 = self.annotator.annotate_token("Mahesh", is_first=True)
        self.assertEqual(tag2, 'Te')
        self.assertEqual(trace2, 'Tier_6_Romanized')

    def test_tier4_mixed(self):
        # English root 'car' + Telugu suffix 'lo'
        tag, conf, trace = self.annotator.annotate_token("carlo", is_first=True)
        self.assertEqual(tag, 'Mixed')
        self.assertEqual(trace, 'Tier_4_Mixed')

    def test_tier5_dict(self):
        tag, conf, trace = self.annotator.annotate_token("movie", is_first=True)
        self.assertEqual(tag, 'En')
        self.assertEqual(trace, 'Tier_5_Dict')

    def test_tier6_romanized(self):
        # 'bagundi' is > 2 chars and not in english dict
        tag, conf, trace = self.annotator.annotate_token("bagundi", is_first=False)
        self.assertEqual(tag, 'Te')
        self.assertEqual(trace, 'Tier_6_Romanized')

    def test_tier7_ambiguous(self):
        # 'aa' is <= 2 chars and not in dict
        tag, conf, trace = self.annotator.annotate_token("aa", is_first=False, prev_tag='Te')
        self.assertEqual(tag, 'Te') # context resolves to Te
        self.assertEqual(trace, 'Tier_7_Ambiguous')

    def test_annotate_sentence(self):
        # "movie చాలా bagundi"
        tokens = ["movie", "చాలా", "bagundi"]
        res = self.annotator.annotate_sentence(tokens)
        
        self.assertListEqual(res['labels'], ['En', 'Te', 'Te'])
        self.assertListEqual(res['traces'], ['Tier_5_Dict', 'Tier_2_Unicode', 'Tier_6_Romanized'])
        self.assertTrue(res['is_high_confidence']) # (0.9 + 1.0 + 0.7) / 3 = 0.86 > 0.75

    def test_dataset_processing(self):
        corpus = [
            ["movie", "చాలా", "bagundi"], # high conf
            ["aa", "o", "eh"] # low conf (all ambiguous/fallback)
        ]
        out = self.annotator.process_dataset(corpus)
        
        self.assertEqual(len(out["high_confidence_set"]), 1)
        self.assertEqual(len(out["manual_review_set"]), 1)
        self.assertEqual(out["report"]["total_sentences"], 2)
        self.assertEqual(out["report"]["total_tokens"], 6)

if __name__ == '__main__':
    unittest.main()
