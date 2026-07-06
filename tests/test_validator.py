import unittest
import pandas as pd
from src.features.validator import DatasetValidator

class TestDatasetValidator(unittest.TestCase):
    
    def setUp(self):
        self.df_valid = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ["This is a test", "Another valid sentence", "Short"],
            'label': ['En', 'En', 'En']
        })
        
        self.df_invalid = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'text': ["Valid text", None, "   ", "Two words", ""],
            'label': ['En', 'En', 'En', 'En', 'En']
        })

    def test_schema_validation_success(self):
        validator = DatasetValidator(required_columns=['id', 'text', 'label'], text_column='text')
        self.assertTrue(validator.validate_schema(self.df_valid))

    def test_schema_validation_failure(self):
        validator = DatasetValidator(required_columns=['id', 'text', 'nonexistent'], text_column='text')
        self.assertFalse(validator.validate_schema(self.df_valid))
        with self.assertRaises(ValueError):
            validator.validate_and_clean(self.df_valid)

    def test_missing_and_empty_values(self):
        validator = DatasetValidator(required_columns=['text'], text_column='text')
        df_clean = validator.handle_missing_values(self.df_invalid)
        # Should drop None, "   ", and "" (indices 1, 2, 4)
        self.assertEqual(len(df_clean), 2)
        self.assertListEqual(df_clean['text'].tolist(), ["Valid text", "Two words"])

    def test_filter_by_length(self):
        # Require at least 2 words
        validator = DatasetValidator(required_columns=['text'], text_column='text', min_words=2)
        df_clean = validator.filter_by_length(self.df_valid)
        # "Short" should be dropped
        self.assertEqual(len(df_clean), 2)
        self.assertNotIn("Short", df_clean['text'].tolist())

    def test_full_pipeline(self):
        validator = DatasetValidator(required_columns=['id', 'text'], text_column='text', min_words=2)
        df_clean = validator.validate_and_clean(self.df_invalid)
        # Only "Valid text" and "Two words" have >= 2 words and aren't empty
        self.assertEqual(len(df_clean), 2)

if __name__ == '__main__':
    unittest.main()
