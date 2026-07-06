import unittest
import pandas as pd
from pathlib import Path
import tempfile
import yaml
from src.features.preprocessing_pipeline import PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = PreprocessingPipeline(config_path="configs/preprocessing.yaml")
        
        # Mock raw dataset
        self.raw_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'text': [
                "This is a great movie! 😂",                   # English + Emoji
                "చాలా బాగుంది",                                # Telugu
                "carloki vellu",                             # Mixed morphology
                "  "                                         # Empty -> should be dropped by Validator
            ]
        })

    def test_pipeline_execution(self):
        output = self.pipeline.process_dataset(
            df=self.raw_data, 
            text_column='text', 
            required_columns=['id', 'text']
        )
        
        report = output["report"]
        high_conf_df = output["high_confidence_df"]
        low_conf_df = output["manual_review_df"]
        
        # 1. Validation should have dropped the empty row
        self.assertEqual(report["initial_rows"], 4)
        self.assertEqual(report["valid_rows"], 3)
        self.assertEqual(len(report["errors"]), 0)
        
        # 2. Execution times should be recorded
        self.assertIn("validation_sec", report["execution_times"])
        self.assertIn("total_pipeline_sec", report["execution_times"])
        
        # 3. Unicode Report
        self.assertIn("script_distribution", report["unicode_report"])
        
        # 4. Emoji Statistics
        self.assertEqual(report["emoji_statistics"]["total_emojis_found"], 1) # The 😂
        
        # 5. Annotation Report
        # 3 sentences processed.
        # "this is a great movie <emoji_laughter>" -> high conf (En, En, En, En, En, Univ)
        # "చాలా బాగుంది" -> high conf (Te, Te)
        # "carloki vellu" -> Mixed, Te (Romanized) -> high conf
        # Total processed samples = 3.
        self.assertEqual(len(high_conf_df) + len(low_conf_df), 3)
        
        # Check that dataframe columns were appended correctly
        self.assertIn("tokens", high_conf_df.columns)
        self.assertIn("labels", high_conf_df.columns)
        self.assertIn("avg_confidence", high_conf_df.columns)

    def test_pipeline_error_handling(self):
        # Missing required column 'text' will cause validator to fail
        bad_data = pd.DataFrame({'wrong_col': ["hello"]})
        output = self.pipeline.process_dataset(bad_data, text_column='text', required_columns=['text'])
        
        report = output["report"]
        # Error should be caught gracefully without crashing the app
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("Validation Error", report["errors"][0])
        
        # Dataframes should be empty
        self.assertTrue(output["high_confidence_df"].empty)
        self.assertTrue(output["manual_review_df"].empty)

    def test_save_results(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            output = self.pipeline.process_dataset(self.raw_data, text_column='text')
            
            self.pipeline.save_results(output, tmpdirname, "test_dataset")
            
            # Check if files exist
            p = Path(tmpdirname)
            self.assertTrue((p / "test_dataset_high_conf.parquet").exists())
            self.assertTrue((p / "test_dataset_pipeline_report.yaml").exists())

if __name__ == '__main__':
    unittest.main()
