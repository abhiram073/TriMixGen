import logging
import time
import pandas as pd
from typing import Dict, Any, List
import yaml
from pathlib import Path

from src.features.validator import DatasetValidator
from src.features.unicode_utils import UnicodeUtils
from src.features.normalizer import TextNormalizer
from src.features.emoji_handler import EmojiHandler
from src.features.language_annotator import LanguageAnnotator

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    End-to-End Orchestrator for the TriMixGen Preprocessing Phase.
    Coordinates validation, Unicode analysis, normalization, emoji handling, and language annotation.
    """
    def __init__(self, config_path: str = "configs/preprocessing.yaml"):
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load {config_path}, using defaults. Error: {e}")
            self.config = {}

        # Initialize modular components
        # (Validator is instantiated per-dataset in process_dataset)
        norm_cfg = self.config.get('normalizer', {})
        self.normalizer = TextNormalizer(**norm_cfg)
        self.emoji_handler = EmojiHandler(config_path=config_path)
        self.annotator = LanguageAnnotator(config_path=config_path)

    def process_dataset(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        dataset_name: str = "",
        required_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline on a DataFrame.
        Returns a dictionary containing the processed data and comprehensive metrics.
        """
        report = {
            "execution_times": {},
            "initial_rows": len(df),
            "unicode_report": {},
            "emoji_statistics": {},
            "annotation_report": {},
            "errors": []
        }
        
        required_columns = required_columns or [text_column]
        validator = DatasetValidator(required_columns=required_columns, text_column=text_column)
        
        # 1. Validation
        start_time = time.time()
        try:
            df = validator.validate_and_clean(df)
            report["execution_times"]["validation_sec"] = round(time.time() - start_time, 4)
            report["valid_rows"] = len(df)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            report["errors"].append(f"Validation Error: {e}")
            return {"report": report, "high_confidence_df": pd.DataFrame(), "manual_review_df": pd.DataFrame()}

        # 2. Unicode Analysis
        start_time = time.time()
        try:
            report["unicode_report"] = UnicodeUtils.generate_report(df, text_column)
            report["execution_times"]["unicode_analysis_sec"] = round(time.time() - start_time, 4)
        except Exception as e:
            logger.error(f"Unicode analysis failed: {e}")
            report["errors"].append(f"Unicode Error: {e}")

        # 3 & 4. Normalization and Emoji Handling
        start_time = time.time()
        try:
            # Process text row by row
            processed_texts = []
            for text in df[text_column]:
                # Normalize text first (removes HTML, standardizes spacing, etc.)
                norm_text = self.normalizer.normalize(str(text))
                # Handle emojis (convert to tags/remove based on config)
                clean_text = self.emoji_handler.handle(norm_text)
                processed_texts.append(clean_text)
                
            df[f"{text_column}_clean"] = processed_texts
            
            # Extract emoji dataset statistics
            report["emoji_statistics"] = self.emoji_handler.get_statistics(df[text_column].astype(str).tolist())
            report["execution_times"]["normalization_and_emoji_sec"] = round(time.time() - start_time, 4)
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            report["errors"].append(f"Normalization Error: {e}")
            return {"report": report, "high_confidence_df": pd.DataFrame(), "manual_review_df": pd.DataFrame()}

        # 5. Language Annotation
        start_time = time.time()
        try:
            # Simple whitespace tokenization for the heuristic annotator
            tokenized_corpus = [text.split() for text in df[f"{text_column}_clean"]]
            
            # Execute Annotator Pipeline
            annotation_results = self.annotator.process_dataset(tokenized_corpus, dataset_name=dataset_name)
            
            report["annotation_report"] = annotation_results["report"]
            report["execution_times"]["language_annotation_sec"] = round(time.time() - start_time, 4)
            
            # Segregate datasets based on the annotator's confidence scoring
            # We must map the annotated output back to DataFrames
            
            # Create a list of original rows mapped to annotation outputs
            # Since annotation maintains sequential order with tokenized_corpus, we can iterate jointly
            high_conf_rows = []
            low_conf_rows = []
            
            # The annotator splits high/low based on confidence, but we need the original dataframe rows.
            # Instead of using the dicts grouped by annotator, we can evaluate row by row:
            for idx, (original_idx, row) in enumerate(df.iterrows()):
                tokens = tokenized_corpus[idx]
                if not tokens:
                    continue # Skip empty sentences post-cleaning
                
                ann_res = self.annotator.annotate_sentence(tokens, dataset_name=dataset_name)
                
                # Combine row data with annotation metadata
                row_dict = row.to_dict()
                row_dict["tokens"] = ann_res["tokens"]
                row_dict["labels"] = ann_res["labels"]
                row_dict["confidences"] = ann_res["confidences"]
                row_dict["avg_confidence"] = ann_res["avg_confidence"]
                row_dict["traces"] = ann_res["traces"]
                
                if ann_res["is_high_confidence"]:
                    high_conf_rows.append(row_dict)
                else:
                    low_conf_rows.append(row_dict)
                    
            high_conf_df = pd.DataFrame(high_conf_rows)
            low_conf_df = pd.DataFrame(low_conf_rows)
            
            report["high_confidence_samples"] = len(high_conf_df)
            report["manual_review_samples"] = len(low_conf_df)
            
        except Exception as e:
            logger.error(f"Language annotation failed: {e}")
            report["errors"].append(f"Annotation Error: {e}")
            high_conf_df = pd.DataFrame()
            low_conf_df = pd.DataFrame()
            
        report["execution_times"]["total_pipeline_sec"] = sum(report["execution_times"].values())
        
        return {
            "report": report,
            "high_confidence_df": high_conf_df,
            "manual_review_df": low_conf_df
        }

    def save_results(self, output: Dict[str, Any], output_dir: str, dataset_name: str):
        """Saves datasets and reports to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        if not output["high_confidence_df"].empty:
            output["high_confidence_df"].to_parquet(out_path / f"{dataset_name}_high_conf.parquet")
            
        if not output["manual_review_df"].empty:
            output["manual_review_df"].to_parquet(out_path / f"{dataset_name}_manual_review.parquet")
            
        with open(out_path / f"{dataset_name}_pipeline_report.yaml", "w") as f:
            yaml.dump(output["report"], f, sort_keys=False)
