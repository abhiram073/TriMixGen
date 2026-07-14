import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatasetStatistics:
    """
    Generates statistics and reports for loaded datasets.
    """
    
    @staticmethod
    def generate_report(df: pd.DataFrame, dataset_name: str, task: str) -> Dict[str, Any]:
        logger.info(f"Generating statistics for {dataset_name} ({task})...")
        
        stats = {
            "dataset_name": dataset_name,
            "task": task,
            "total_rows": len(df),
            "language_pairs": df["language_pair"].value_counts().to_dict() if "language_pair" in df.columns else {},
        }
        
        if task == "language_identification" and "tokens" in df.columns and "labels" in df.columns:
            # Token and label stats
            total_tokens = df["tokens"].apply(len).sum()
            stats["total_tokens"] = int(total_tokens)
            
            # Aggregate all labels to find language distribution
            all_labels = [label for sublist in df["labels"].tolist() for label in sublist]
            stats["label_distribution"] = pd.Series(all_labels).value_counts().to_dict()
            
            # Average sentence length
            stats["avg_sentence_length_tokens"] = float(df["tokens"].apply(len).mean())
            
        elif task == "generation" and "target" in df.columns:
            # Simple word count stats for generation targets
            word_counts = df["target"].apply(lambda x: len(str(x).split()))
            stats["total_target_words"] = int(word_counts.sum())
            stats["avg_target_length_words"] = float(word_counts.mean())
            
        return stats
