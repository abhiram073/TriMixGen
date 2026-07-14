import os
import json
import logging
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from src.data.manager import DatasetManager

logger = logging.getLogger(__name__)

class ExploratoryDataAnalyzer:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on loaded datasets.
    """
    
    def __init__(self, manager: DatasetManager):
        self.manager = manager
        self.reports_dir = "datasets/reports"
        self.plots_dir = "datasets/reports/plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def run_analysis(self, dataset_name: str) -> bool:
        logger.info(f"--- Starting EDA for {dataset_name} ---")
        metadata = self.manager.registry.get_dataset(dataset_name)
        if not metadata:
            logger.error(f"Dataset {dataset_name} not found in registry.")
            return False
            
        try:
            # 1. Load Dataset (Use existing processed if available, or try to load via manager)
            # To avoid re-processing, we assume manager.process_dataset was run, or we run it now if needed.
            # But wait, user said "Load every dataset exclusively through the Dataset Manager."
            # The manager's process_dataset does loading, validation, stats, splitting, export.
            # We can load the splits from the processed folder.
            processed_dir = os.path.join("datasets/processed", dataset_name)
            if not os.path.exists(processed_dir):
                logger.info(f"Dataset {dataset_name} not processed yet. Processing now...")
                success = self.manager.process_dataset(dataset_name)
                if not success:
                    logger.error(f"Cannot perform EDA: Failed to load/process {dataset_name}.")
                    return False
            
            # Load the train split for EDA
            train_path = os.path.join(processed_dir, "train.parquet")
            if not os.path.exists(train_path):
                logger.error(f"Train split not found at {train_path}")
                return False
                
            df = pd.read_parquet(train_path)
            
            # 2. General Statistics & Quality Assessment
            general_stats, quality_issues = self._analyze_general_and_quality(df, dataset_name)
            
            # 3. Task-Specific Analysis (Generation vs LID)
            task = metadata.tasks[0]
            task_stats = {}
            if task == "language_identification":
                task_stats = self._analyze_lid(df, dataset_name)
            elif task == "generation":
                task_stats = self._analyze_generation(df, dataset_name)
                
            # 4. Generate Reports
            self._export_reports(dataset_name, metadata.model_dump(), general_stats, quality_issues, task_stats)
            
            logger.info(f"--- EDA completed successfully for {dataset_name} ---")
            return True
            
        except Exception as e:
            logger.error(f"EDA failed for {dataset_name}: {e}")
            return False

    def _analyze_general_and_quality(self, df: pd.DataFrame, dataset_name: str) -> tuple[Dict, Dict]:
        stats = {
            "total_samples": len(df),
            "duplicate_samples": int(df.duplicated().sum()),
            "missing_values": df.isnull().sum().to_dict(),
        }
        
        quality = {
            "has_duplicates": stats["duplicate_samples"] > 0,
            "has_missing": sum(stats["missing_values"].values()) > 0,
            "suspicious_samples": []
        }
        
        # Check for empty samples or strings
        for col in df.columns:
            if df[col].dtype == object:
                empty_count = (df[col] == "").sum()
                if empty_count > 0:
                    quality["suspicious_samples"].append(f"{empty_count} empty strings found in column '{col}'")
                    
        return stats, quality

    def _analyze_lid(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        # Token and Label Analysis
        all_tokens = [token for tokens_list in df["tokens"] for token in tokens_list]
        all_labels = [label for labels_list in df["labels"] for label in labels_list]
        
        token_lengths = [len(t) for t in df["tokens"]]
        
        stats = {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(set(all_tokens)),
            "label_distribution": pd.Series(all_labels).value_counts().to_dict(),
            "avg_sentence_length": float(pd.Series(token_lengths).mean()),
            "max_sentence_length": int(max(token_lengths)),
            "min_sentence_length": int(min(token_lengths))
        }
        
        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.histplot(token_lengths, bins=50, kde=True)
        plt.title(f"Sentence Length Distribution - {dataset_name}")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.plots_dir, f"{dataset_name}_sentence_length.png"))
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.countplot(x=all_labels, order=pd.Series(all_labels).value_counts().index)
        plt.title(f"Label Distribution - {dataset_name}")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.plots_dir, f"{dataset_name}_label_dist.png"))
        plt.close()
        
        return stats

    def _analyze_generation(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        # Sentence Analysis
        s1_lens = df["sentence_1"].apply(lambda x: len(str(x).split()))
        target_lens = df["target"].apply(lambda x: len(str(x).split()))
        
        stats = {
            "avg_s1_length_words": float(s1_lens.mean()),
            "avg_target_length_words": float(target_lens.mean()),
            "max_target_length": int(target_lens.max())
        }
        
        plt.figure(figsize=(10, 6))
        sns.histplot(target_lens, bins=50, kde=True, color='green')
        plt.title(f"Target Sentence Length Distribution - {dataset_name}")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.plots_dir, f"{dataset_name}_target_length.png"))
        plt.close()
        
        return stats

    def _export_reports(self, dataset_name: str, metadata: Dict, general: Dict, quality: Dict, task: Dict):
        # JSON Reports
        report_data = {
            "metadata": metadata,
            "general_statistics": general,
            "quality_assessment": quality,
            "task_statistics": task
        }
        
        json_path = os.path.join(self.reports_dir, f"{dataset_name}_eda_report.json")
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        # Markdown Report
        md_path = os.path.join(self.reports_dir, f"{dataset_name}_eda_report.md")
        with open(md_path, "w") as f:
            f.write(f"# EDA Report for {dataset_name}\n\n")
            f.write(f"## Metadata\n- Source: {metadata['source']}\n- Tasks: {metadata['tasks']}\n- Combinations: {metadata['supported_combinations']}\n\n")
            
            f.write("## General Statistics\n")
            for k, v in general.items():
                f.write(f"- **{k}**: {v}\n")
                
            f.write("\n## Task Specific Statistics\n")
            for k, v in task.items():
                f.write(f"- **{k}**: {v}\n")
                
            f.write("\n## Quality Assessment\n")
            for k, v in quality.items():
                f.write(f"- **{k}**: {v}\n")
                
            f.write("\n## Recommendations for Phase 4 (Preprocessing)\n")
            if quality.get("has_missing") or quality.get("has_duplicates"):
                f.write("- **Recommendation**: Implement aggressive deduplication and NaN removal in the dataset cleaning pipeline.\n")
            if task.get("label_distribution"):
                f.write("- **Recommendation**: Evaluate class imbalance techniques if OTHER vastly outnumbers valid language tags.\n")
            f.write("- **Recommendation**: Apply Unicode normalization and standard Romanization cleaning routines before training.\n")
            
        logger.info(f"Exported reports to {json_path} and {md_path}")
