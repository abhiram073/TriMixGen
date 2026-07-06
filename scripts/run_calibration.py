import sys
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Add the project root to sys.path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.features.preprocessing_pipeline import PreprocessingPipeline
from scripts.data_utils import robust_read_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_calibration():
    data_dir = Path("data/raw")
    output_dir = Path("data/processed/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PreprocessingPipeline(config_path="configs/preprocessing.yaml")
    
    parquet_files = list(data_dir.glob("*.parquet"))
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    
    calibration_report = {}
    
    for file_path in parquet_files:
        dataset_name = file_path.stem
        logger.info(f"Calibrating {dataset_name}...")
        
        try:
            df, text_col = robust_read_dataset(str(file_path))
            logger.info(f"Resolved text column: '{text_col}' for {dataset_name}")
            
            # Run pipeline to get the annotated data
            result = pipeline.process_dataset(df, text_column=text_col, required_columns=[text_col])
            
            high_df = result["high_confidence_df"]
            low_df = result["manual_review_df"]
            
            # Combine back to get all sentences with their confidences
            all_df = pd.concat([high_df, low_df], ignore_index=True)
            if all_df.empty:
                continue
                
            total_sentences = len(all_df)
            
            # 1. Threshold Experiment
            threshold_stats = {}
            for t in thresholds:
                retained = len(all_df[all_df["avg_confidence"] >= t])
                retained_pct = (retained / total_sentences) * 100 if total_sentences else 0
                threshold_stats[str(t)] = {
                    "retained_count": retained,
                    "retained_pct": round(retained_pct, 2),
                    "manual_review_count": total_sentences - retained
                }
                
            # 2. Visualizations
            # Confidence Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(data=all_df, x="avg_confidence", bins=20, kde=True)
            plt.title(f"Sentence Confidence Distribution: {dataset_name}")
            plt.xlabel("Average Sentence Confidence")
            plt.ylabel("Frequency")
            plt.axvline(x=0.75, color='r', linestyle='--', label='Default Threshold (0.75)')
            plt.legend()
            plt.savefig(output_dir / f"{dataset_name}_confidence_hist.png")
            plt.close()
            
            # Heuristic Trace Usage (Bar Chart)
            # Flatten traces
            all_traces = [trace for traces_list in all_df["traces"] for trace in traces_list]
            trace_series = pd.Series(all_traces)
            plt.figure(figsize=(12, 6))
            sns.countplot(y=trace_series, order=trace_series.value_counts().index)
            plt.title(f"Heuristic Tier Usage: {dataset_name}")
            plt.xlabel("Token Count")
            plt.ylabel("Heuristic Tier")
            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset_name}_heuristic_usage.png")
            plt.close()
            
            # 3. Compile report
            calibration_report[dataset_name] = {
                "total_sentences": total_sentences,
                "threshold_experiment": threshold_stats,
                "heuristic_usage": trace_series.value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to calibrate {dataset_name}: {e}")
            
    # Save Report
    with open(output_dir / "calibration_report.yaml", "w") as f:
        yaml.dump(calibration_report, f, sort_keys=False)
        
    logger.info("Calibration complete.")

if __name__ == "__main__":
    run_calibration()
