import sys
import logging
from pathlib import Path
import pandas as pd
import random
import yaml

# Add the project root to sys.path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.features.preprocessing_pipeline import PreprocessingPipeline
from scripts.data_utils import robust_read_dataset

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PreprocessingPipeline(config_path="configs/preprocessing.yaml")
    
    parquet_files = list(data_dir.glob("*.parquet"))
    
    all_reports = {}
    
    for file_path in parquet_files:
        dataset_basename = file_path.stem
        logger.info(f"Processing {file_path.name}...")
        
        try:
            df, text_col = robust_read_dataset(str(file_path))
            logger.info(f"Using text column: '{text_col}'")
            
            # Run the pipeline
            result = pipeline.process_dataset(df, text_column=text_col, dataset_name=dataset_basename)
            
            # Save results
            dataset_basename = file_path.stem
            pipeline.save_results(result, str(output_dir), dataset_basename)
            
            all_reports[dataset_basename] = result["report"]
            logger.info(f"Finished {dataset_basename}. High Conf: {len(result['high_confidence_df'])}, Review: {len(result['manual_review_df'])}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            
    # Sample 100 sentences for manual inspection from HOLD-Telugu_train if available
    hold_high_conf_path = output_dir / "HOLD-Telugu_train_high_conf.parquet"
    if hold_high_conf_path.exists():
        logger.info("Sampling 100 sentences for quality inspection...")
        hold_df = pd.read_parquet(hold_high_conf_path)
        
        sample_size = min(100, len(hold_df))
        sample_df = hold_df.sample(sample_size, random_state=42)
        
        sample_out = output_dir / "HOLD-Telugu_train_sample_100.json"
        
        # We need to drop complex pandas columns like lists (tokens, labels, confidences) for JSON if needed
        # Or just export to JSON directly. Pandas `to_json(orient='records')` works well.
        sample_df.to_json(sample_out, orient='records', indent=4)
        logger.info(f"Saved 100-sentence sample to {sample_out}")

    # Save aggregated report summary
    with open(output_dir / "aggregated_preprocessing_report.yaml", "w") as f:
        yaml.dump(all_reports, f, sort_keys=False)
        
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    run_pipeline()
