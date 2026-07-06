import os
import json
import logging
import pandas as pd
from src.features.eda_utils import DatasetAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_eda():
    raw_dir = os.path.join("data", "raw")
    if not os.path.exists(raw_dir):
        logger.error("No raw data directory found. Please run download_datasets.py first.")
        return

    report = {}
    
    # 1. Analyze HOLD-Telugu
    hold_file = os.path.join(raw_dir, "HOLD-Telugu_train.parquet")
    if os.path.exists(hold_file):
        logger.info("Analyzing HOLD-Telugu...")
        df = pd.read_parquet(hold_file)
        # Using the actual text column from DravidianLangTech
        text_col = 'Comments' if 'Comments' in df.columns else df.columns[1]
        
        analyzer = DatasetAnalyzer(df, "HOLD-Telugu", text_col)
        report["HOLD-Telugu"] = {
            "overview": analyzer.dataset_overview(),
            "text_analysis": analyzer.text_analysis(),
            "quality": analyzer.quality_analysis()
        }
        analyzer.generate_visualizations()
    else:
        logger.warning(f"{hold_file} not found.")

    # 2. Analyze Telugu Alpaca Romanized
    alpaca_file = os.path.join(raw_dir, "Telugu-Alpaca-Romanized_train.parquet")
    if os.path.exists(alpaca_file):
        logger.info("Analyzing Telugu Alpaca Romanized...")
        df = pd.read_parquet(alpaca_file)
        # It's an instruction dataset, usually 'instruction', 'input', 'output'
        # We'll analyze the output/response column since that contains the Romanized Telugu
        text_col = 'output' if 'output' in df.columns else ('response' if 'response' in df.columns else df.columns[-1])
        
        analyzer = DatasetAnalyzer(df, "Telugu-Alpaca", text_col)
        report["Telugu-Alpaca"] = {
            "overview": analyzer.dataset_overview(),
            "text_analysis": analyzer.text_analysis(),
            "quality": analyzer.quality_analysis()
        }
        analyzer.generate_visualizations()
    else:
        logger.warning(f"{alpaca_file} not found.")
        
    # 3. Analyze Telugu Sentiment
    senti_file = os.path.join(raw_dir, "Telugu-Sentiment_train.parquet")
    if os.path.exists(senti_file):
        logger.info("Analyzing Telugu Sentiment...")
        df = pd.read_parquet(senti_file)
        text_col = 'Sentence' if 'Sentence' in df.columns else df.columns[1]
        
        analyzer = DatasetAnalyzer(df, "Telugu-Sentiment", text_col)
        report["Telugu-Sentiment"] = {
            "overview": analyzer.dataset_overview(),
            "text_analysis": analyzer.text_analysis(),
            "quality": analyzer.quality_analysis()
        }
        analyzer.generate_visualizations()
    else:
        logger.warning(f"{senti_file} not found.")

    # Save complete report
    os.makedirs("outputs/eda", exist_ok=True)
    with open(os.path.join("outputs", "eda", "full_eda_report.json"), "w") as f:
        # custom serializer for numpy types
        def default(o):
            if isinstance(o, (np.int64, np.int32)): return int(o)
            if isinstance(o, (np.float64, np.float32)): return float(o)
            raise TypeError
        import numpy as np
        json.dump(report, f, indent=4, default=default)
        
    logger.info("EDA Pipeline completed successfully.")

if __name__ == "__main__":
    run_full_eda()
