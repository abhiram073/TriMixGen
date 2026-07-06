import os
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/Telugu-Alpaca-Romanized_train.parquet"
OUTPUT_DIR = "data/processed/gen_001"
SEED = 42

def prepare_dataset():
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    df = pd.read_parquet(RAW_DATA_PATH)
    
    initial_len = len(df)
    
    # 1. Clean Data
    logger.info("Cleaning dataset...")
    # Drop empty rows
    # Alpaca dataset typically has 'instruction', 'input', 'output'
    for col in ['instruction', 'output']:
        if col in df.columns:
            df = df.dropna(subset=[col])
            df = df[df[col].str.strip() != ""]
            
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop malformed (e.g. outputs too short or incredibly long)
    if 'output' in df.columns:
        df = df[df['output'].str.len() > 5]
        df = df[df['output'].str.len() < 2000]
        
    cleaned_len = len(df)
    logger.info(f"Removed {initial_len - cleaned_len} malformed/duplicate rows. {cleaned_len} remain.")

    # 2. Split Data (80/10/10)
    logger.info("Splitting dataset into 80/10/10 (Train/Valid/Test) with seed 42")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)
    
    # 3. Save processed datasets
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(out_path / "train.parquet", index=False)
    valid_df.to_parquet(out_path / "valid.parquet", index=False)
    test_df.to_parquet(out_path / "test.parquet", index=False)
    
    # 4. Save metadata
    metadata = {
        "random_seed": SEED,
        "split_ratios": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "dataset_version": "1.0",
        "preprocessing_version": "gen_001_v1",
        "creation_timestamp": datetime.now().isoformat(),
        "sizes": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
            "total_cleaned": cleaned_len,
            "total_raw": initial_len
        }
    }
    
    with open(out_path / "split_metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
        
    # 5. Generate Reports
    generate_reports(train_df, valid_df, test_df, initial_len, cleaned_len, out_path)
    
def generate_reports(train_df, valid_df, test_df, initial_len, cleaned_len, out_path):
    # Dataset Statistics
    stats_md = f"""# GEN_001 Dataset Statistics
    
## Cleaning Pipeline
- **Raw Samples:** {initial_len}
- **Cleaned Samples:** {cleaned_len}
- **Discarded:** {initial_len - cleaned_len} (duplicates, empty, malformed)

## Splits
- **Train (80%):** {len(train_df)}
- **Validation (10%):** {len(valid_df)}
- **Test (10%):** {len(test_df)}
"""
    with open(out_path / "dataset_statistics.md", "w") as f:
        f.write(stats_md)
        
    # Split Summary
    split_md = f"""# GEN_001 Split Summary

The dataset was statically split with `random_seed=42`.
These splits are frozen and MUST be reused for GEN_002 and GEN_003.

- Train: `data/processed/gen_001/train.parquet`
- Valid: `data/processed/gen_001/valid.parquet`
- Test: `data/processed/gen_001/test.parquet`
"""
    with open(out_path / "split_summary.md", "w") as f:
        f.write(split_md)
        
    # Vocabulary Summary (Basic character/word count approximation)
    if 'output' in train_df.columns:
        words = " ".join(train_df['output'].astype(str).tolist()).split()
        unique_words = len(set(words))
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        
        vocab_md = f"""# Vocabulary Summary (Approximation)

- **Total Words (Tokens) in Train Output:** {len(words):,}
- **Unique Words (Heuristic):** {unique_words:,}
- **Average Word Length:** {avg_len:.2f} characters

*Note: True subword vocabulary is managed by the SentencePiece mT5 Tokenizer (250,112 tokens).*
"""
        with open(out_path / "vocabulary_summary.md", "w") as f:
            f.write(vocab_md)
            
    logger.info("Dataset preparation complete.")

if __name__ == "__main__":
    prepare_dataset()
