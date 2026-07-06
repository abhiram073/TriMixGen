import os
import re
import yaml
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "gen_002"
RAW_DATA_PATH_TRAIN = "data/raw/HOLD-Telugu_train.parquet"
RAW_DATA_PATH_TEST = "data/raw/HOLD-Telugu_test.parquet"
PROCESSED_DIR = Path(f"data/processed/{EXPERIMENT_NAME}")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove excessive punctuation
    text = re.sub(r'([.?!])\1+', r'\1', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_length(text: str) -> bool:
    words = text.split()
    return 3 <= len(words) <= 100

def create_conversational_pairs(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        comment = row["Comments"]
        # Split at first question mark if present
        if '?' in comment:
            parts = comment.split('?', 1)
            instruction = parts[0].strip() + '?'
            output = parts[1].strip()
            if output and is_valid_length(instruction) and is_valid_length(output):
                records.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
        else:
            # Generic conversational instruction
            if is_valid_length(comment):
                records.append({
                    "instruction": "Continue the conversation naturally in Telugu-English:",
                    "input": "",
                    "output": comment
                })
    return pd.DataFrame(records)

def main():
    logger.info("Starting GEN_002 Dataset Preparation...")
    
    # 1. Load Data
    df_train = pd.read_parquet(RAW_DATA_PATH_TRAIN)
    df_test = pd.read_parquet(RAW_DATA_PATH_TEST)
    df = pd.concat([df_train, df_test], ignore_index=True)
    initial_size = len(df)
    logger.info(f"Loaded {initial_size} raw samples from HOLD-Telugu.")
    
    # 2. Toxicity Filtering
    # We remove 'hate' label samples as per the quality filtering requirement.
    df_non_hate = df[df["Label"] != "hate"].copy()
    hate_removed = initial_size - len(df_non_hate)
    
    # 3. Preprocessing
    df_non_hate["Comments"] = df_non_hate["Comments"].apply(preprocess_text)
    
    # 4. Create Q&A Pairs & Filter by length
    qa_df = create_conversational_pairs(df_non_hate)
    
    # 5. Duplicate Detection (Exact match for simplicity instead of heavy LSH)
    before_dedup = len(qa_df)
    qa_df = qa_df.drop_duplicates(subset=["instruction", "output"])
    duplicates_removed = before_dedup - len(qa_df)
    
    final_size = len(qa_df)
    removal_percentage = ((initial_size - final_size) / initial_size) * 100
    
    logger.info(f"Filtering Summary:")
    logger.info(f" - Original dataset size: {initial_size}")
    logger.info(f" - Removed due to toxicity (hate): {hate_removed}")
    logger.info(f" - Removed due to duplicates: {duplicates_removed}")
    logger.info(f" - Retained samples: {final_size}")
    logger.info(f" - Removal percentage: {removal_percentage:.2f}%")
    
    # Generate filtering summary artifact
    summary_md = f"""# GEN_002 Dataset Filtering Summary
* **Original dataset size**: {initial_size}
* **Removed due to toxicity (hate)**: {hate_removed}
* **Removed due to exact duplicates/length**: {duplicates_removed + (len(df_non_hate) - before_dedup)}
* **Retained samples**: {final_size}
* **Removal percentage**: {removal_percentage:.2f}%
"""
    with open(PROCESSED_DIR / "filtering_summary.md", "w") as f:
        f.write(summary_md)
        
    # 6. Train/Valid/Test Split (80/10/10 with seed 42)
    train_df, temp_df = train_test_split(qa_df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    logger.info(f"Train split: {len(train_df)}")
    logger.info(f"Valid split: {len(valid_df)}")
    logger.info(f"Test split: {len(test_df)}")
    
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    valid_df.to_parquet(PROCESSED_DIR / "valid.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    
    # 7. Metadata
    metadata = {
        "dataset_name": "HOLD-Telugu",
        "random_seed": 42,
        "split_ratios": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "total_samples": final_size
    }
    with open(PROCESSED_DIR / "split_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
        
    logger.info("GEN_002 Preparation Complete!")

if __name__ == "__main__":
    main()
