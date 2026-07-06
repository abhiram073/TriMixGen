import os
import re
import yaml
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "gen_003"
PROCESSED_DIR = Path(f"data/processed/{EXPERIMENT_NAME}")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Common english words used in Tenglish
ENGLISH_WORDS = {"super", "good", "bad", "movie", "song", "hero", "director", "acting", "story", "bgm", "time", "waste", "hit", "flop"}

def infer_formality(text: str) -> str:
    # Simple heuristic for formality
    text_lower = text.lower()
    if "andi" in text_lower or "garu" in text_lower or "meeruu" in text_lower:
        return "Use a formal and respectful tone."
    return "Use a casual, conversational tone."

def infer_english_usage(text: str) -> str:
    # Simple heuristic: count occurrences of known english words
    text_lower = text.lower()
    words = set(text_lower.split())
    overlap = len(words.intersection(ENGLISH_WORDS))
    if overlap >= 2:
        return "Use a high amount of English vocabulary."
    return "Use predominantly Telugu vocabulary."

def get_sentiment_prompt(sentiment: str) -> str:
    if sentiment == "pos":
        return "Write a positive Telugu-English review."
    elif sentiment == "neg":
        return "Write a negative Telugu-English review."
    else:
        return "Write a neutral Telugu-English review."

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        sentence = str(row["Sentence"]).strip()
        sentiment = str(row["Sentiment"]).strip()
        
        # Length filtering (3 to 100 words)
        word_count = len(sentence.split())
        if word_count < 3 or word_count > 100:
            continue
            
        # Infer styles
        sentiment_prompt = get_sentiment_prompt(sentiment)
        formality_prompt = infer_formality(sentence)
        english_prompt = infer_english_usage(sentence)
        
        instruction = f"{sentiment_prompt} {formality_prompt} {english_prompt}"
        
        records.append({
            "instruction": instruction,
            "input": "",
            "output": sentence,
            "sentiment": sentiment
        })
    return pd.DataFrame(records)

def main():
    logger.info("Starting GEN_003 Dataset Preparation...")
    
    # 1. Load Data
    logger.info("Loading raw Telugu-Sentiment dataset...")
    df_train_raw = pd.read_parquet("data/raw/Telugu-Sentiment_train.parquet")
    df_valid_raw = pd.read_parquet("data/raw/Telugu-Sentiment_validation.parquet")
    df_test_raw = pd.read_parquet("data/raw/Telugu-Sentiment_test.parquet")
    
    # Process
    df_train = process_dataset(df_train_raw)
    df_valid = process_dataset(df_valid_raw)
    df_test = process_dataset(df_test_raw)
    
    logger.info(f"Processed training samples: {len(df_train)}")
    
    # 2. Sentiment Balancing (Training Set Only)
    # Find the minimum class count
    class_counts = df_train["sentiment"].value_counts()
    min_class_count = class_counts.min()
    logger.info(f"Balancing training set to {min_class_count} samples per class.")
    
    balanced_dfs = []
    for sentiment in class_counts.index:
        df_subset = df_train[df_train["sentiment"] == sentiment]
        df_subset_sampled = df_subset.sample(n=min_class_count, random_state=42)
        balanced_dfs.append(df_subset_sampled)
        
    df_train_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Balanced training set size: {len(df_train_balanced)}")
    
    # 3. Duplicate Removal
    logger.info("Removing exact duplicates...")
    df_train_balanced = df_train_balanced.drop_duplicates(subset=["instruction", "output"])
    df_valid = df_valid.drop_duplicates(subset=["instruction", "output"])
    df_test = df_test.drop_duplicates(subset=["instruction", "output"])
    
    logger.info(f"Final Train size: {len(df_train_balanced)}")
    logger.info(f"Final Valid size: {len(df_valid)}")
    logger.info(f"Final Test size: {len(df_test)}")
    
    # 4. Save
    df_train_balanced.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    df_valid.to_parquet(PROCESSED_DIR / "valid.parquet", index=False)
    df_test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    
    # 5. Metadata
    metadata = {
        "dataset_name": "Telugu-Sentiment",
        "random_seed": 42,
        "balanced": True,
        "total_train_samples": len(df_train_balanced),
        "total_valid_samples": len(df_valid),
        "total_test_samples": len(df_test)
    }
    with open(PROCESSED_DIR / "split_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
        
    logger.info("GEN_003 Preparation Complete!")

if __name__ == "__main__":
    main()
