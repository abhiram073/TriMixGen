import os
import pandas as pd
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

def download_dravidian_codemix(output_dir: str):
    """
    Downloads the DravidianCodeMix Telugu dataset from HuggingFace.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Downloading DravidianCodeMix Telugu dataset...")
    
    try:
        # Note: Using a popular community upload of the DravidianLangTech dataset
        dataset = load_dataset("appalaraju/telugu_english_codemixed_sentiment")
        
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            output_path = os.path.join(output_dir, f"dravidian_telugu_{split}.csv")
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {split} split to {output_path} (Shape: {df.shape})")
            
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    # Ensure the script runs from the project root
    raw_data_dir = os.path.join("data", "raw")
    download_dravidian_codemix(raw_data_dir)
