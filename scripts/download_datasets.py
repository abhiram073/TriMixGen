import os
import yaml
import logging
import urllib.request
import pandas as pd
from datasets import load_dataset

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/datasets.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_github_file(name, url, output_dir):
    try:
        ext = url.split('.')[-1].split('?')[0]
        output_path = os.path.join(output_dir, f"{name}.{ext}")
        logger.info(f"Downloading {name} from GitHub: {url}")
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())

        # Convert to parquet for standardized IO
        if ext == 'xlsx':
            df = pd.read_excel(output_path)
        elif ext == 'csv':
            df = pd.read_csv(output_path)
        else:
            raise ValueError(f"Unsupported extension: {ext}")
            
        parquet_path = os.path.join(output_dir, f"{name}.parquet")
        # Ensure all columns are string to avoid arrow conversion issues with mixed types
        df = df.astype(str)
        df.to_parquet(parquet_path)
        logger.info(f"Successfully downloaded and converted {name} to parquet.")
    except Exception as e:
        logger.error(f"Failed to download {name} from GitHub: {e}")

def download_huggingface_dataset(name, repo_id, output_dir):
    try:
        logger.info(f"Downloading {name} from Hugging Face: {repo_id}")
        dataset = load_dataset(repo_id)
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            # Ensure safe string types
            df = df.astype(str)
            output_path = os.path.join(output_dir, f"{name}_{split}.parquet")
            df.to_parquet(output_path, index=False)
        logger.info(f"Successfully downloaded {name} splits to {output_dir}")
    except Exception as e:
        logger.error(f"Critical failure downloading {name} ({repo_id}): {e}")

def main():
    logger.info("Initializing Dataset Download Pipeline...")
    config = load_config()
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    for key, ds_info in config.items():
        if ds_info['source'] == 'huggingface':
            download_huggingface_dataset(ds_info['name'], ds_info['repo_id'], output_dir)
        elif ds_info['source'] == 'github':
            download_github_file(f"{ds_info['name']}_train", ds_info['repo_url'], output_dir)
            if 'test_url' in ds_info:
                download_github_file(f"{ds_info['name']}_test", ds_info['test_url'], output_dir)

    logger.info("Dataset Download Pipeline completed.")

if __name__ == "__main__":
    main()
