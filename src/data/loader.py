import os
import logging
from datasets import load_dataset, Dataset
from src.data.schemas import DatasetMetadata

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Handles downloading and loading of datasets into memory.
    """
    
    @staticmethod
    def load(metadata: DatasetMetadata) -> Dataset:
        """Loads a dataset into a Hugging Face Dataset object."""
        
        # 1. Check if it's already downloaded / processed locally
        processed_path = os.path.join("datasets/processed", metadata.name)
        if os.path.exists(processed_path):
            logger.info(f"Loading {metadata.name} from local processed path: {processed_path}")
            return load_dataset("parquet", data_files=f"{processed_path}/*.parquet", split="train")
            
        # 2. Check if we have raw local files
        if os.path.exists(metadata.local_storage_path) and os.listdir(metadata.local_storage_path):
            logger.info(f"Loading {metadata.name} from local raw path: {metadata.local_storage_path}")
            # Naively assuming parquet or csv for local files
            files = [os.path.join(metadata.local_storage_path, f) for f in os.listdir(metadata.local_storage_path)]
            if files[0].endswith(".csv"):
                return load_dataset("csv", data_files=files, split="train")
            elif files[0].endswith(".parquet"):
                return load_dataset("parquet", data_files=files, split="train")
        
        # 3. Download from HuggingFace
        if metadata.source == "huggingface":
            if metadata.requires_manual_download:
                logger.error(f"Dataset {metadata.name} is gated and requires manual download to {metadata.local_storage_path}")
                raise PermissionError(f"Gated dataset {metadata.name} needs manual download.")
                
            logger.info(f"Downloading {metadata.name} from HuggingFace Hub ({metadata.download_url})...")
            try:
                ds = load_dataset(metadata.download_url, split="train")
                
                # Cache it locally to raw
                os.makedirs(metadata.local_storage_path, exist_ok=True)
                cache_path = os.path.join(metadata.local_storage_path, "data.parquet")
                ds.to_parquet(cache_path)
                logger.info(f"Saved raw {metadata.name} to {cache_path}")
                
                return ds
            except Exception as e:
                logger.error(f"Failed to download {metadata.name} from HF: {e}")
                raise
                
        raise NotImplementedError(f"Source {metadata.source} loader not implemented.")
