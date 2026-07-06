import pandas as pd
from torch.utils.data import Dataset
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TriMixGenerationDataset(Dataset):
    """
    Unified Dataset class for loading raw data components across different curriculum stages.
    Yields dictionaries containing raw fields (e.g., 'instruction', 'context', 'target').
    The PromptBuilder will later format these into the final model inputs.
    """
    def __init__(self, data_path, dataset_type="alpaca"):
        """
        Args:
            data_path: Path to the parquet/csv file.
            dataset_type: The type of dataset ('alpaca', 'hold', 'sentiment').
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.data = self._load_data()
        
    def _load_data(self):
        logger.info(f"Loading {self.dataset_type} dataset from {self.data_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
            
        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}. Use .csv or .parquet")
            
        processed_data = []
        if self.dataset_type == "alpaca":
            for _, row in df.iterrows():
                processed_data.append({
                    "instruction": str(row.get("instruction", "")),
                    "input": str(row.get("input", "")),
                    "target": str(row.get("output", "")),
                    "dataset_type": "alpaca"
                })
        elif self.dataset_type == "hold":
            for _, row in df.iterrows():
                processed_data.append({
                    "context": str(row.get("context", "")),
                    "target": str(row.get("response", "")),
                    "dataset_type": "hold"
                })
        elif self.dataset_type == "sentiment":
            for _, row in df.iterrows():
                processed_data.append({
                    "label": str(row.get("label", "")),
                    "target": str(row.get("text", "")),
                    "dataset_type": "sentiment"
                })
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        # Filter out rows where the target is empty or NaN
        processed_data = [
            item for item in processed_data 
            if item.get("target") and str(item["target"]).strip() and str(item["target"]).strip().lower() != "nan"
        ]
        logger.info(f"Loaded {len(processed_data)} valid samples.")
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
