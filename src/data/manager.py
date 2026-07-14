import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict

from src.data.registry import DatasetRegistry
from src.data.loader import DatasetLoader
from src.data.validator import DatasetValidator
from src.data.statistics import DatasetStatistics

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Facade class managing the complete dataset lifecycle:
    Registration -> Downloading -> Loading -> Validation -> Stats -> Splitting -> Exporting
    """
    
    def __init__(self, registry_config_path: str = "configs/datasets.yaml"):
        self.registry = DatasetRegistry(registry_config_path)
        os.makedirs("datasets/reports", exist_ok=True)
        os.makedirs("datasets/processed", exist_ok=True)
        os.makedirs("datasets/metadata", exist_ok=True)

    def process_dataset(self, name: str) -> bool:
        """Runs the full pipeline for a single dataset."""
        logger.info(f"--- Starting processing for dataset: {name} ---")
        
        metadata = self.registry.get_dataset(name)
        if not metadata:
            logger.error(f"Dataset {name} not found in registry.")
            return False
            
        try:
            # 1. Load (and Download if necessary)
            hf_dataset = DatasetLoader.load(metadata)
            df = hf_dataset.to_pandas()
            
            # 2. Validate against schema
            # Assuming the task is the first task defined in metadata
            task = metadata.tasks[0] if metadata.tasks else "unknown"
            
            if task == "generation":
                val_report = DatasetValidator.validate_generation_schema(df)
            elif task == "language_identification":
                val_report = DatasetValidator.validate_lid_schema(df)
            else:
                val_report = {"is_valid": True, "warnings": [f"No schema validation implemented for task: {task}"]}
                
            if not val_report.get("is_valid", False):
                logger.error(f"Validation failed for {name}: {val_report.get('errors')}")
                return False
                
            # 3. Generate Statistics
            stats = DatasetStatistics.generate_report(df, name, task)
            
            # 4. Generate Split (Train/Val/Test)
            splits = self.split_dataset(hf_dataset)
            
            # 5. Export Processed Data & Metadata
            self.export_processed(name, splits, metadata.model_dump(), stats, val_report)
            
            logger.info(f"--- Successfully processed dataset: {name} ---")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process dataset {name}: {e}")
            return False

    def split_dataset(self, dataset: Dataset) -> DatasetDict:
        """Splits a dataset based on registry config."""
        splits_cfg = self.registry.config.splits
        seed = splits_cfg.random_seed
        
        # Train / Test Split
        test_size = splits_cfg.test + splits_cfg.validation
        train_test = dataset.train_test_split(test_size=test_size, seed=seed)
        
        # Validation / Test Split
        val_ratio = splits_cfg.validation / test_size
        val_test = train_test['test'].train_test_split(test_size=1.0 - val_ratio, seed=seed)
        
        return DatasetDict({
            'train': train_test['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })

    def export_processed(self, name: str, splits: DatasetDict, metadata: dict, stats: dict, val_report: dict):
        """Saves the unified schemas to disk along with reports and metadata versioning."""
        processed_dir = os.path.join("datasets/processed", name)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save Parquet splits
        for split_name, ds in splits.items():
            ds.to_parquet(os.path.join(processed_dir, f"{split_name}.parquet"))
            
        # Save Metadata / Versioning
        version_info = {
            "dataset": name,
            "processed_timestamp": datetime.now().isoformat(),
            "original_metadata": metadata,
            "schema_version": "1.0",
        }
        with open(os.path.join("datasets/metadata", f"{name}_version.json"), "w") as f:
            json.dump(version_info, f, indent=2)
            
        # Save Report
        report = {
            "version_info": version_info,
            "validation": val_report,
            "statistics": stats
        }
        with open(os.path.join("datasets/reports", f"{name}_report.json"), "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Exported processed data, metadata, and reports for {name}.")
