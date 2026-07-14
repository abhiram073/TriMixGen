import yaml
import logging
from typing import List, Optional
from src.data.schemas import DatasetRegistryConfig, DatasetMetadata

logger = logging.getLogger(__name__)

class DatasetRegistry:
    """
    Central registry that loads datasets.yaml and provides access to registered datasets.
    """
    def __init__(self, config_path: str = "configs/datasets.yaml"):
        self.config_path = config_path
        self.config: Optional[DatasetRegistryConfig] = None
        self._load_registry()

    def _load_registry(self):
        try:
            with open(self.config_path, "r") as f:
                raw_config = yaml.safe_load(f)
            self.config = DatasetRegistryConfig(**raw_config)
            logger.info(f"Loaded {len(self.config.datasets)} datasets from registry.")
        except Exception as e:
            logger.error(f"Failed to load dataset registry from {self.config_path}: {e}")
            raise

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        for ds in self.config.datasets:
            if ds.name == name:
                return ds
        return None

    def get_datasets_by_task(self, task: str) -> List[DatasetMetadata]:
        return [ds for ds in self.config.datasets if task in ds.tasks]

    def get_datasets_by_combination(self, combination: str) -> List[DatasetMetadata]:
        return [ds for ds in self.config.datasets if combination in ds.supported_combinations]
