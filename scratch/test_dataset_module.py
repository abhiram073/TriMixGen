import sys
import logging
from src.data.manager import DatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_registry():
    logger.info("Initializing DatasetManager...")
    manager = DatasetManager()
    
    registry = manager.registry
    logger.info(f"Loaded {len(registry.config.datasets)} datasets.")
    
    # Test getting dataset by name
    ds = registry.get_dataset("IndicCMix")
    assert ds is not None
    assert ds.name == "IndicCMix"
    assert "generation" in ds.tasks
    assert ds.requires_manual_download == True
    logger.info("IndicCMix metadata successfully parsed.")
    
    # Test getting dataset by task
    lid_datasets = registry.get_datasets_by_task("language_identification")
    assert len(lid_datasets) == 2
    logger.info(f"Found {len(lid_datasets)} LID datasets.")
    
    # Check splits config
    splits = registry.config.splits
    assert splits.train == 0.8
    assert splits.validation == 0.1
    logger.info("Splits configuration parsed successfully.")
    
    logger.info("Dataset Module Registry and Architecture verified.")

if __name__ == "__main__":
    test_registry()
