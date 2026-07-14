import os
import sys
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.manager import DatasetManager
from src.data.eda import ExploratoryDataAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    manager = DatasetManager()
    analyzer = ExploratoryDataAnalyzer(manager)
    
    datasets = manager.registry.config.datasets
    logger.info(f"Starting Phase 3 EDA on {len(datasets)} datasets.")
    
    for ds in datasets:
        # Since IndicCMix and GLUECoS-LID require manual downloads, they will fail unless processed.
        # But we still run the analyzer which attempts to process them.
        logger.info(f"Running EDA for {ds.name}...")
        success = analyzer.run_analysis(ds.name)
        if success:
            logger.info(f"Successfully generated reports for {ds.name}")
        else:
            logger.warning(f"Skipped full EDA for {ds.name} due to missing data or loading error.")
            
    logger.info("EDA Phase Completed. Reports are available in datasets/reports/")

if __name__ == "__main__":
    main()
