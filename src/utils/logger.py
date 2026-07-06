import os
import yaml
import logging.config

def setup_logging(default_path='configs/logging.yaml', default_level=logging.INFO):
    """
    Setup logging configuration from a YAML file.
    All modules in TriMixGen should use this utility.
    """
    # Ensure outputs directory exists for the rotating file handler
    os.makedirs('outputs', exist_ok=True)
    
    if os.path.exists(default_path):
        with open(default_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error loading logging config: {e}. Using basic config.")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Failed to load configuration file at {default_path}. Using basic config.")

def get_logger(name: str):
    """
    Returns a configured logger instance.
    Usage in other modules:
    
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message")
    """
    return logging.getLogger(name)
