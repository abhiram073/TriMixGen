import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensuring deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to: {seed}")

def get_device() -> torch.device:
    """
    Detects and returns the best available hardware device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    logger.info(f"Using device: {device}")
    return device

def generate_experiment_name(model_name: str, task: str) -> str:
    """
    Generates a unique experiment name based on timestamp.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{task}_{timestamp}"
