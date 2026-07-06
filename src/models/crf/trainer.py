import logging
from src.models.crf.model import build_crf_model
import joblib

logger = logging.getLogger(__name__)

class CRFTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model = build_crf_model(
            c1=config.get("c1", 0.1),
            c2=config.get("c2", 0.1),
            max_iterations=config.get("max_iterations", 100)
        )
        
    def train(self, X_train, y_train):
        logger.info(f"Training CRF on {len(X_train)} sentences...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")
        
    def save_model(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
