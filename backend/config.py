import os
import torch
from typing import List

class Settings:
    PROJECT_NAME: str = "TriMixGen Code-Mixing Generation API"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        origin.strip() for origin in os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    ]
    
    # Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "google/mt5-small")
    LORA_ADAPTER_PATH: str = os.getenv("LORA_ADAPTER_PATH", "outputs/experiments/gen_003/best_model")
    INDICBERT_MODEL_PATH: str = os.getenv("INDICBERT_MODEL_PATH", "ai4bharat/indic-bert")
    USE_MOCK_LID: bool = os.getenv("USE_MOCK_LID", "true").lower() == "true"
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    PROMPTS_CONFIG_PATH: str = os.getenv("PROMPTS_CONFIG_PATH", "configs/prompts.yaml")
    
    # Constraints
    MAX_PROMPT_LENGTH: int = 500

    @property
    def DEVICE(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

settings = Settings()
