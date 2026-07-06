import yaml
import logging
from transformers import GenerationConfig

logger = logging.getLogger(__name__)

class GenerationConfigManager:
    """
    Manages and validates various decoding strategies for text generation.
    """
    def __init__(self, config_path: str = "configs/generation.yaml"):
        self.config_path = config_path
        self.raw_config = self._load_config()
        self.inference_params = self.raw_config.get("inference", {})
        
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Generation config {self.config_path} not found. Using defaults.")
            return {"inference": {}}

    def get_strategy_config(self, strategy_name: str) -> GenerationConfig:
        """
        Constructs a Hugging Face GenerationConfig based on the requested strategy.
        Valid strategies: 'greedy', 'beam_search', 'top_k_sampling', 'nucleus_sampling'
        """
        if strategy_name not in self.inference_params:
            raise ValueError(f"Strategy '{strategy_name}' not found in configuration.")
            
        # Base limits
        base_kwargs = {
            "max_new_tokens": self.inference_params.get("max_new_tokens", 128),
            "min_new_tokens": self.inference_params.get("min_new_tokens", 5)
        }
        
        # Merge base with strategy-specific kwargs
        strategy_kwargs = self.inference_params[strategy_name]
        
        # Validation checks
        if strategy_kwargs.get("do_sample", False) and strategy_kwargs.get("num_beams", 1) > 1:
            # While technically supported in HF as Beam Sampling, it's usually unintended.
            logger.warning("do_sample=True and num_beams>1 selected. This enables Beam Search with Sampling.")
            
        merged_kwargs = {**base_kwargs, **strategy_kwargs}
        
        # Convert dictionary to HF GenerationConfig
        return GenerationConfig(**merged_kwargs)
