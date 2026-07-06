import logging
import os
from backend.config import settings
from src.models.generation.inference import Generator
from backend.services.demo_generator import DemoGenerator

logger = logging.getLogger(__name__)

class GenerationService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenerationService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
        
    def __init__(self):
        if self.initialized:
            return
            
        logger.info(f"Initializing GenerationService on device: {settings.DEVICE}")
        try:
            self.generator = Generator(
                base_model_name=settings.MODEL_PATH,
                lora_adapter_path=settings.LORA_ADAPTER_PATH if os.path.exists(settings.LORA_ADAPTER_PATH) else None,
                config_path="configs/generation.yaml"
            )
            # Override device dynamically if possible (Generator might hardcode or use default)
            if hasattr(self.generator.model.model, 'to'):
                self.generator.model.model.to(settings.DEVICE)
                
            self.initialized = True
            logger.info("GenerationService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GenerationService: {e}")
            raise e
            
    def generate(self, instruction: str, context: str = "", template: str = "instruction", temperature: float = 0.8) -> dict:
        if not self.initialized:
            raise RuntimeError("GenerationService is not initialized.")
            
        if settings.DEMO_MODE:
            logger.info("DEMO_MODE is active. Bypassing mT5 generation.")
            # We can extract the style directly from the instruction string
            style_str = "neutral"
            if "positive" in instruction.lower(): style_str = "positive"
            if "negative" in instruction.lower(): style_str = "negative"
            if "formal" in instruction.lower(): style_str = "formal"
            
            return DemoGenerator.generate(instruction=instruction, style=style_str, temperature=temperature)
            
        # We assume the generator exposes a single generate method or batch.
        # inference.py has generate() or generate_batch()
        outputs = self.generator.generate_batch(
            instructions=[instruction],
            contexts=[context],
            template=template
        )
        return outputs[0]
        
    def format_style_prompt(self, base_prompt: str, style: str, english_usage: str) -> str:
        """Translates user options into explicit GEN_003 instruction strings."""
        parts = [base_prompt]
        
        # Style
        if style == "positive":
            parts.append("Write a positive Telugu-English review.")
        elif style == "negative":
            parts.append("Write a negative Telugu-English review.")
        elif style == "formal":
            parts.append("Use a formal and respectful tone.")
        elif style == "informal":
            parts.append("Use a casual, conversational tone.")
            
        # English Usage
        if english_usage == "high":
            parts.append("Use a high amount of English vocabulary.")
        elif english_usage == "low":
            parts.append("Use predominantly Telugu vocabulary.")
            
        return " ".join(parts).strip()
