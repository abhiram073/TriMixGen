import logging
import os
from backend.config import settings
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
        logger.info(f"DEMO_MODE: {settings.DEMO_MODE}")
        
        try:
            if settings.DEMO_MODE:
                # Lightweight demo mode - no ML libraries needed
                logger.info("Using DemoGenerator (lightweight mode, no PyTorch)")
                self.generator = DemoGenerator()
            else:
                # Production mode - requires transformers and torch
                logger.info("Loading full ML model (Generator with mT5 + LoRA)")
                from src.models.generation.inference import Generator
                
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
        
    def format_style_prompt(self, s1: str, s2: str, s3: str, language_pair: str, style: str, english_usage: str) -> str:
        """Translates user options into explicit GEN_003 instruction strings."""
        parts = [f"Translate and mix the following sentences into trilingual code-mixed text:\n1: {s1}\n2: {s2}\n3: {s3}\n\nConstraints:"]
        
        # Language Pair
        if language_pair == "HIN-BEN-ENG":
            parts.append("- Must use Hindi, Bengali, and English.")
        elif language_pair == "HIN-GUJ-ENG":
            parts.append("- Must use Hindi, Gujarati, and English.")
        
        # Style
        if style == "positive":
            parts.append("- Make it a positive review.")
        elif style == "negative":
            parts.append("- Make it a negative review.")
        elif style == "formal":
            parts.append("- Use a formal and respectful tone.")
        elif style == "informal":
            parts.append("- Use a casual, conversational tone.")
            
        # English Usage
        if english_usage == "high":
            parts.append("- Use a high amount of English vocabulary.")
        elif english_usage == "low":
            parts.append("- Use predominantly Indic vocabulary.")
            
        return "\n".join(parts).strip()
