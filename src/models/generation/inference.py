import time
import logging
from typing import List, Dict, Any, Union

from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.tokenizer import TriMixTokenizer
from src.models.generation.prompt_builder import PromptBuilder
from src.models.generation.generation_config import GenerationConfigManager
from src.models.generation.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class Generator:
    """
    Production-ready inference engine for TriMixGen code-mixed generation.
    Supports LoRA adapters, merged checkpoints, and configurable decoding.
    """
    def __init__(self, 
                 base_model_name: str = "google/mt5-small",
                 lora_adapter_path: str = None,
                 merge_weights: bool = False,
                 config_path: str = "configs/generation.yaml"):
        
        logger.info("Initializing TriMixGen Inference Engine...")
        
        # 1. Initialize Tokenizer
        self.tokenizer = TriMixTokenizer(config_path=config_path)
        
        # 2. Initialize Prompt Builder
        self.prompt_builder = PromptBuilder()
        
        # 3. Initialize Model (CPU mode enforced by lack of explicit device mapping if CPU-only)
        self.model = TriMixGeneratorModel(model_name_or_path=base_model_name)
        
        # 4. Load LoRA Adapters if specified
        if lora_adapter_path:
            self.model = CheckpointManager.load_lora_adapter(self.model, lora_adapter_path)
            
            if merge_weights:
                self.model = CheckpointManager.merge_and_unload(self.model)
                
        # Put model in eval mode
        if hasattr(self.model.model, "eval"):
            self.model.model.eval()
            
        # 5. Load Generation Configuration
        self.config_manager = GenerationConfigManager(config_path=config_path)

    def generate(self, 
                 instruction: str, 
                 context: str = None, 
                 template: str = "english_to_codemixed",
                 decoding_preset: str = "nucleus_sampling") -> Dict[str, Any]:
        """
        Generates a single code-mixed response.
        """
        results = self.generate_batch(
            instructions=[instruction],
            contexts=[context],
            template=template,
            decoding_preset=decoding_preset
        )
        return results[0] if results else {}

    def generate_batch(self, 
                       instructions: List[str], 
                       contexts: List[str] = None, 
                       template: str = "english_to_codemixed",
                       decoding_preset: str = "nucleus_sampling") -> List[Dict[str, Any]]:
        """
        Batch inference supporting all decoding strategies.
        """
        if not instructions:
            return []
            
        if contexts is None:
            contexts = [None] * len(instructions)
            
        start_time = time.time()
        
        # 1. Build Prompts
        raw_prompts = []
        for inst, ctx in zip(instructions, contexts):
            kwargs = {"instruction": inst}
            if ctx is not None: kwargs["input"] = ctx
            raw_prompts.append(self.prompt_builder.render(template, **kwargs))
            
        # 2. Tokenize
        inputs = self.tokenizer.tokenize_for_inference(raw_prompts)
        
        # 3. Retrieve Decoding Config
        gen_config = self.config_manager.get_strategy_config(decoding_preset)
        gen_config_dict = gen_config.to_dict()
        # Prevent input length exceeding from HF warnings
        if hasattr(gen_config, "max_length"):
            gen_config.max_length = None
            
        # 4. Generate
        # We assume CPU inference natively since we don't move to .cuda()
        outputs = self.model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            generation_config=gen_config
        )
        
        # 5. Decode
        decoded_texts = [self.tokenizer.decode(out) for out in outputs]
        
        end_time = time.time()
        time_per_sample = (end_time - start_time) / len(instructions)
        
        # 6. Format Output
        results = []
        for i, text in enumerate(decoded_texts):
            # Calculate generated tokens (excluding padding)
            token_count = len([t for t in outputs[i] if t != self.tokenizer.tokenizer.pad_token_id])
            
            results.append({
                "generated_text": text,
                "raw_prompt": raw_prompts[i],
                "generation_time_sec": round(time_per_sample, 4),
                "token_count": token_count,
                "decoding_strategy": decoding_preset,
                "generation_parameters": gen_config_dict
            })
            
        return results
