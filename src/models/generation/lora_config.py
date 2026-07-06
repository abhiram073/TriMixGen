import yaml
import logging
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from src.models.generation.model import TriMixGeneratorModel

logger = logging.getLogger(__name__)

class LoRAConfigurator:
    """
    Manages the injection of PEFT/LoRA adapters into the base generator model.
    """
    def __init__(self, config_path: str = "configs/lora.yaml"):
        self.config_path = config_path
        self.lora_params = self._load_config()
        self.peft_config = self._build_config()
        
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"LoRA config {self.config_path} not found. Using defaults.")
            return {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q", "v"]
            }

    def _build_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.lora_params.get("r", 8),
            lora_alpha=self.lora_params.get("lora_alpha", 16),
            lora_dropout=self.lora_params.get("lora_dropout", 0.05),
            bias=self.lora_params.get("bias", "none"),
            target_modules=self.lora_params.get("target_modules", ["q", "v"]),
            task_type="SEQ_2_SEQ_LM"
        )
        
    def inject_adapter(self, model_wrapper: TriMixGeneratorModel) -> TriMixGeneratorModel:
        """
        Takes a base TriMixGeneratorModel and injects the LoRA adapters.
        Modifies the underlying Hugging Face model in place.
        """
        logger.info("Injecting LoRA adapters into base model...")
        base_model = model_wrapper.model
        
        # Apply PEFT
        peft_model = get_peft_model(base_model, self.peft_config)
        
        # Log parameter statistics
        trainable_params, all_param = peft_model.get_nb_trainable_parameters()
        percent = 100 * trainable_params / all_param
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable %: {percent:.4f}"
        )
        
        model_wrapper.model = peft_model
        return model_wrapper
        
    def load_adapter(self, model_wrapper: TriMixGeneratorModel, adapter_path: str) -> TriMixGeneratorModel:
        """
        Loads pre-trained LoRA adapter weights into a base model.
        Used for inference or resumed training.
        """
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path {adapter_path} does not exist.")
            
        base_model = model_wrapper.model
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        model_wrapper.model = peft_model
        return model_wrapper
