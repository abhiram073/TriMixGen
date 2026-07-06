import pytest
from pathlib import Path
from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.lora_config import LoRAConfigurator

@pytest.fixture(scope="module")
def base_model():
    # Load base model on CPU for testing
    return TriMixGeneratorModel(model_name_or_path="google/mt5-small", device="cpu")

@pytest.fixture(scope="module")
def configurator():
    return LoRAConfigurator(config_path="configs/lora.yaml")

def test_configurator_loading(configurator):
    assert configurator.peft_config.r == 8
    assert configurator.peft_config.lora_alpha == 16
    assert configurator.peft_config.lora_dropout == 0.05
    assert "q" in configurator.peft_config.target_modules

def test_adapter_injection_and_param_count(base_model, configurator):
    # Store base trainable params
    base_trainable = sum(p.numel() for p in base_model.model.parameters() if p.requires_grad)
    
    # Inject LoRA
    adapted_model = configurator.inject_adapter(base_model)
    
    # Get new trainable params
    lora_trainable, all_param = adapted_model.model.get_nb_trainable_parameters()
    
    # LoRA trainable params should be drastically smaller than base
    assert lora_trainable < base_trainable
    assert lora_trainable > 0
    # For mT5-small, total params ~ 300M. 
    # With r=8 on q, v, trainable params should be ~850k (which is < 1%)
    assert (lora_trainable / all_param) < 0.01

def test_adapter_save_and_load(base_model, configurator, tmp_path):
    save_dir = tmp_path / "lora_test_checkpoint"
    
    # Inject and save
    adapted_model = configurator.inject_adapter(base_model)
    adapted_model.model.save_pretrained(str(save_dir))
    
    assert (save_dir / "adapter_config.json").exists()
    assert (save_dir / "adapter_model.safetensors").exists()
    
    # Load clean base model
    clean_base = TriMixGeneratorModel(model_name_or_path="google/mt5-small", device="cpu")
    
    # Load adapter into clean base
    loaded_model = configurator.load_adapter(clean_base, str(save_dir))
    
    assert loaded_model.model is not None
    # Ensure it's recognized as a PeftModel
    assert type(loaded_model.model).__name__ == "PeftModelForSeq2SeqLM"
