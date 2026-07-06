import sys
from pathlib import Path
# Ensure the src module is discoverable
sys.path.append(str(Path(__file__).parent.parent))

from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.lora_config import LoRAConfigurator

def main():
    print("\n--- LoRA Injection Demonstration ---")
    
    # 1. Load Base Model
    print("Loading Base mT5-small Model...")
    base_model = TriMixGeneratorModel(model_name_or_path="google/mt5-small", device="cpu")
    
    # Base params
    base_trainable = sum(p.numel() for p in base_model.model.parameters() if p.requires_grad)
    base_total = sum(p.numel() for p in base_model.model.parameters())
    
    print(f"\n[Before LoRA]")
    print(f"Total Parameters:     {base_total:,}")
    print(f"Trainable Parameters: {base_trainable:,}")
    
    # 2. Inject LoRA
    print("\nLoading LoRA Configurator and Injecting Adapters...")
    configurator = LoRAConfigurator(config_path="configs/lora.yaml")
    adapted_model = configurator.inject_adapter(base_model)
    
    # LoRA params
    lora_trainable, lora_total = adapted_model.model.get_nb_trainable_parameters()
    
    print(f"\n[After LoRA Injection]")
    print(f"Total Parameters:     {lora_total:,}")
    print(f"Trainable Parameters: {lora_trainable:,}")
    print(f"Reduction in Trainable Params: {(1 - (lora_trainable/base_trainable))*100:.2f}%")
    
    print("\nDemonstration Complete. Ready for CPU Fine-Tuning!")

if __name__ == "__main__":
    main()
