import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.tokenizer import TriMixTokenizer
from src.models.generation.generation_config import GenerationConfigManager

def main():
    print("\n--- Decoding Strategy Benchmark ---")
    print("Loading mT5-small Base Model (Untrained)...")
    # Base model untrained for this task, so outputs will be nonsensical, 
    # but it proves the API works and strategies alter behavior.
    model_wrapper = TriMixGeneratorModel(model_name_or_path="google/mt5-small", device="cpu")
    tokenizer = TriMixTokenizer(config_path="configs/generation.yaml")
    config_mgr = GenerationConfigManager(config_path="configs/generation.yaml")
    
    prompt = "Translate to colloquial Telugu-English: The movie was excellent."
    input_tensors = tokenizer.tokenize_for_inference([prompt])
    input_ids = input_tensors["input_ids"]
    
    strategies = ["greedy", "beam_search", "top_k_sampling", "nucleus_sampling"]
    
    for strategy in strategies:
        print(f"\n[Strategy: {strategy.upper()}]")
        gen_config = config_mgr.get_strategy_config(strategy)
        
        # Set seed for sampling reproducibility during testing
        torch.manual_seed(42)
        
        outputs = model_wrapper.model.generate(
            input_ids,
            generation_config=gen_config
        )
        
        decoded = tokenizer.decode(outputs[0])
        print(f"Output: {decoded}")
        
    print("\nBenchmark Complete!")

if __name__ == "__main__":
    main()
