# Inference Pipeline Architecture: TriMixGen

This document outlines the architecture for the TriMixGen production inference engine (`Generator`).

## 1. Modular Assembly
The `Generator` class is the culmination of all TriMixGen generation modules. It imports and coordinates:
*   **`TriMixGeneratorModel`**: Handles the underlying `mT5-small` huggingface instance.
*   **`TriMixTokenizer`**: Handles multilingual SentencePiece encoding/decoding.
*   **`PromptBuilder`**: Orchestrates formatting (e.g., mapping `instruction` and `context` into `english_to_codemixed` YAML templates).
*   **`GenerationConfigManager`**: Injects `top_p`, `temperature`, and sequence length constraints from YAML into the forward pass.
*   **`CheckpointManager`**: Safely attaches 3MB LoRA matrices to the base model or fully merges them for zero-latency inference.

## 2. Checkpoint Loading Strategies
The engine supports three modes of execution based on deployment constraints:
1.  **Base Mode:** Pure `mT5` inference (used to establish pre-training baselines).
2.  **Adapter Mode:** Loads base weights, then dynamically injects LoRA. Best for testing multiple adapters (e.g., swapping between `GEN_001` and `GEN_002` experiments on the fly).
3.  **Merged Mode:** Calls `CheckpointManager.merge_and_unload()`. Best for final production deployment, as it mathematically fuses the adapters, reducing the computational overhead of the LoRA path during matrix multiplication.

## 3. Decoding and Entropy Management
Code-mixed generation is highly entropic. The engine enforces **Nucleus Sampling (Top-p)** by default because greedy strategies collapse into safe, monolingual output. The inference output strictly traces the decoding configuration used, appending `generation_time` and `token_count` to ensure CPU bounds are respected.

## 4. API Specification
The final pipeline exposes a dead-simple interface for downstream applications:
```python
generator = Generator(base_model_path="google/mt5-small", lora_path="outputs/best_lora")
output = generator.generate(
    instruction="Translate this to code-mixed Telugu",
    context="The movie was absolutely fantastic.",
    template="english_to_codemixed"
)
```
