# Checkpoint Management Architecture

This document describes the design and lifecycle of model checkpoints during the TriMixGen code-mixed generation phase.

## 1. LoRA Adapter Isolation
Unlike full fine-tuning which overwrites the base 1.2GB mT5 weights, our LoRA implementation only trains a tiny subset of injected matrices (~3 MB). The `CheckpointManager` strictly enforces the separation of these artifacts.
*   **Saving:** Only the LoRA `adapter_model.safetensors` and `adapter_config.json` are written to disk. The frozen base weights are never duplicated.
*   **Loading:** The manager dynamically attaches a 3MB adapter checkpoint onto a standard `google/mt5-small` instance at runtime.

## 2. Checkpoint Versioning
During a training run, checkpoints are saved at the end of each epoch or at specific step intervals (e.g., `checkpoint-500`, `checkpoint-1000`).
The `CheckpointManager` parses the experiment directory to track all active versions.

## 3. Best Model Selection
Because generation tasks can easily overfit on small datasets (manifesting as a drop in vocabulary diversity despite a lowering cross-entropy loss), the manager relies on the validation `trainer_state.json` to parse metrics and select the absolute best checkpoint, ensuring the final artifact is optimized for generalization.

## 4. Adapter Merging for Inference
While loading adapters dynamically is fine for training, it adds slight computational overhead during live inference because the matrix multiplications must be computed separately: $h = W_0 x + B A x$.
For production deployment, the `CheckpointManager` supports **Merging and Unloading**. This mathematically fuses the LoRA matrices directly into the base weights ($W_{new} = W_0 + BA$), yielding a standard, unified transformer that incurs zero inference latency penalty compared to the original mT5 model.

## 5. Storage Cleanup
To prevent disk exhaustion during hyperparameter sweeps on the D:\ drive, the `cleanup_checkpoints(keep_top_k=3)` method automatically scans an experiment directory and deletes the worst-performing intermediate checkpoints, preserving only the top performing iterations and the final output.
