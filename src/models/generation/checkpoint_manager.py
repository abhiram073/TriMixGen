import os
import json
import shutil
import logging
from pathlib import Path
from peft import PeftModel

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Modular component for managing LoRA adapters, versioning, 
    and checkpoint lifecycle. Decoupled from the trainer and inference logic.
    """
    
    @staticmethod
    def save_lora_adapter(model_wrapper, output_dir: str):
        """
        Saves only the LoRA adapter weights and config to the specified directory.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving LoRA adapter to {output_dir}...")
        
        # Save adapter. The Hugging Face save_pretrained natively handles PeftModels.
        if hasattr(model_wrapper.model, "save_pretrained"):
            model_wrapper.model.save_pretrained(str(path))
        else:
            raise ValueError("Provided model does not support save_pretrained.")

    @staticmethod
    def load_lora_adapter(model_wrapper, adapter_dir: str):
        """
        Dynamically attaches a LoRA adapter to a base model.
        """
        logger.info(f"Loading LoRA adapter from {adapter_dir}...")
        if not Path(adapter_dir).exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
            
        peft_model = PeftModel.from_pretrained(model_wrapper.model, adapter_dir)
        model_wrapper.model = peft_model
        return model_wrapper

    @staticmethod
    def merge_and_unload(model_wrapper):
        """
        Mathematically merges the LoRA matrices into the base weights for 
        zero-latency inference, and unloads the adapter states.
        """
        logger.info("Merging LoRA adapters into base weights for inference...")
        if hasattr(model_wrapper.model, "merge_and_unload"):
            merged_model = model_wrapper.model.merge_and_unload()
            model_wrapper.model = merged_model
        else:
            logger.warning("Model does not support merge_and_unload. Skipping merge.")
        return model_wrapper

    @staticmethod
    def get_best_checkpoint(experiment_dir: str) -> str:
        """
        Parses an experiment directory to find the checkpoint with the lowest eval_loss.
        Expects a 'trainer_state.json' file inside checkpoint folders.
        """
        exp_path = Path(experiment_dir)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory {experiment_dir} not found.")
            
        best_checkpoint = None
        best_loss = float("inf")
        
        for ckpt_dir in exp_path.glob("checkpoint-*"):
            state_file = ckpt_dir / "trainer_state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    
                # Look for best_metric in state, or fallback to parsing log_history
                if "best_metric" in state and state["best_metric"] is not None:
                    loss = float(state["best_metric"])
                else:
                    # Find lowest eval_loss in log history
                    history = state.get("log_history", [])
                    eval_losses = [log["eval_loss"] for log in history if "eval_loss" in log]
                    loss = min(eval_losses) if eval_losses else float("inf")
                    
                if loss < best_loss:
                    best_loss = loss
                    best_checkpoint = str(ckpt_dir)
                    
        if best_checkpoint is None:
            # Fallback to the latest modified directory if states are missing
            checkpoints = list(exp_path.glob("checkpoint-*"))
            if checkpoints:
                best_checkpoint = str(max(checkpoints, key=os.path.getmtime))
                
        return best_checkpoint

    @staticmethod
    def cleanup_checkpoints(experiment_dir: str, keep_top_k: int = 3):
        """
        Deletes the worst performing checkpoints to save disk space, 
        keeping only the top K best models based on eval_loss.
        """
        exp_path = Path(experiment_dir)
        checkpoints = []
        
        for ckpt_dir in exp_path.glob("checkpoint-*"):
            state_file = ckpt_dir / "trainer_state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    
                history = state.get("log_history", [])
                eval_losses = [log["eval_loss"] for log in history if "eval_loss" in log]
                loss = min(eval_losses) if eval_losses else float("inf")
                
                checkpoints.append((loss, ckpt_dir))
            else:
                # Keep checkpoints without state files to be safe
                checkpoints.append((-float("inf"), ckpt_dir))
                
        # Sort by loss (ascending)
        checkpoints.sort(key=lambda x: x[0])
        
        # Determine directories to delete
        to_delete = checkpoints[keep_top_k:]
        
        for loss, ckpt_dir in to_delete:
            logger.info(f"Cleaning up checkpoint: {ckpt_dir} (Loss: {loss})")
            shutil.rmtree(ckpt_dir, ignore_errors=True)
