import os
import torch
import logging
from pathlib import Path
from transformers import MT5ForConditionalGeneration, GenerationConfig

logger = logging.getLogger(__name__)

class TriMixGeneratorModel:
    """
    Encapsulates the mT5 generation model.
    Isolates model initialization, loading, and saving from the training loop.
    Does NOT handle LoRA (which is managed by lora_config.py).
    """
    def __init__(self, model_name_or_path: str = "google/mt5-small", device: str = None):
        self.model_name_or_path = str(model_name_or_path)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        
    def _load_model(self):
        logger.info(f"Loading MT5ForConditionalGeneration from {self.model_name_or_path}...")
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        logger.info(f"Model loaded successfully on {self.device}.")
        
    def save_checkpoint(self, output_dir: str):
        """Saves the base model weights to a directory."""
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_generation_config(self, config_path: str):
        """Applies a GenerationConfig from a local file or creates a default one."""
        # For mT5, generation config usually dictates beam search, etc.
        # Here we just initialize a default one if path is abstract
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        logger.info("Generation configuration loaded.")
        
    def get_diagnostics(self) -> dict:
        """Returns comprehensive model architecture diagnostics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate approximate memory footprint in MB (assuming float32 which is 4 bytes)
        mem_mb = (total_params * 4) / (1024 ** 2)
        
        encoder_layers = len(self.model.encoder.block)
        decoder_layers = len(self.model.decoder.block)
        
        vocab_size = self.model.config.vocab_size
        d_model = self.model.config.d_model
        
        return {
            "model_name_or_path": self.model_name_or_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_footprint_mb": round(mem_mb, 2),
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "device": self.device
        }

    def generate_diagnostics_report(self, output_path: str = "docs/model_summary.md"):
        """Generates a markdown report containing the model architecture details."""
        diag = self.get_diagnostics()
        report = (
            f"# Model Architecture Summary: {diag['model_name_or_path']}\n\n"
            f"## Computational Diagnostics\n"
            f"- **Total Parameters:** {diag['total_parameters']:,}\n"
            f"- **Trainable Parameters:** {diag['trainable_parameters']:,}\n"
            f"- **Memory Footprint (FP32):** ~{diag['memory_footprint_mb']} MB\n"
            f"- **Device:** {diag['device']}\n\n"
            f"## Structural Diagnostics\n"
            f"- **Architecture Type:** Encoder-Decoder (T5)\n"
            f"- **Encoder Layers:** {diag['encoder_layers']}\n"
            f"- **Decoder Layers:** {diag['decoder_layers']}\n"
            f"- **Hidden Size (d_model):** {diag['d_model']}\n"
            f"- **Vocabulary Size (SentencePiece):** {diag['vocab_size']:,}\n"
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Diagnostics report saved to {output_path}")
