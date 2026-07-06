import yaml
import logging
from pathlib import Path
from transformers import T5TokenizerFast

logger = logging.getLogger(__name__)

class TriMixTokenizer:
    """
    Wrapper for T5TokenizerFast to handle formatting for mT5-small training and inference.
    Isolates tokenization logic from dataset loading.
    """
    def __init__(self, config_path: str = "configs/generation.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = self.config['model']['name']
        self.tok_config = self.config['tokenization']
        
        logger.info(f"Loading T5TokenizerFast for {self.model_name}")
        # mT5 requires T5TokenizerFast. It relies on SentencePiece under the hood.
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
        
    def tokenize_for_training(self, source_texts, target_texts):
        """
        Tokenizes inputs and targets simultaneously for seq2seq training.
        Returns input_ids, attention_mask, and labels.
        """
        # Tokenize source (encoder input)
        model_inputs = self.tokenizer(
            source_texts,
            max_length=self.tok_config['max_source_length'],
            padding=self.tok_config['padding'],
            truncation=self.tok_config['truncation'],
            return_tensors="pt"
        )
        
        # Tokenize target (decoder output)
        labels = self.tokenizer(
            text_target=target_texts,
            max_length=self.tok_config['max_target_length'],
            padding=self.tok_config['padding'],
            truncation=self.tok_config['truncation'],
            return_tensors="pt"
        )
            
        # Replace padding token id in labels with -100 so it's ignored by CrossEntropyLoss
        labels_ids = labels["input_ids"]
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        
        model_inputs["labels"] = labels_ids
        
        return model_inputs

    def tokenize_for_inference(self, source_texts):
        """
        Tokenizes inputs only. Used during inference/evaluation.
        """
        return self.tokenizer(
            source_texts,
            max_length=self.tok_config['max_source_length'],
            padding=True, # Dynamic padding is usually preferred for inference batches
            truncation=self.tok_config['truncation'],
            return_tensors="pt"
        )
        
    def decode(self, token_ids, skip_special_tokens=True):
        """Decodes token IDs back to a string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
    def batch_decode(self, token_ids_list, skip_special_tokens=True):
        """Decodes a batch of token IDs back to strings."""
        # Ensure we replace -100 with pad_token_id before decoding (if decoding labels)
        import torch
        if isinstance(token_ids_list, torch.Tensor):
            token_ids_list = torch.where(
                token_ids_list != -100, 
                token_ids_list, 
                self.tokenizer.pad_token_id
            )
        elif isinstance(token_ids_list, list) and isinstance(token_ids_list[0], list):
            token_ids_list = [
                [token if token != -100 else self.tokenizer.pad_token_id for token in seq]
                for seq in token_ids_list
            ]
            
        return self.tokenizer.batch_decode(token_ids_list, skip_special_tokens=skip_special_tokens)
