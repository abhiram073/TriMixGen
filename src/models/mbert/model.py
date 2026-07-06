import torch
from transformers import AutoModelForTokenClassification
import logging

logger = logging.getLogger(__name__)

def build_mbert_model(num_labels, id2label, label2id):
    """
    Initializes mBERT for Token Classification and applies the requested layer freezing strategy.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Freeze Strategy: Embeddings + Lower 6 Layers
    logger.info("Freezing Embeddings...")
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
        
    logger.info("Freezing Encoder Layers 0 to 5...")
    for i in range(6):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
            
    # Calculate trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    return model
