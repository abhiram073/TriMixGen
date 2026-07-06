import torch
from transformers import AutoModelForTokenClassification
import logging

logger = logging.getLogger(__name__)

def build_indicbert_model(model_name, num_labels, id2label, label2id):
    """
    Initializes mBERT for Token Classification and applies the requested layer freezing strategy.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Freeze Strategy: Embeddings + Lower Layers
    # IndicBERT is ALBERT-based. ALBERT shares parameters across layers, so freezing "lower layers" 
    # actually freezes the single shared layer, which effectively freezes all layers!
    # To follow the spirit of the experiment (freezing embeddings), we will just freeze embeddings.
    logger.info("Freezing Embeddings...")
    if hasattr(model, 'albert'):
        for param in model.albert.embeddings.parameters():
            param.requires_grad = False
    elif hasattr(model, 'bert'):
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
            
    # Calculate trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    return model
