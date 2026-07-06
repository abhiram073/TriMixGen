import sys
import logging
from pathlib import Path
import random
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
from src.models.crf.dataset import load_pseudo_labeled_data, load_gold_data
from src.models.bilstm.dataset import Vocabulary, SequenceDataset, collate_fn
from src.models.bilstm_attn.model import BiLSTMAttention_LID
from src.models.bilstm_attn.trainer import BiLSTMAttnTrainer

def setup_logger(log_file: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def run_experiment_004():
    exp_dir = Path("data/outputs/experiments/exp_004_bilstm_attn")
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(exp_dir / "train.log")
    logger.info("Starting Experiment 004: BiLSTM + Attention (Weighted Loss)")
    
    config = {
        "experiment_id": "004",
        "model_name": "BiLSTM + Attention",
        "dataset_version": "V1.1",
        "random_seed": 42,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 5,
        "min_freq": 2,
        "max_length": 128,
        "use_class_weights": True
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading pseudo-labeled training data...")
    # Use 10000 sentences for a good balance on CPU
    all_sentences_labels = load_pseudo_labeled_data([Path("data/processed")], max_samples=5000)
    train_data, val_data = train_test_split(all_sentences_labels, test_size=0.1, random_state=config["random_seed"])
    
    train_sentences = [s[0] for s in train_data]
    train_labels = [s[1] for s in train_data]
    val_sentences = [s[0] for s in val_data]
    val_labels = [s[1] for s in val_data]
    
    vocab = Vocabulary(min_freq=config["min_freq"])
    vocab.build(train_sentences, train_labels)
    
    # Calculate Class Weights
    class_weights = None
    if config["use_class_weights"]:
        label_counts = Counter([label for seq in train_labels for label in seq])
        total_labels = sum(label_counts.values())
        class_weights = {}
        for label, count in label_counts.items():
            # Inverse frequency weighting
            class_weights[label] = total_labels / (len(label_counts) * count)
        logger.info(f"Calculated Class Weights: {class_weights}")
        
    train_dataset = SequenceDataset(train_sentences, train_labels, vocab, max_length=config["max_length"])
    val_dataset = SequenceDataset(val_sentences, val_labels, vocab, max_length=config["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    model = BiLSTMAttention_LID(
        vocab_size=len(vocab.token2id),
        num_classes=len(vocab.label2id),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    
    trainer = BiLSTMAttnTrainer(model, config, device, vocab, class_weights=class_weights)
    trainer.train(train_loader, val_loader, config["epochs"], checkpoints_dir)
    
    # Evaluate
    logger.info("Evaluating on Gold Standard...")
    gold_data = load_gold_data(Path("data/gold_standard/gold_lid_dataset.parquet"))
    gold_dataset = SequenceDataset([s[0] for s in gold_data], [s[1] for s in gold_data], vocab, max_length=config["max_length"])
    gold_loader = DataLoader(gold_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model.load_state_dict(torch.load(checkpoints_dir / "best_model.pt"))
    model.eval()
    
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for seqs, labels, lengths in gold_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            logits, _ = model(seqs, lengths)
            preds = torch.argmax(logits, dim=-1)
            
            true_len = lengths[0].item()
            true_seq = labels[0][:true_len].cpu().numpy()
            pred_seq = preds[0][:true_len].cpu().numpy()
            
            for j in range(true_len):
                all_y_true.append(vocab.id2label[true_seq[j]])
                all_y_pred.append(vocab.id2label[pred_seq[j]])
                
    mac_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    report_str = classification_report(all_y_true, all_y_pred, zero_division=0)
    
    with open(exp_dir / "classification_report.txt", "w") as f:
        f.write(report_str)
        
    logger.info(f"Experiment 004 complete! Macro F1: {mac_f1:.4f}")

if __name__ == "__main__":
    run_experiment_004()
