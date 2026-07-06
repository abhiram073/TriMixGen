import sys
import logging
from pathlib import Path
import random
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))
from src.models.crf.dataset import load_pseudo_labeled_data, load_gold_data
from src.models.bilstm.dataset import Vocabulary, SequenceDataset, collate_fn
from src.models.bilstm.model import BiLSTM_LID
from src.models.bilstm.trainer import BiLSTMTrainer
from sklearn.model_selection import train_test_split

def setup_logger(log_file: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
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

def run_experiment_003():
    exp_dir = Path("data/outputs/experiments/exp_003_bilstm")
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(exp_dir / "train.log")
    logger.info("Starting Experiment 003: BiLSTM")
    
    # 1. Config
    config = {
        "experiment_id": "003",
        "model_name": "BiLSTM",
        "dataset_version": "V1.1",
        "preprocessing_version": "V1.1",
        "gold_standard_version": "Gold Standard LID v1.0",
        "random_seed": 42,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 3,
        "early_stopping_patience": 3,
        "min_freq": 2,
        "max_length": 256
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Data Loading & Vocabulary
    logger.info("Loading pseudo-labeled training data...")
    data_dirs = [Path("data/processed")]
    # Using full dataset or sample. We'll use 25k to balance speed and learning capability.
    all_sentences_labels = load_pseudo_labeled_data(data_dirs, max_samples=2000)
    
    train_data, val_data = train_test_split(all_sentences_labels, test_size=0.1, random_state=config["random_seed"])
    
    train_sentences = [s[0] for s in train_data]
    train_labels = [s[1] for s in train_data]
    
    val_sentences = [s[0] for s in val_data]
    val_labels = [s[1] for s in val_data]
    
    vocab = Vocabulary(min_freq=config["min_freq"])
    vocab.build(train_sentences, train_labels)
    logger.info(f"Vocabulary built. Vocab size: {len(vocab.token2id)}. Classes: {len(vocab.label2id)}")
    
    # Create DataLoaders
    train_dataset = SequenceDataset(train_sentences, train_labels, vocab, max_length=config["max_length"])
    val_dataset = SequenceDataset(val_sentences, val_labels, vocab, max_length=config["max_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # 3. Model & Training
    model = BiLSTM_LID(
        vocab_size=len(vocab.token2id),
        num_classes=len(vocab.label2id),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    
    trainer = BiLSTMTrainer(model, config, device, vocab)
    train_losses, val_losses = trainer.train(train_loader, val_loader, config["epochs"], checkpoints_dir)
    
    # Plot learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('BiLSTM Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "learning_curve.png")
    plt.close()
    
    # 4. Final Evaluation on Gold Standard
    logger.info("Loading Gold Standard evaluation dataset...")
    gold_data = load_gold_data(Path("data/gold_standard/gold_lid_dataset.parquet"))
    gold_sentences = [s[0] for s in gold_data]
    gold_labels = [s[1] for s in gold_data]
    
    gold_dataset = SequenceDataset(gold_sentences, gold_labels, vocab, max_length=config["max_length"])
    gold_loader = DataLoader(gold_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load best model
    model.load_state_dict(torch.load(checkpoints_dir / "best_model.pt"))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    predictions_rows = []
    
    with torch.no_grad():
        for i, (seqs, labels, lengths) in enumerate(gold_loader):
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs, lengths)
            preds = torch.argmax(logits, dim=-1)
            
            # Remove padding
            true_len = lengths[0].item()
            true_seq = labels[0][:true_len].cpu().numpy()
            pred_seq = preds[0][:true_len].cpu().numpy()
            
            orig_sentence = gold_sentences[i]
            
            for j in range(true_len):
                gt_label = vocab.id2label[true_seq[j]]
                pt_label = vocab.id2label[pred_seq[j]]
                all_y_true.append(gt_label)
                all_y_pred.append(pt_label)
                
                predictions_rows.append({
                    "sentence_id": f"Gold_S_{i:03d}",
                    "token_id": j,
                    "token": orig_sentence[j],
                    "gold_label": gt_label,
                    "predicted_label": pt_label
                })
                
    # 5. Metrics & Outputs
    acc = accuracy_score(all_y_true, all_y_pred)
    mac_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    wt_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    
    report_str = classification_report(all_y_true, all_y_pred, zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(mac_f1),
        "weighted_f1": float(wt_f1)
    }
    
    import json
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(exp_dir / "classification_report.txt", "w") as f:
        f.write(report_str)
        
    pd.DataFrame(predictions_rows).to_csv(exp_dir / "predictions.csv", index=False)
    
    # Confusion Matrix
    unique_labels = sorted(list(set(all_y_true) | set(all_y_pred)))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - BiLSTM')
    plt.tight_layout()
    plt.savefig(exp_dir / "confusion_matrix.png")
    plt.close()
    
    # Model Summary
    with open(exp_dir / "model_summary.txt", "w") as f:
        f.write(str(model) + "\n")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"\nTotal Trainable Parameters: {total_params}\n")
        
    # Results Summary
    with open(exp_dir / "results_summary.md", "w") as f:
        f.write("# Experiment 003 Results Summary\n\n")
        f.write(f"**Accuracy:** {metrics['accuracy']:.4f}\n")
        f.write(f"**Macro F1:** {metrics['macro_f1']:.4f}\n")
        f.write(f"**Weighted F1:** {metrics['weighted_f1']:.4f}\n\n")
        f.write("## Notes\n")
        f.write("BiLSTM baseline trained on pseudo-labeled data and evaluated on Gold Standard v1.0. ")
        f.write("Captures semantic representations via embeddings and long-range context via recurrent gates.\n")
        
    logger.info(f"Experiment 003 complete! Macro F1: {mac_f1:.4f}")

if __name__ == "__main__":
    run_experiment_003()
