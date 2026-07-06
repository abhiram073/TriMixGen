import sys
import logging
from pathlib import Path
import pandas as pd
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

sys.path.append(str(Path(__file__).parent.parent))
from src.features.language_annotator import LanguageAnnotator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment_001():
    # 1. Setup paths
    exp_dir = Path("data/outputs/experiments/exp_001_rule_based")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    gold_file = Path("data/processed/gold_standard.csv")
    if not gold_file.exists():
        logger.error("Gold standard missing!")
        return
        
    gold_df = pd.read_csv(gold_file)
    
    # 2. Reconstruct sentences from gold
    sentences = []
    gold_labels = []
    
    current_sent_id = None
    curr_tokens = []
    curr_labels = []
    
    for _, row in gold_df.iterrows():
        sid = row["sentence_id"]
        if sid != current_sent_id:
            if current_sent_id is not None:
                sentences.append(curr_tokens)
                gold_labels.append(curr_labels)
            current_sent_id = sid
            curr_tokens = []
            curr_labels = []
            
        curr_tokens.append(str(row["token"]))
        curr_labels.append(str(row["gold_label"]))
        
    if curr_tokens:
        sentences.append(curr_tokens)
        gold_labels.append(curr_labels)
        
    # 3. Run Inference using Rule-Based V1.1
    annotator = LanguageAnnotator(config_path="configs/preprocessing.yaml")
    
    pred_labels = []
    pred_confidences = []
    
    all_y_true = []
    all_y_pred = []
    predictions_rows = []
    
    logger.info("Running inference...")
    for i, tokens in enumerate(sentences):
        ann_res = annotator.annotate_sentence(tokens, dataset_name="hold")
        
        preds = ann_res["labels"]
        confs = ann_res["confidences"]
        
        for j, tok in enumerate(tokens):
            gt = gold_labels[i][j]
            pt = preds[j]
            cf = confs[j]
            
            all_y_true.append(gt)
            all_y_pred.append(pt)
            
            predictions_rows.append({
                "sentence_id": f"S_{i:03d}",
                "token_id": j,
                "token": tok,
                "gold_label": gt,
                "predicted_label": pt,
                "confidence": cf
            })
            
    # 4. Calculate Metrics
    acc = accuracy_score(all_y_true, all_y_pred)
    mac_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    wt_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    
    report_dict = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(all_y_true, all_y_pred, zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(mac_f1),
        "weighted_f1": float(wt_f1)
    }
    
    # 5. Save Outputs
    # a. config.yaml
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump({
            "experiment_id": "001",
            "model_name": "Rule_Based_Heuristic",
            "dataset_version": "V1.1",
            "preprocessing_version": "V1.1",
            "hyperparameters": annotator.config
        }, f)
        
    # b. metrics.json
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # c. classification_report.txt
    with open(exp_dir / "classification_report.txt", "w") as f:
        f.write(report_str)
        
    # d. predictions.csv
    pd.DataFrame(predictions_rows).to_csv(exp_dir / "predictions.csv", index=False)
    
    # e. confusion_matrix.png
    labels = sorted(list(set(all_y_true) | set(all_y_pred)))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Rule-Based Baseline')
    plt.tight_layout()
    plt.savefig(exp_dir / "confusion_matrix.png")
    plt.close()
    
    # f. model_summary.txt
    with open(exp_dir / "model_summary.txt", "w") as f:
        f.write("Model: 7-Tier Heuristic Waterfall (TriMixGen Phase 4)\n")
        f.write("Parameters: 0 (Deterministic rules + Dictionary lookup)\n")
        
    # g. results_summary.md
    with open(exp_dir / "results_summary.md", "w") as f:
        f.write("# Experiment 001 Results Summary\n\n")
        f.write(f"**Accuracy:** {acc:.4f}\n")
        f.write(f"**Macro F1:** {mac_f1:.4f}\n")
        f.write(f"**Weighted F1:** {wt_f1:.4f}\n\n")
        f.write("## Notes\n")
        f.write("Zero-shot baseline evaluating the regex and dictionary heuristics on the 250-sentence gold standard. ")
        f.write("Misclassifications represent cases where pure contextual inference (which rule-based lacks) is necessary.\n")
        
    logger.info(f"Experiment 001 complete! Macro F1: {mac_f1:.4f}")

if __name__ == "__main__":
    run_experiment_001()
