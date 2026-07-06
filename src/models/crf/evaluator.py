import logging
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class CRFEvaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, X_test, y_test, original_sentences, output_dir: Path):
        logger.info("Predicting on Gold Standard...")
        y_pred = self.model.predict(X_test)
        
        # Flatten sequences
        all_y_true = [label for seq in y_test for label in seq]
        all_y_pred = [label for seq in y_pred for label in seq]
        
        # Calculate metrics
        acc = accuracy_score(all_y_true, all_y_pred)
        mac_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        wt_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
        
        report_str = classification_report(all_y_true, all_y_pred, zero_division=0)
        
        metrics = {
            "accuracy": float(acc),
            "macro_f1": float(mac_f1),
            "weighted_f1": float(wt_f1)
        }
        
        # Save metrics
        import json
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write(report_str)
            
        # Predictions CSV
        predictions_rows = []
        for i, seq in enumerate(original_sentences):
            for j, token in enumerate(seq):
                predictions_rows.append({
                    "sentence_id": f"Gold_S_{i:03d}",
                    "token_id": j,
                    "token": token,
                    "gold_label": y_test[i][j],
                    "predicted_label": y_pred[i][j]
                })
        pd.DataFrame(predictions_rows).to_csv(output_dir / "predictions.csv", index=False)
        
        # Confusion Matrix
        labels = sorted(list(set(all_y_true) | set(all_y_pred)))
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - CRF')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
        
        logger.info(f"Evaluation complete. Macro F1: {mac_f1:.4f}")
        return metrics
