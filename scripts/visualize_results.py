import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_learning_curves(log_file: Path, output_dir: Path):
    """Parses train.log and plots Train vs Val Loss."""
    if not log_file.exists():
        return
    
    train_losses, val_losses = [], []
    with open(log_file, 'r') as f:
        for line in f:
            if "Train Loss:" in line and "Val Loss:" in line:
                parts = line.split("|")
                train_loss = float(parts[0].split("Train Loss:")[1].strip())
                val_loss = float(parts[1].split("Val Loss:")[1].strip())
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
    if train_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='o')
        plt.title('Learning Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "learning_curve.png")
        plt.close()

def plot_class_performance(report_file: Path, output_dir: Path):
    """Parses classification_report.txt and generates a bar chart for F1 scores."""
    if not report_file.exists():
        return
        
    classes, f1_scores = [], []
    with open(report_file, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            if line.strip() == "": break
            parts = line.split()
            if len(parts) >= 4:
                classes.append(parts[0])
                f1_scores.append(float(parts[3]))
                
    if classes:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=classes, y=f1_scores, palette="viridis")
        plt.title("Per-Class F1 Score Performance")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        plt.savefig(output_dir / "per_class_f1.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist.")
        return
        
    plot_learning_curves(exp_dir / "train.log", exp_dir)
    plot_class_performance(exp_dir / "classification_report.txt", exp_dir)
    print(f"Visualizations generated successfully in {exp_dir}")

if __name__ == "__main__":
    main()
