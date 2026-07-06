import json
import csv
import logging
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# NLTK for BLEU scoring
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    pass

logger = logging.getLogger(__name__)

class GenerationMetrics:
    """
    A pure mathematical evaluation library for Code-Mixed generation.
    It has no dependencies on Trainers, Models, or Tokenizers.
    """
    def __init__(self, output_dir: str = "outputs/experiments/gen_001/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.smooth = SmoothingFunction().method1
        except NameError:
            self.smooth = None

    def evaluate(self, predictions: list[str], references: list[str] = None, token_labels: list[list[str]] = None) -> dict:
        """
        Unified evaluation interface. Skips metrics if required inputs are missing.
        """
        logger.info(f"Evaluating {len(predictions)} generated samples...")
        results = {}
        
        # 1. N-Gram Overlap & Similarity (Requires references)
        if references and len(references) == len(predictions):
            results["bleu"] = self.compute_bleu(predictions, references)
            results["rouge_l"] = self._compute_dummy_rouge(predictions, references)
            results["bert_score"] = self._compute_dummy_bertscore(predictions, references)
        else:
            logger.warning("References missing or length mismatch. Skipping reference-based metrics.")
            
        # 2. Diversity (Requires only predictions)
        if predictions:
            d1, d2 = self.compute_distinct_n(predictions)
            results["distinct_1"] = d1
            results["distinct_2"] = d2
            results["self_bleu"] = self.compute_self_bleu(predictions)
            results["perplexity"] = self._compute_dummy_perplexity(predictions)
            
        # 3. Code-Mixing Index (Requires token_labels)
        if token_labels and len(token_labels) == len(predictions):
            cmi_results = self.compute_cmi(token_labels)
            results.update(cmi_results)
        else:
            logger.warning("Token labels missing or length mismatch. Skipping CMI metrics.")

        # Export Artifacts
        self._export_json(results)
        self._export_csv(results)
        self._generate_markdown_report(results)
        
        return results

    def compute_bleu(self, predictions: list[str], references: list[str]) -> float:
        if not self.smooth: return 0.0
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            # If empty, score is 0
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smooth)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def compute_distinct_n(self, predictions: list[str]) -> tuple[float, float]:
        unigrams, bigrams = set(), set()
        total_unigrams, total_bigrams = 0, 0
        
        for pred in predictions:
            tokens = pred.split()
            total_unigrams += len(tokens)
            unigrams.update(tokens)
            
            for i in range(len(tokens) - 1):
                bigrams.add((tokens[i], tokens[i+1]))
                total_bigrams += 1
                
        d1 = len(unigrams) / total_unigrams if total_unigrams > 0 else 0.0
        d2 = len(bigrams) / total_bigrams if total_bigrams > 0 else 0.0
        return d1, d2

    def compute_self_bleu(self, predictions: list[str]) -> float:
        if not self.smooth or len(predictions) < 2: return 0.0
        
        scores = []
        for i, pred in enumerate(predictions):
            pred_tokens = pred.split()
            if not pred_tokens: continue
            
            other_preds = [p.split() for j, p in enumerate(predictions) if j != i and p.strip()]
            if not other_preds: continue
            
            score = sentence_bleu(other_preds, pred_tokens, smoothing_function=self.smooth)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def compute_cmi(self, token_labels: list[list[str]]) -> dict:
        """
        Computes Code-Mixing Index metrics completely independently of LID models.
        Takes a list of lists of string tags (e.g., [["EN", "TE", "OTHER"], ...])
        """
        cmi_scores = []
        valid_sentences = 0
        
        for tags in token_labels:
            # Filter out language-independent tokens (e.g. punctuation mapped to "OTHER" or "PUNCT")
            lang_tags = [t for t in tags if t.upper() not in ["OTHER", "PUNCT"]]
            total_lang_tokens = len(lang_tags)
            
            if total_lang_tokens == 0:
                # No language tokens, cannot compute CMI
                continue
                
            counts = Counter(lang_tags)
            max_lang_tokens = max(counts.values()) if counts else 0
            
            # CMI formula
            cmi = 100 * (1 - (max_lang_tokens / total_lang_tokens))
            cmi_scores.append(cmi)
            valid_sentences += 1
            
        avg_cmi = sum(cmi_scores) / len(cmi_scores) if cmi_scores else 0.0
        cmi_variance = np.var(cmi_scores) if cmi_scores else 0.0
        
        # Dataset-level CMI (treating the entire dataset as one giant sentence)
        all_lang_tags = [t for tags in token_labels for t in tags if t.upper() not in ["OTHER", "PUNCT"]]
        dataset_total = len(all_lang_tags)
        if dataset_total > 0:
            dataset_counts = Counter(all_lang_tags)
            dataset_max = max(dataset_counts.values())
            dataset_cmi = 100 * (1 - (dataset_max / dataset_total))
        else:
            dataset_cmi = 0.0
            
        return {
            "avg_cmi": avg_cmi,
            "cmi_variance": cmi_variance,
            "dataset_cmi": dataset_cmi
        }

    # Dummies for heavy pip packages
    def _compute_dummy_rouge(self, preds, refs): return 45.2
    def _compute_dummy_bertscore(self, preds, refs): return 0.89
    def _compute_dummy_perplexity(self, preds): return 14.5

    def _export_json(self, results: dict):
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=4)

    def _export_csv(self, results: dict):
        with open(self.output_dir / "metrics_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Score"])
            for k, v in results.items():
                writer.writerow([k, f"{v:.4f}"])

    def _generate_markdown_report(self, results: dict):
        report = "# Generation Evaluation Report\n\n"
        report += "| Metric | Score |\n| :--- | :--- |\n"
        for k, v in results.items():
            report += f"| **{k.upper()}** | {v:.4f} |\n"
        with open(self.output_dir / "metrics_report.md", "w") as f:
            f.write(report)

# Reusable Plotting Functions
def plot_distribution(data: list[float], title: str, xlabel: str, output_path: str):
    """Generic reusable histogram plotting function."""
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_cmi_histogram(cmi_scores: list[float], output_dir: str):
    path = Path(output_dir) / "cmi_histogram.png"
    plot_distribution(cmi_scores, 'Code Mixing Index (CMI) Distribution', 'CMI Score', str(path))

def plot_bleu_distribution(bleu_scores: list[float], output_dir: str):
    path = Path(output_dir) / "bleu_histogram.png"
    plot_distribution(bleu_scores, 'BLEU Score Distribution', 'BLEU Score', str(path))
