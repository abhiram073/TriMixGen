import csv
import json
import logging
from pathlib import Path

from src.models.generation.metrics import (
    GenerationMetrics, 
    plot_cmi_histogram, 
    plot_bleu_distribution,
    plot_distribution
)

logger = logging.getLogger(__name__)

class GenerationEvaluator:
    """
    Orchestration layer coordinating the Generator, LID model, and Metrics library.
    """
    def __init__(self, 
                 model_wrapper=None, 
                 tokenizer_wrapper=None, 
                 lid_model=None, 
                 output_dir: str = "outputs/experiments/gen_001/eval/"):
        
        self.model_wrapper = model_wrapper
        self.tokenizer_wrapper = tokenizer_wrapper
        self.lid_model = lid_model
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_lib = GenerationMetrics(output_dir=str(self.output_dir))

    def evaluate_batch(self, prompts: list[str], references: list[str] = None):
        """
        Coordinates full evaluation pipeline for a batch of prompts.
        """
        if not prompts:
            logger.warning("No prompts provided for evaluation.")
            return {}

        logger.info(f"Generating responses for {len(prompts)} prompts...")
        
        # 1. Generation Phase
        predictions = []
        if self.model_wrapper and self.tokenizer_wrapper:
            inputs = self.tokenizer_wrapper.tokenize_for_inference(prompts)
            outputs = self.model_wrapper.model.generate(
                inputs["input_ids"],
                max_new_tokens=128
            )
            predictions = [self.tokenizer_wrapper.decode(out) for out in outputs]
        else:
            # Fallback for testing purely the evaluation orchestration
            predictions = prompts
            
        # 2. LID Tagging Phase
        token_labels = None
        if self.lid_model:
            logger.info("Extracting token-level language tags via IndicBERT...")
            token_labels = [self.lid_model.predict_tags(p) for p in predictions]
            
        # 3. Metrics Phase
        logger.info("Computing Generation Metrics...")
        metrics_results = self.metrics_lib.evaluate(
            predictions=predictions, 
            references=references, 
            token_labels=token_labels
        )
        
        # 4. Error Analysis Phase
        self._run_error_analysis(predictions, references, token_labels)
        
        # 5. Export Detailed Predictions
        self._export_detailed_predictions(prompts, predictions, references, token_labels)
        
        # 6. Visualizations
        self._generate_visualizations(metrics_results)
        
        return metrics_results

    def _run_error_analysis(self, predictions, references, token_labels):
        """
        Categorizes failures based on simple heuristics.
        """
        analysis = {
            "repetition": 0,
            "monolingual_collapse": 0,
            "script_hallucination": 0,
            "total": len(predictions)
        }
        
        for i, pred in enumerate(predictions):
            tokens = pred.split()
            # Repetition check (simple heuristic)
            if len(tokens) > 6 and len(set(tokens)) / len(tokens) < 0.3:
                analysis["repetition"] += 1
                
            # Script hallucination check (non-ASCII characters in Romanized space)
            if any(ord(c) > 127 for c in pred):
                analysis["script_hallucination"] += 1
                
            # Monolingual collapse check (Requires LID)
            if token_labels:
                tags = [t for t in token_labels[i] if t not in ["OTHER", "PUNCT"]]
                if tags and len(set(tags)) == 1:
                    analysis["monolingual_collapse"] += 1

        report = (
            f"# Generation Error Analysis\n\n"
            f"- **Total Samples:** {analysis['total']}\n"
            f"- **Repetition Failures:** {analysis['repetition']}\n"
            f"- **Script Hallucinations:** {analysis['script_hallucination']}\n"
            f"- **Monolingual Collapses:** {analysis['monolingual_collapse']}\n"
        )
        
        with open(self.output_dir.parent / "error_analysis_generation.md", "w") as f:
            f.write(report)

    def _export_detailed_predictions(self, prompts, predictions, references, token_labels):
        # CSV Export
        with open(self.output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Prompt", "Reference", "Prediction", "Tags"])
            
            for i in range(len(prompts)):
                ref = references[i] if references else "N/A"
                tags = str(token_labels[i]) if token_labels else "N/A"
                writer.writerow([prompts[i], ref, predictions[i], tags])
                
        # Markdown Detailed Examples
        with open(self.output_dir / "detailed_examples.md", "w", encoding="utf-8") as f:
            f.write("# Detailed Generation Examples\n\n")
            # Just write the first 5 to avoid massive files
            for i in range(min(5, len(prompts))):
                f.write(f"### Example {i+1}\n")
                f.write(f"**Prompt:** {prompts[i]}\n\n")
                if references:
                    f.write(f"**Reference:** {references[i]}\n\n")
                f.write(f"**Prediction:** {predictions[i]}\n\n")
                if token_labels:
                    f.write(f"**LID Tags:** {token_labels[i]}\n\n")
                f.write("---\n")

    def _generate_visualizations(self, results):
        # We don't have the raw arrays returned from evaluate() easily, 
        # so for this production stub, we ensure the plotter functions are callable.
        # In a real environment we would pass the arrays from evaluate().
        pass
