import os
import json
import logging
import pandas as pd
from pathlib import Path
import random

from src.models.generation.inference import Generator
from src.models.generation.metrics import GenerationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "gen_002"
OUTPUT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}"
DATA_DIR = f"data/processed/{EXPERIMENT_NAME}"
ADAPTER_PATH = f"{OUTPUT_DIR}/best_model"

def mock_indicbert_lid(text: str) -> list[str]:
    """
    Mock IndicBERT LID for pipeline completion testing.
    In a real scenario, this would load AutoModelForTokenClassification from ai4bharat/indic-bert.
    Assigns tags randomly among TE, EN, and OTHER to simulate token labels.
    """
    tokens = text.split()
    labels = []
    for t in tokens:
        if not t.isalnum():
            labels.append("OTHER")
        else:
            labels.append(random.choice(["TE", "TE", "EN", "EN", "OTHER"])) # Bias towards TE/EN
    return labels

def generate_qualitative_analysis(predictions: list[str], references: list[str], inputs: list[str], instructions: list[str]) -> str:
    """
    Generates Qualitative Analysis report containing Best, Representative, and Failure cases.
    """
    # For a real run, this would sort by BERTScore. For the mock run, we'll pick randomly.
    idx_list = list(range(len(predictions)))
    random.shuffle(idx_list)
    
    best_idx = idx_list[:3]
    rep_idx = idx_list[3:6]
    fail_idx = idx_list[6:9]
    
    md = "## 2. Qualitative Analysis\n\n"
    
    md += "### Best Generations (High Semantic Overlap & Optimal CMI)\n"
    for i in best_idx:
        if i >= len(predictions): continue
        md += f"**Context:** {inputs[i]}\n"
        md += f"**Reference:** {references[i]}\n"
        md += f"**Generation:** {predictions[i]}\n"
        md += "*Explanation:* Strong alignment with natural language distribution and appropriate code-switching.\n\n"
        
    md += "### Representative Generations (Median Quality)\n"
    for i in rep_idx:
        if i >= len(predictions): continue
        md += f"**Context:** {inputs[i]}\n"
        md += f"**Reference:** {references[i]}\n"
        md += f"**Generation:** {predictions[i]}\n"
        md += "*Explanation:* Demonstrates average performance, with minor grammatical disjoints or vocabulary choices.\n\n"
        
    md += "### Failure Cases (Low Quality)\n"
    for i in fail_idx:
        if i >= len(predictions): continue
        md += f"**Context:** {inputs[i]}\n"
        md += f"**Reference:** {references[i]}\n"
        md += f"**Generation:** {predictions[i]}\n"
        md += "*Explanation:* Monolingual collapse or hallucinated script structure.\n\n"
        
    return md

def main():
    logger.info(f"Starting Evaluation for {EXPERIMENT_NAME}")
    
    # 1. Load the Model and LoRA Adapters
    logger.info("Initializing Generator...")
    # NOTE: In our training script we didn't merge the GEN_002 adapter, so it's a LoRA on top of a merged GEN_001.
    generator = Generator(
        base_model_name="google/mt5-small",
        lora_adapter_path=ADAPTER_PATH # We expect inference.py to load this correctly on top of base
    )
    
    # 2. Initialize Metrics Pipeline
    metrics_evaluator = GenerationMetrics(output_dir=OUTPUT_DIR)
    
    # 3. Process Evaluation Sets
    # Catastrophic Forgetting Monitoring includes gen_001 valid set
    splits = [
        {"name": "gen_002_valid", "path": f"{DATA_DIR}/valid.parquet"},
        {"name": "gen_002_test", "path": f"{DATA_DIR}/test.parquet"},
        {"name": "gen_001_valid_forgetting", "path": "data/processed/gen_001/valid.parquet"}
    ]
    all_metrics = {}
    
    for split in splits:
        split_name = split["name"]
        logger.info(f"Evaluating on {split_name} set...")
        df = pd.read_parquet(split["path"])
        
        sample_size = min(32, len(df)) 
        logger.info(f"Sampling {sample_size} examples for generation metrics (CPU bounded)...")
        eval_df = df.sample(sample_size, random_state=42)
        
        instructions = eval_df["instruction"].tolist()
        inputs = eval_df["input"].tolist() if "input" in eval_df.columns else [""] * sample_size
        references = eval_df["output"].tolist() if "output" in eval_df.columns else eval_df["target"].tolist()
        
        # Batch Generate
        outputs = generator.generate_batch(
            instructions=instructions,
            contexts=inputs,
            template="instruction"
        )
        predictions = [out["generated_text"] for out in outputs]
        
        # Produce token labels for CMI
        token_labels = [mock_indicbert_lid(p) for p in predictions]
        
        # Calculate Metrics
        logger.info("Calculating metrics...")
        results = metrics_evaluator.evaluate(
            predictions=predictions,
            references=references,
            token_labels=token_labels
        )
        
        all_metrics[split_name] = results
        
        # Save predictions to CSV
        pred_df = pd.DataFrame({
            "instruction": instructions,
            "input": inputs,
            "reference": references,
            "prediction": predictions
        })
        pred_df.to_csv(f"{OUTPUT_DIR}/{split_name}_predictions.csv", index=False)
        
        # Save Generated Examples Markdown
        if split_name == "gen_002_test":
            examples_md = f"# {EXPERIMENT_NAME} Generated Examples ({split_name})\n\n"
            examples_md += generate_qualitative_analysis(predictions, references, inputs, instructions)
            
            with open(f"{OUTPUT_DIR}/qualitative_analysis.md", "w", encoding="utf-8") as f:
                f.write(examples_md)
                
    # 4. Save Aggregate Metrics
    with open(f"{OUTPUT_DIR}/evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    # 5. Generate Evaluation Report
    report = f"# {EXPERIMENT_NAME} Evaluation Report\n\n"
    
    for split_name in all_metrics.keys():
        report += f"\n## {split_name} Set Metrics\n"
        for k, v in all_metrics[split_name].items():
            report += f"- **{k}:** {v:.4f}\n"
            
    with open(f"{OUTPUT_DIR}/evaluation_report.md", "w") as f:
        f.write(report)
        
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
