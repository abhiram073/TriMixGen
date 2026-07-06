import sys
import logging
from pathlib import Path
import random
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from src.models.crf.dataset import load_pseudo_labeled_data, load_gold_data, prepare_crf_data
from src.models.crf.trainer import CRFTrainer
from src.models.crf.evaluator import CRFEvaluator
from sklearn.model_selection import train_test_split

def setup_logger(log_file: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def run_experiment_002():
    # Setup paths
    exp_dir = Path("data/outputs/experiments/exp_002_crf")
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(exp_dir / "train.log")
    
    # Global random seed
    random.seed(42)
    
    logger.info("Starting Experiment 002: CRF")
    
    # Config
    config = {
        "experiment_id": "002",
        "model_name": "Conditional Random Field (CRF)",
        "dataset_version": "V1.1",
        "preprocessing_version": "V1.1",
        "gold_standard_version": "Gold Standard LID v1.0",
        "random_seed": 42,
        "c1": 0.1,
        "c2": 0.1,
        "max_iterations": 100
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    # Load Training Data (pseudo-labeled)
    data_dirs = [Path("data/processed")]
    logger.info("Loading pseudo-labeled data...")
    # To save time and memory for CRF baseline, let's train on 10,000 sentences
    sentences = load_pseudo_labeled_data(data_dirs, max_samples=10000)
    
    logger.info(f"Loaded {len(sentences)} training sentences.")
    X_all, y_all = prepare_crf_data(sentences)
    
    # Train / Val Split
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
    
    # Train
    trainer = CRFTrainer(config)
    trainer.train(X_train, y_train)
    
    trainer.save_model(str(checkpoints_dir / "crf_model.pkl"))
    
    # Load Gold Standard for Final Eval
    logger.info("Loading Gold Standard evaluation dataset...")
    gold_sentences = load_gold_data(Path("data/gold_standard/gold_lid_dataset.parquet"))
    X_test, y_test = prepare_crf_data(gold_sentences)
    original_sentences = [s[0] for s in gold_sentences]
    
    # Evaluate
    evaluator = CRFEvaluator(trainer.model)
    metrics = evaluator.evaluate(X_test, y_test, original_sentences, exp_dir)
    
    # Model Summary
    with open(exp_dir / "model_summary.txt", "w") as f:
        f.write("Model: Conditional Random Field (CRF)\n")
        f.write(f"Transitions computed: {len(trainer.model.transition_features_)}\n")
        f.write(f"State features computed: {len(trainer.model.state_features_)}\n")
        
    # Results Summary
    with open(exp_dir / "results_summary.md", "w") as f:
        f.write("# Experiment 002 Results Summary\n\n")
        f.write(f"**Accuracy:** {metrics['accuracy']:.4f}\n")
        f.write(f"**Macro F1:** {metrics['macro_f1']:.4f}\n")
        f.write(f"**Weighted F1:** {metrics['weighted_f1']:.4f}\n\n")
        f.write("## Notes\n")
        f.write("CRF baseline trained on pseudo-labeled data and evaluated on Gold Standard v1.0. ")
        f.write("This establishes our first sequence labeling baseline.\n")

if __name__ == "__main__":
    run_experiment_002()
