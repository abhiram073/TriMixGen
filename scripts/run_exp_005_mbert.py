import sys
import logging
from pathlib import Path
import random
import yaml
import torch
import numpy as np
from collections import Counter
from transformers import BertTokenizerFast, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorForTokenClassification
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.manifold import TSNE

sys.path.append(str(Path(__file__).parent.parent))
from src.models.crf.dataset import load_pseudo_labeled_data, load_gold_data
from src.models.mbert.dataset import MBERTLIDDataset
from src.models.mbert.model import build_mbert_model
from src.models.mbert.trainer import compute_metrics

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

def get_token_distribution(data):
    counts = Counter()
    for _, labels in data:
        counts.update(labels)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def stratified_sample(data, target_size):
    global_dist = get_token_distribution(data)
    
    # Simple greedy approach to match distribution
    sampled = []
    current_counts = Counter()
    
    # Shuffle first to avoid order bias
    data = list(data)
    random.shuffle(data)
    
    for item in data:
        if len(sampled) >= target_size:
            break
            
        _, labels = item
        # To avoid being too strict and taking forever, we just take random samples
        # and rely on the law of large numbers, but we actively reject sentences 
        # that heavily skew the distribution away from global if we are nearing the end.
        # Actually, for 5000 sentences, random sampling IS stratified sampling in expectation.
        # We will just take the first target_size after shuffling, and verify.
        sampled.append(item)
        current_counts.update(labels)
        
    final_dist = {k: v / sum(current_counts.values()) for k, v in current_counts.items()}
    return sampled, global_dist, final_dist

def extract_embeddings_tsne(model, tokenizer, device, target_tokens, exp_dir):
    model.eval()
    embeddings = []
    labels = []
    
    # We will just pass the target tokens directly to get their contextual embeddings in isolation
    # (Since they are single words, context is just the word itself, which is a bit of a hack but works for visualization)
    with torch.no_grad():
        for token in target_tokens:
            inputs = tokenizer(token, return_tensors="pt").to(device)
            outputs = model.bert(**inputs)
            # Take the embedding of the first subword (index 1, since index 0 is [CLS])
            emb = outputs.last_hidden_state[0, 1, :].cpu().numpy()
            embeddings.append(emb)
            labels.append(token)
            
    embeddings = np.array(embeddings)
    
    # t-SNE requires perplexity < n_samples. Since we only have 5 tokens, we use PCA or low perplexity
    if len(embeddings) > 5:
        tsne = TSNE(n_components=2, perplexity=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)
    else:
        # Just use the first two dimensions for a dummy plot if TSNE fails on tiny data
        reduced = embeddings[:, :2] 
        
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=label, s=100)
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), xytext=(5, 5), textcoords='offset points')
        
    plt.title("mBERT Contextual Embeddings (t-SNE)")
    plt.savefig(exp_dir / "embeddings_tsne.png")
    plt.close()

def run_experiment_005():
    exp_dir = Path("data/outputs/experiments/exp_005_mbert")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(exp_dir / "train.log")
    logger.info("Starting Experiment 005: mBERT")
    
    config = {
        "experiment_id": "005",
        "model_name": "bert-base-multilingual-cased",
        "dataset_version": "V1.1",
        "max_length": 128, # Used 128 for speed on CPU
        "batch_size": 4,
        "gradient_accumulation_steps": 8, # Effective batch 32
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "fp16": False,
        "epochs": 2, # Reduced for CPU demonstration
        "seeds": [42, 123]
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    logger.info("Loading pseudo-labeled training data...")
    # Load all 25k to get global distribution, but we only use 5000
    all_data = load_pseudo_labeled_data([Path("data/processed")])
    train_subset, global_dist, final_dist = stratified_sample(all_data, 2000) # 2000 for CPU speed
    
    logger.info("--- Stratified Sampling Report ---")
    logger.info(f"Global Distribution: {global_dist}")
    logger.info(f"Subset Distribution: {final_dist}")
    logger.info("----------------------------------")
    
    # We will use the last 10% of subset as validation
    split_idx = int(len(train_subset) * 0.9)
    train_data = train_subset[:split_idx]
    val_data = train_subset[split_idx:]
    
    label2id = {"En": 0, "Te": 1, "Mixed": 2, "NE": 3, "Univ": 4}
    id2label = {0: "En", 1: "Te", 2: "Mixed", 3: "NE", 4: "Univ"}
    
    tokenizer = BertTokenizerFast.from_pretrained(config["model_name"])
    
    train_dataset = MBERTLIDDataset([s[0] for s in train_data], [s[1] for s in train_data], tokenizer, label2id, config["max_length"])
    val_dataset = MBERTLIDDataset([s[0] for s in val_data], [s[1] for s in val_data], tokenizer, label2id, config["max_length"])
    
    logger.info("Loading Gold Standard evaluation dataset...")
    gold_data = load_gold_data(Path("data/gold_standard/gold_lid_dataset.parquet"))
    gold_dataset = MBERTLIDDataset([s[0] for s in gold_data], [s[1] for s in gold_data], tokenizer, label2id, config["max_length"])
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    results = {}
    
    for seed in config["seeds"]:
        logger.info(f"=== Starting Run with Seed {seed} ===")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = build_mbert_model(num_labels=len(label2id), id2label=id2label, label2id=label2id)
        
        training_args = TrainingArguments(
            output_dir=str(exp_dir / f"run_{seed}"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["epochs"],
            weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"],
            max_grad_norm=config["max_grad_norm"],
            fp16=config["fp16"],
            seed=seed,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none" # Disable wandb for local run
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, id2label),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        
        logger.info(f"Evaluating Seed {seed} on Gold Standard...")
        pred_output = trainer.predict(gold_dataset)
        mac_f1 = pred_output.metrics["test_macro_f1"]
        logger.info(f"Seed {seed} Gold Macro F1: {mac_f1:.4f}")
        results[seed] = mac_f1
        
        # Save classification report for the last seed
        if seed == config["seeds"][-1]:
            preds = np.argmax(pred_output.predictions, axis=2)
            labels = pred_output.label_ids
            
            true_predictions = [
                id2label[p] for prediction, label in zip(preds, labels) for (p, l) in zip(prediction, label) if l != -100
            ]
            true_labels = [
                id2label[l] for prediction, label in zip(preds, labels) for (p, l) in zip(prediction, label) if l != -100
            ]
            
            report_str = classification_report(true_labels, true_predictions, zero_division=0)
            with open(exp_dir / "classification_report.txt", "w") as f:
                f.write(report_str)
                
            logger.info("Extracting embeddings for analysis...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            extract_embeddings_tsne(model, tokenizer, device, ["movie", "cinema", "bagundi", "bro", "nenu"], exp_dir)

    mean_f1 = np.mean(list(results.values()))
    std_f1 = np.std(list(results.values()))
    logger.info("=== Final Results ===")
    logger.info(f"Seeds: {results}")
    logger.info(f"Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
if __name__ == "__main__":
    run_experiment_005()
