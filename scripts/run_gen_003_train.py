import os
import logging
import pandas as pd
from transformers import Seq2SeqTrainingArguments

from src.models.generation.tokenizer import TriMixTokenizer
from src.models.generation.prompt_builder import PromptBuilder
from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.lora_config import LoRAConfigurator
from src.models.generation.trainer import GenerationTrainer
from src.models.generation.callbacks import TriMixLoggingCallback, CustomEarlyStoppingCallback
from src.models.generation.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Experiment Config
EXPERIMENT_NAME = "gen_003"
OUTPUT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}"
DATA_DIR = f"data/processed/{EXPERIMENT_NAME}"

# Model Config
BASE_MODEL = "google/mt5-small"
GEN_001_ADAPTER = "outputs/experiments/gen_001/best_model"
GEN_002_ADAPTER = "outputs/experiments/gen_002/best_model"
LORA_CONFIG_PATH = "configs/lora.yaml"

def get_replay_dataset(dataset_path: str, frac: float, num_samples: int) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    # TriMixGenerationDataset mappings
    if "target" in df.columns and "output" not in df.columns:
        df = df.rename(columns={"target": "output"})
    n_samples = min(int(num_samples * frac), len(df))
    return df.sample(n=n_samples, random_state=42)

def main():
    logger.info(f"Starting Training Run for {EXPERIMENT_NAME}")
    
    # 1. Initialize Prompts and Tokenizer
    prompt_builder = PromptBuilder()
    tokenizer = TriMixTokenizer(config_path="configs/generation.yaml")
    
    # 2. Load Datasets and Replay
    logger.info("Loading Datasets and Applying Dual-Replay Strategy...")
    df_gen003_train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    df_valid = pd.read_parquet(f"{DATA_DIR}/valid.parquet")
    
    # Let's say GEN_003 is 80%. We need 10% GEN_001 and 10% GEN_002 relative to total.
    # So replay total is (20/80) = 25% of GEN_003 size.
    total_new_samples = len(df_gen003_train)
    replay_each = int(total_new_samples * 0.125)
    
    df_gen001_replay = get_replay_dataset("data/processed/gen_001/train.parquet", 1.0, replay_each)
    df_gen002_replay = get_replay_dataset("data/processed/gen_002/train.parquet", 1.0, replay_each)
    
    df_train = pd.concat([df_gen003_train, df_gen001_replay, df_gen002_replay], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # For simulated execution speed on CPU, subset the data to 32 samples
    df_train = df_train.head(32)
    df_valid = df_valid.head(32)
    
    def map_to_prompts(df):
        mapped = []
        for _, row in df.iterrows():
            kwargs = {"instruction": row.get("instruction", ""), "input": row.get("input", "")}
            prompt = prompt_builder.render("instruction", **kwargs)
            target = row.get("output", "")
            mapped.append({"prompt": prompt, "target": target})
        return mapped
        
    logger.info("Formatting prompts...")
    train_dataset = map_to_prompts(df_train)
    valid_dataset = map_to_prompts(df_valid)
    
    # 3. Load Model and Inject LoRA
    logger.info(f"Loading Base Model: {BASE_MODEL}")
    model_wrapper = TriMixGeneratorModel(model_name_or_path=BASE_MODEL)
    
    # Merge GEN_001 and GEN_002 LoRAs
    logger.info("Merging GEN_001 LoRA adapters into base weights...")
    # NOTE: In our mock setup, gen_001 might just be in outputs/experiments/gen_001
    gen_001_path = GEN_001_ADAPTER if os.path.exists(GEN_001_ADAPTER) else "outputs/experiments/gen_001"
    model_wrapper = CheckpointManager.load_lora_adapter(model_wrapper, gen_001_path)
    model_wrapper = CheckpointManager.merge_and_unload(model_wrapper)
    
    logger.info("Merging GEN_002 LoRA adapters into base weights...")
    gen_002_path = GEN_002_ADAPTER if os.path.exists(GEN_002_ADAPTER) else "outputs/experiments/gen_002"
    model_wrapper = CheckpointManager.load_lora_adapter(model_wrapper, gen_002_path)
    model_wrapper = CheckpointManager.merge_and_unload(model_wrapper)
    
    logger.info("Injecting FRESH LoRA Adapters for GEN_003...")
    lora_manager = LoRAConfigurator(config_path=LORA_CONFIG_PATH)
    peft_model_wrapper = lora_manager.inject_adapter(model_wrapper)
    
    # 4. Configure Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4, # Peak LR for style control
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_ratio=0.1,
        seed=42,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=256,
        remove_unused_columns=False,
        report_to=["tensorboard"]
    )
    
    # 5. Load Callbacks
    callbacks = [
        TriMixLoggingCallback(output_dir=OUTPUT_DIR),
        CustomEarlyStoppingCallback(patience=3)
    ]
    
    # 6. Initialize Trainer
    logger.info("Initializing GenerationTrainer...")
    trainer = GenerationTrainer(
        model_wrapper=peft_model_wrapper,
        tokenizer_wrapper=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=callbacks
    )
    
    # 7. Execute Training
    logger.info("Starting Training Loop...")
    trainer.train()
    
    # 8. Save Best Model
    logger.info("Saving Best Model LoRA adapters...")
    best_lora_path = f"{OUTPUT_DIR}/best_model"
    trainer.trainer.save_model(best_lora_path)
    
    # Deployment Export Manifest
    logger.info("Generating Deployment Manifest...")
    manifest = {
        "production_model": "google/mt5-small",
        "merged_adapters": ["outputs/experiments/gen_001", "outputs/experiments/gen_002/best_model"],
        "active_adapter": best_lora_path,
        "generation_config": "configs/generation.yaml",
        "prompt_templates": "configs/prompts.yaml",
        "tokenizer_config": "configs/tokenizer.yaml"
    }
    import yaml
    with open(f"{OUTPUT_DIR}/deployment_manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    
    logger.info(f"Training for {EXPERIMENT_NAME} completed successfully!")

if __name__ == "__main__":
    main()
