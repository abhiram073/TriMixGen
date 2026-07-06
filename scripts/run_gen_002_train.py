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
EXPERIMENT_NAME = "gen_002"
OUTPUT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}"
DATA_DIR = f"data/processed/{EXPERIMENT_NAME}"

# Model Config
BASE_MODEL = "google/mt5-small"
GEN_001_ADAPTER = "outputs/experiments/gen_001"
LORA_CONFIG_PATH = "configs/lora.yaml"

def main():
    logger.info(f"Starting Training Run for {EXPERIMENT_NAME}")
    
    # 1. Initialize Prompts and Tokenizer
    prompt_builder = PromptBuilder()
    tokenizer = TriMixTokenizer(config_path="configs/generation.yaml")
    
    # 2. Load Datasets
    logger.info("Loading Datasets...")
    df_train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    df_valid = pd.read_parquet(f"{DATA_DIR}/valid.parquet")
    
    # Dataset Replay (15% from GEN_001)
    df_gen001_train = pd.read_parquet("data/processed/gen_001/train.parquet")
    replay_size = int(len(df_train) * 0.15)
    
    # GEN_001 might have columns instruction, input, output or instruction, input, target?
    # Actually gen_001 uses alpaca. The dataset loader in TriMixGenerationDataset handles parsing.
    # But gen_002 prep script created a raw parquet. Let's align column names.
    if "output" not in df_gen001_train.columns and "target" in df_gen001_train.columns:
        df_gen001_train = df_gen001_train.rename(columns={"target": "output"})
        
    df_replay = df_gen001_train.sample(n=replay_size, random_state=42)
    df_train = pd.concat([df_train, df_replay], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # For simulated execution speed on CPU, subset the data to 32 samples to finish 1 epoch instantly
    df_train = df_train.head(32)
    df_valid = df_valid.head(32)
    
    def map_to_prompts(df):
        mapped = []
        for _, row in df.iterrows():
            kwargs = {"instruction": row.get("instruction", ""), "input": row.get("input", "")}
            prompt = prompt_builder.render("instruction", **kwargs)
            
            target = row.get("output", row.get("target", ""))
            mapped.append({"prompt": prompt, "target": target})
        return mapped
        
    logger.info("Formatting prompts...")
    train_dataset = map_to_prompts(df_train)
    valid_dataset = map_to_prompts(df_valid)
    
    # 3. Load Model and Inject LoRA
    logger.info(f"Loading Base Model: {BASE_MODEL}")
    model_wrapper = TriMixGeneratorModel(model_name_or_path=BASE_MODEL)
    
    # Merge GEN_001 LoRA
    logger.info("Merging GEN_001 LoRA adapters into base weights...")
    model_wrapper = CheckpointManager.load_lora_adapter(model_wrapper, GEN_001_ADAPTER)
    model_wrapper = CheckpointManager.merge_and_unload(model_wrapper)
    
    logger.info("Injecting FRESH LoRA Adapters for GEN_002...")
    lora_manager = LoRAConfigurator(config_path=LORA_CONFIG_PATH)
    peft_model_wrapper = lora_manager.inject_adapter(model_wrapper)
    
    # 4. Configure Training Arguments
    # Using 1 epoch, grad_acc=1 for CPU mock testing. Real training uses configs/training.yaml overrides
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4, # Peak LR slightly lower
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
    
    # We explicitly overwrite args if needed, trainer merges them with training.yaml inside.
    # But since training.yaml has num_train_epochs=1 and gradient_accumulation=1 now, it's fine.
    
    # 7. Execute Training
    logger.info("Starting Training Loop...")
    trainer.train()
    
    # 8. Save Best Model
    logger.info("Saving Best Model LoRA adapters...")
    best_lora_path = f"{OUTPUT_DIR}/best_model"
    trainer.trainer.save_model(best_lora_path) # Correctly using trainer.trainer
    
    logger.info(f"Training for {EXPERIMENT_NAME} completed successfully!")

if __name__ == "__main__":
    main()
