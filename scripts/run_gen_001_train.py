import os
import logging
from transformers import Seq2SeqTrainingArguments

from src.models.generation.tokenizer import TriMixTokenizer
from src.models.generation.prompt_builder import PromptBuilder
from src.models.generation.dataset import TriMixGenerationDataset
from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.lora_config import LoRAConfigurator
from src.models.generation.trainer import GenerationTrainer
from src.models.generation.callbacks import TriMixLoggingCallback, CustomEarlyStoppingCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Experiment Config
EXPERIMENT_NAME = "gen_001"
OUTPUT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}"
DATA_DIR = f"data/processed/{EXPERIMENT_NAME}"

# Model Config
BASE_MODEL = "google/mt5-small"
LORA_CONFIG_PATH = "configs/lora.yaml"

def main():
    logger.info(f"Starting Training Run for {EXPERIMENT_NAME}")
    
    # 1. Initialize Prompts and Tokenizer
    prompt_builder = PromptBuilder()
    tokenizer = TriMixTokenizer(config_path="configs/generation.yaml")
    
    # 2. Load Datasets
    logger.info("Loading Datasets...")
    raw_train = TriMixGenerationDataset(data_path=f"{DATA_DIR}/train.parquet", dataset_type="alpaca")
    raw_valid = TriMixGenerationDataset(data_path=f"{DATA_DIR}/valid.parquet", dataset_type="alpaca")
    
    # Map raw items to standard dataset format expected by the trainer
    # The trainer data collator expects f["prompt"] and f["target"]
    # For simulated execution speed on CPU, subset the data to 32 samples to finish 1 epoch instantly
    raw_train = raw_train[:32]
    raw_valid = raw_valid[:32]
    
    def map_to_prompts(raw_dataset):
        mapped = []
        for item in raw_dataset:
            # instruction template expects 'instruction' and 'input'
            kwargs = {"instruction": item.get("instruction", ""), "input": item.get("input", "")}
            prompt = prompt_builder.render("instruction", **kwargs)
            mapped.append({"prompt": prompt, "target": item["target"]})
        return mapped
        
    logger.info("Formatting prompts...")
    train_dataset = map_to_prompts(raw_train)
    valid_dataset = map_to_prompts(raw_valid)
    
    # 3. Load Model and Inject LoRA
    logger.info(f"Loading Base Model: {BASE_MODEL}")
    model_wrapper = TriMixGeneratorModel(model_name_or_path=BASE_MODEL)
    
    logger.info("Injecting LoRA Adapters...")
    lora_manager = LoRAConfigurator(config_path=LORA_CONFIG_PATH)
    peft_model_wrapper = lora_manager.inject_adapter(model_wrapper)
    
    # 4. Configure Training Arguments
    # Note: eval_strategy replaces evaluation_strategy in recent transformers
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        seed=42,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        predict_with_generate=True, # Allows generating metrics if needed later
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
    trainer.save_model(best_lora_path)
    
    logger.info(f"Training for {EXPERIMENT_NAME} completed successfully!")

if __name__ == "__main__":
    main()
