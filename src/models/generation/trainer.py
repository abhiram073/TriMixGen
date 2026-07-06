import yaml
import logging
from pathlib import Path
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.tokenizer import TriMixTokenizer

logger = logging.getLogger(__name__)

class GenerationTrainer:
    """
    Wraps the Hugging Face Seq2SeqTrainer for mT5-small training.
    Strictly uses Dependency Injection for model, dataset, and tokenizer.
    """
    def __init__(self, 
                 model_wrapper: TriMixGeneratorModel, 
                 tokenizer_wrapper: TriMixTokenizer, 
                 train_dataset, 
                 eval_dataset, 
                 callbacks=None,
                 config_path: str = "configs/training.yaml"):
        
        self.model_wrapper = model_wrapper
        self.tokenizer_wrapper = tokenizer_wrapper
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config_path = config_path
        
        self.callbacks = callbacks
        self.train_config = self._load_config()
        self.training_args = self._build_training_args()
        self.trainer = self._build_trainer()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f).get("training", {})
        except FileNotFoundError:
            logger.warning(f"Training config {self.config_path} not found. Using empty config.")
            return {}

    def _build_training_args(self) -> Seq2SeqTrainingArguments:
        output_dir = self.train_config.get("output_dir", "outputs/experiments/default/")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.train_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=self.train_config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=self.train_config.get("gradient_accumulation_steps", 16),
            learning_rate=float(self.train_config.get("learning_rate", 3e-4)),
            weight_decay=self.train_config.get("weight_decay", 0.01),
            num_train_epochs=self.train_config.get("num_train_epochs", 3),
            warmup_ratio=self.train_config.get("warmup_ratio", 0.1),
            logging_steps=self.train_config.get("logging_steps", 10),
            eval_strategy=self.train_config.get("eval_strategy", "epoch"),
            save_strategy=self.train_config.get("save_strategy", "epoch"),
            max_steps=self.train_config.get("max_steps", -1),
            eval_steps=self.train_config.get("eval_steps", None),
            save_steps=self.train_config.get("save_steps", None),
            load_best_model_at_end=self.train_config.get("load_best_model_at_end", True),
            metric_for_best_model=self.train_config.get("metric_for_best_model", "loss"),
            greater_is_better=self.train_config.get("greater_is_better", False),
            seed=self.train_config.get("seed", 42),
            max_grad_norm=self.train_config.get("max_grad_norm", 1.0),
            fp16=self.train_config.get("fp16", False),
            dataloader_num_workers=self.train_config.get("dataloader_num_workers", 0),
            predict_with_generate=True, # Required for Seq2Seq metric evaluation
            report_to="none", # Disable W&B/Tensorboard for now
            remove_unused_columns=self.train_config.get("remove_unused_columns", False)
        )

    def data_collator(self, features):
        """
        Dynamically tokenizes and batches the raw text data on the fly.
        """
        prompts = [f["prompt"] for f in features]
        targets = [f["target"] for f in features]
        
        # tokenizer.tokenize_for_training natively handles target padding masking (-100)
        return self.tokenizer_wrapper.tokenize_for_training(prompts, targets)

    def _build_trainer(self) -> Seq2SeqTrainer:
        return Seq2SeqTrainer(
            model=self.model_wrapper.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer_wrapper.tokenizer,
            data_collator=self.data_collator,
            callbacks=self.callbacks
        )
        
    def train(self, resume_from_checkpoint: bool = False):
        logger.info("Starting training loop...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.trainer.save_model(self.training_args.output_dir)
        
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        
        # Save a summary report
        self._generate_summary(metrics)
        logger.info("Training complete.")
        return train_result

    def _generate_summary(self, metrics: dict):
        output_dir = Path(self.training_args.output_dir)
        summary_path = output_dir / "training_summary.md"
        
        report = (
            f"# Training Summary\n\n"
            f"- **Train Loss:** {metrics.get('train_loss', 'N/A')}\n"
            f"- **Epochs:** {metrics.get('epoch', 'N/A')}\n"
            f"- **Train Runtime:** {metrics.get('train_runtime', 'N/A')} seconds\n"
            f"- **Tokens per Second:** {metrics.get('train_samples_per_second', 'N/A') * self.train_config.get('per_device_train_batch_size', 2)}\n"
        )
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report)
