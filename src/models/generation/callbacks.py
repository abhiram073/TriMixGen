import time
import json
import yaml
import psutil
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

class TriMixLoggingCallback(TrainerCallback):
    """
    Handles custom logging to structured files (JSON/YAML/MD) and TensorBoard.
    Monitors RAM usage, epoch timing, and core metrics.
    """
    def __init__(self, output_dir: str = "outputs/experiments/gen_001/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))
        self.epoch_start_time = 0
        self.callback_logs = []
        
    def _log_to_yaml(self, event: str, details: dict):
        self.callback_logs.append({"event": event, "timestamp": time.time(), "details": details})
        with open(self.output_dir / "callback_log.yaml", "w") as f:
            yaml.dump(self.callback_logs, f)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        ram_usage = psutil.virtual_memory().percent
        self._log_to_yaml("on_train_begin", {"initial_ram_percent": ram_usage})
        
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch_duration = time.time() - self.epoch_start_time
        ram_usage = psutil.virtual_memory().percent
        
        # Log to TensorBoard
        self.tb_writer.add_scalar("Performance/Epoch_Duration_sec", epoch_duration, state.epoch)
        self.tb_writer.add_scalar("Performance/RAM_Usage_Percent", ram_usage, state.epoch)
        
        # Log to timing report
        with open(self.output_dir / "timing_report.md", "a") as f:
            f.write(f"- **Epoch {state.epoch}:** {epoch_duration:.2f} seconds | RAM: {ram_usage}%\n")
            
        self._log_to_yaml("on_epoch_end", {"epoch": state.epoch, "duration": epoch_duration})

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
            
        step = state.global_step
        
        # Write to JSON
        metrics_file = self.output_dir / "metrics.json"
        existing_metrics = []
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                try:
                    existing_metrics = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        logs["step"] = step
        existing_metrics.append(logs)
        
        with open(metrics_file, "w") as f:
            json.dump(existing_metrics, f, indent=4)

        # Write core metrics to TensorBoard
        if "loss" in logs:
            self.tb_writer.add_scalar("Metrics/Train_Loss", logs["loss"], step)
        if "eval_loss" in logs:
            self.tb_writer.add_scalar("Metrics/Eval_Loss", logs["eval_loss"], step)
        if "learning_rate" in logs:
            self.tb_writer.add_scalar("Training/Learning_Rate", logs["learning_rate"], step)
        if "grad_norm" in logs:
            self.tb_writer.add_scalar("Training/Gradient_Norm", logs["grad_norm"], step)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._log_to_yaml("on_save", {"step": state.global_step, "epoch": state.epoch})

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tb_writer.close()
        self._log_to_yaml("on_train_end", {"final_step": state.global_step})


class CustomEarlyStoppingCallback(TrainerCallback):
    """
    Monitors eval_loss and halts training if it degrades for `patience` consecutive evaluations.
    """
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.degradation_count = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return

        current_loss = metrics["eval_loss"]
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.degradation_count = 0
        else:
            self.degradation_count += 1
            
        if self.degradation_count >= self.patience:
            control.should_training_stop = True
            
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_training_stop:
            pass # We could log exactly when it stopped
