import pytest
import json
import yaml
from pathlib import Path
from transformers import TrainerState, TrainerControl, TrainingArguments
from src.models.generation.callbacks import TriMixLoggingCallback, CustomEarlyStoppingCallback

@pytest.fixture
def mock_args():
    return TrainingArguments(output_dir="outputs/experiments/test_callbacks")

@pytest.fixture
def mock_state():
    state = TrainerState()
    state.global_step = 10
    state.epoch = 1.0
    return state

@pytest.fixture
def mock_control():
    return TrainerControl()

def test_logging_callback_files_created(tmp_path, mock_args, mock_state, mock_control):
    out_dir = tmp_path / "gen_test"
    cb = TriMixLoggingCallback(output_dir=str(out_dir))
    
    # Simulate events
    cb.on_train_begin(mock_args, mock_state, mock_control)
    cb.on_epoch_begin(mock_args, mock_state, mock_control)
    cb.on_epoch_end(mock_args, mock_state, mock_control)
    cb.on_log(mock_args, mock_state, mock_control, logs={"loss": 0.5, "learning_rate": 0.01})
    cb.on_train_end(mock_args, mock_state, mock_control)
    
    assert (out_dir / "callback_log.yaml").exists()
    assert (out_dir / "timing_report.md").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "runs").is_dir() # TensorBoard folder

def test_early_stopping_behavior(mock_args, mock_state, mock_control):
    cb = CustomEarlyStoppingCallback(patience=2)
    
    # 1. First eval, establishes baseline
    cb.on_evaluate(mock_args, mock_state, mock_control, metrics={"eval_loss": 2.0})
    assert control_should_stop(mock_control) is False
    assert cb.best_loss == 2.0
    
    # 2. Improves
    cb.on_evaluate(mock_args, mock_state, mock_control, metrics={"eval_loss": 1.5})
    assert control_should_stop(mock_control) is False
    assert cb.best_loss == 1.5
    
    # 3. Degrades (count = 1)
    cb.on_evaluate(mock_args, mock_state, mock_control, metrics={"eval_loss": 1.6})
    assert control_should_stop(mock_control) is False
    
    # 4. Degrades again (count = 2 -> Hits patience threshold)
    cb.on_evaluate(mock_args, mock_state, mock_control, metrics={"eval_loss": 1.7})
    assert control_should_stop(mock_control) is True

def control_should_stop(control):
    return getattr(control, "should_training_stop", False)
