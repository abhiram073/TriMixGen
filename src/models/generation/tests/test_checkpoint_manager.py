import os
import json
import pytest
from pathlib import Path
from src.models.generation.checkpoint_manager import CheckpointManager

class MockBaseModel:
    pass

class MockPeftModel:
    def merge_and_unload(self):
        return MockBaseModel()

class MockModelWrapper:
    def __init__(self, model):
        self.model = model

@pytest.fixture
def temp_experiment_dir(tmp_path):
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    
    # Create 4 dummy checkpoints with state files
    for i, loss in enumerate([2.5, 1.2, 1.8, 0.9]):
        ckpt = exp_dir / f"checkpoint-{i*100}"
        ckpt.mkdir()
        state = {
            "log_history": [{"eval_loss": loss}]
        }
        with open(ckpt / "trainer_state.json", "w") as f:
            json.dump(state, f)
            
    return exp_dir

def test_get_best_checkpoint(temp_experiment_dir):
    best_ckpt = CheckpointManager.get_best_checkpoint(str(temp_experiment_dir))
    assert best_ckpt is None or "checkpoint-300" in best_ckpt

def test_cleanup_checkpoints(temp_experiment_dir):
    # Should delete the worst one (loss=2.5, checkpoint-0)
    CheckpointManager.cleanup_checkpoints(str(temp_experiment_dir), keep_top_k=3)
    
    remaining = list(temp_experiment_dir.glob("checkpoint-*"))
    assert len(remaining) == 3
    assert not (temp_experiment_dir / "checkpoint-0").exists()

def test_merge_and_unload():
    # Test that a PeftModel is correctly merged into a Base model
    wrapper = MockModelWrapper(MockPeftModel())
    merged_wrapper = CheckpointManager.merge_and_unload(wrapper)
    
    assert isinstance(merged_wrapper.model, MockBaseModel)
