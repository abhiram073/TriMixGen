import pytest
import torch
import shutil
from pathlib import Path
from src.models.generation.model import TriMixGeneratorModel

@pytest.fixture(scope="module")
def model_wrapper():
    # Use CPU explicitly for tests to avoid CUDA OOM in CI
    return TriMixGeneratorModel(model_name_or_path="google/mt5-small", device="cpu")

def test_model_initialization(model_wrapper):
    assert model_wrapper.model is not None
    assert model_wrapper.device == "cpu"
    assert model_wrapper.model.config.vocab_size == 250112

def test_diagnostics(model_wrapper):
    diag = model_wrapper.get_diagnostics()
    assert diag["encoder_layers"] == 8
    assert diag["decoder_layers"] == 8
    assert diag["total_parameters"] > 290_000_000
    assert diag["trainable_parameters"] == diag["total_parameters"] # No LoRA yet

def test_generation_config_loading(model_wrapper):
    model_wrapper.load_generation_config("configs/generation.yaml")
    assert model_wrapper.model.generation_config is not None

def test_checkpoint_saving_and_loading(model_wrapper, tmp_path):
    save_dir = tmp_path / "test_checkpoint"
    model_wrapper.save_checkpoint(str(save_dir))
    
    assert (save_dir / "config.json").exists()
    assert (save_dir / "model.safetensors").exists()
    
    # Test loading from local checkpoint
    local_model = TriMixGeneratorModel(model_name_or_path=str(save_dir), device="cpu")
    assert local_model.model is not None
    assert local_model.model.config.vocab_size == 250112

def test_deterministic_inference(model_wrapper):
    # Ensure inference is deterministic with a fixed seed
    # Mocking a basic input tensor
    input_ids = torch.tensor([[1, 2, 3, 4, 1]])
    
    torch.manual_seed(42)
    output1 = model_wrapper.model.generate(input_ids, max_new_tokens=5, do_sample=True)
    
    torch.manual_seed(42)
    output2 = model_wrapper.model.generate(input_ids, max_new_tokens=5, do_sample=True)
    
    assert torch.equal(output1, output2)
    
def test_report_generation(model_wrapper, tmp_path):
    report_path = tmp_path / "model_summary.md"
    model_wrapper.generate_diagnostics_report(str(report_path))
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "google/mt5-small" in content
    assert "Total Parameters" in content
