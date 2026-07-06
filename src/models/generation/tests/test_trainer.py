import pytest
from pathlib import Path
from src.models.generation.model import TriMixGeneratorModel
from src.models.generation.tokenizer import TriMixTokenizer
from src.models.generation.trainer import GenerationTrainer

class DummyDataset:
    def __init__(self, size=10):
        self.data = [{"prompt": f"prompt {i}", "target": f"target {i}"} for i in range(size)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

@pytest.fixture(scope="module")
def dependencies():
    model = TriMixGeneratorModel(model_name_or_path="google/mt5-small", device="cpu")
    tokenizer = TriMixTokenizer(config_path="configs/generation.yaml")
    train_ds = DummyDataset(10)
    eval_ds = DummyDataset(2)
    return model, tokenizer, train_ds, eval_ds

def test_trainer_initialization_and_injection(dependencies):
    model, tokenizer, train_ds, eval_ds = dependencies
    trainer = GenerationTrainer(
        model_wrapper=model,
        tokenizer_wrapper=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        config_path="configs/training.yaml"
    )
    
    # Assert dependency injection
    assert trainer.model_wrapper is model
    assert trainer.tokenizer_wrapper is tokenizer
    assert trainer.train_dataset is train_ds
    assert trainer.eval_dataset is eval_ds
    assert trainer.trainer.model is model.model

def test_deterministic_training_config(dependencies):
    model, tokenizer, train_ds, eval_ds = dependencies
    trainer = GenerationTrainer(
        model_wrapper=model,
        tokenizer_wrapper=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        config_path="configs/training.yaml"
    )
    
    args = trainer.training_args
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 16
    assert args.seed == 42
    assert args.learning_rate == 3e-4
    assert args.predict_with_generate is True

def test_data_collator(dependencies):
    model, tokenizer, train_ds, eval_ds = dependencies
    trainer = GenerationTrainer(
        model_wrapper=model,
        tokenizer_wrapper=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        config_path="configs/training.yaml"
    )
    
    batch = [train_ds[0], train_ds[1]]
    collated = trainer.data_collator(batch)
    
    assert "input_ids" in collated
    assert "labels" in collated
    assert collated["input_ids"].shape[0] == 2
