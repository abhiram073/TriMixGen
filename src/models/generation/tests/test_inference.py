import pytest
from src.models.generation.inference import Generator

@pytest.fixture
def generator():
    # Load base model without LoRA for fast testing
    return Generator(base_model_name="google/mt5-small", lora_adapter_path=None, merge_weights=False)

def test_single_generation(generator):
    output = generator.generate(
        instruction="Translate to Telugu:",
        context="Hello world",
        template="english_to_codemixed"
    )
    
    assert "generated_text" in output
    assert "raw_prompt" in output
    assert "generation_time_sec" in output
    assert "token_count" in output
    assert "decoding_strategy" in output
    assert output["decoding_strategy"] == "nucleus_sampling"
    
    # Assert string types
    assert isinstance(output["generated_text"], str)

def test_batch_generation(generator):
    instructions = ["Translate:", "Translate:"]
    contexts = ["One", "Two"]
    
    outputs = generator.generate_batch(
        instructions=instructions,
        contexts=contexts,
        template="english_to_codemixed"
    )
    
    assert len(outputs) == 2
    assert "generated_text" in outputs[0]
    assert "generated_text" in outputs[1]

def test_empty_batch(generator):
    outputs = generator.generate_batch([])
    assert outputs == []

def test_different_decoding_strategies(generator):
    output_greedy = generator.generate(
        instruction="Test",
        context="One",
        decoding_preset="greedy"
    )
    assert output_greedy["decoding_strategy"] == "greedy"
    
    output_beam = generator.generate(
        instruction="Test",
        context="One",
        decoding_preset="beam_search"
    )
    assert output_beam["decoding_strategy"] == "beam_search"
