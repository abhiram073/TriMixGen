import pytest
from src.models.generation.prompt_builder import PromptBuilder

@pytest.fixture(scope="module")
def builder():
    return PromptBuilder(config_path="configs/prompts.yaml")

def test_missing_placeholders(builder):
    with pytest.raises(ValueError, match="Missing required kwargs"):
        builder.render("instruction", instruction="Translate this")
        # Missing 'input'

def test_empty_inputs(builder):
    # Empty inputs are technically valid as long as the kwarg is present,
    # unless dataset.py filters them. The builder itself just formats strings.
    rendered = builder.render("instruction", instruction="", input="")
    assert "Instruction: \nInput: \nResponse:" in rendered

def test_multilingual_inputs(builder):
    rendered = builder.render("en_to_cm", english_text="hello world")
    assert "hello world" in rendered
    
    rendered = builder.render("te_to_cm", telugu_text="నమస్కారం")
    assert "నమస్కారం" in rendered

def test_long_prompts(builder):
    long_text = "word " * 1000
    rendered = builder.render("conversation", context=long_text)
    assert len(rendered) > 5000

def test_special_characters(builder):
    special = "!@#$%^&*()\n\t {}"
    rendered = builder.render("zero_shot", instruction=special)
    assert special in rendered

def test_batch_prompt_generation(builder):
    batch = [
        {"english_text": "Hi"},
        {"english_text": "Bye"}
    ]
    rendered = builder.batch_render("en_to_cm", batch)
    assert len(rendered) == 2
    assert "Hi" in rendered[0]
    assert "Bye" in rendered[1]

def test_preview_report(builder):
    sample = {
        "en_to_cm": {"english_text": "I like this movie."},
        "instruction": {"instruction": "Write a review", "input": "Bahubali"}
    }
    report = builder.preview_templates(sample)
    assert "Prompt Preview Report" in report
    assert "I like this movie" in report
    assert "Bahubali" in report
