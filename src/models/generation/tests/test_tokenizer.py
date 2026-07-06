import pytest
import torch
from pathlib import Path
from src.models.generation.tokenizer import TriMixTokenizer

@pytest.fixture(scope="module")
def tokenizer():
    # Uses the default configs/generation.yaml created earlier
    return TriMixTokenizer(config_path="configs/generation.yaml")

def test_tokenize_for_training_basic(tokenizer):
    source = ["Translate to Telugu: hello bro"]
    target = ["namaskaram bro"]
    
    encoded = tokenizer.tokenize_for_training(source, target)
    
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert "labels" in encoded
    
    # Check shape
    assert encoded["input_ids"].shape == (1, tokenizer.tok_config['max_source_length'])
    assert encoded["labels"].shape == (1, tokenizer.tok_config['max_target_length'])
    
    # Check padding substitution in labels
    # padding should be -100
    assert -100 in encoded["labels"][0]
    
def test_tokenize_empty_string(tokenizer):
    source = [""]
    target = [""]
    encoded = tokenizer.tokenize_for_training(source, target)
    # T5 encodes empty string to just the EOS token `</s>`
    assert encoded["input_ids"][0][0] == tokenizer.tokenizer.eos_token_id

def test_long_sequence_truncation(tokenizer):
    source = ["long word " * 500]
    encoded = tokenizer.tokenize_for_inference(source)
    # Should truncate at max_source_length
    assert encoded["input_ids"].shape[1] == tokenizer.tok_config['max_source_length']
    # Last token should be EOS if properly truncated by tokenizer, 
    # but HF truncation sometimes just cuts it. Let's just check length.
    
def test_unicode_telugu(tokenizer):
    source = ["నమస్కారం"]
    encoded = tokenizer.tokenize_for_inference(source)
    # Check that it's successfully tokenized and not mapped to UNK
    assert tokenizer.tokenizer.unk_token_id not in encoded["input_ids"][0][:2]
    
def test_romanized_mixed(tokenizer):
    source = ["cinema bagundi bro, totally recommend"]
    encoded = tokenizer.tokenize_for_inference(source)
    # Check that it tokenizes
    assert len(encoded["input_ids"][0]) > 2
    
def test_batch_tokenization(tokenizer):
    source = ["First sentence", "Second longer sentence here"]
    target = ["out 1", "out 2"]
    encoded = tokenizer.tokenize_for_training(source, target)
    assert encoded["input_ids"].shape[0] == 2
    assert encoded["labels"].shape[0] == 2

def test_benchmark_tokenizer(tokenizer):
    """
    Benchmarks the tokenizer to report metrics to the user.
    """
    corpus = [
        "movie chala bagundi",
        "worst cinema ra idhi",
        "nenu repu velthanu",
        "totally agree bro",
        "hi how are you"
    ]
    
    total_tokens = 0
    unks = 0
    for text in corpus:
        ids = tokenizer.tokenizer.encode(text)
        total_tokens += len(ids)
        unks += ids.count(tokenizer.tokenizer.unk_token_id)
        
    avg_len = total_tokens / len(corpus)
    vocab_coverage = ((total_tokens - unks) / total_tokens) * 100
    
    print(f"\n--- Tokenizer Benchmark ---")
    print(f"Average tokens per sentence: {avg_len}")
    print(f"Vocabulary Coverage: {vocab_coverage}%")
    print(f"---------------------------\n")
    
    assert vocab_coverage == 100.0 # SentencePiece should perfectly cover this without UNKs
