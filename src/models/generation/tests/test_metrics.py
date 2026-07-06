import pytest
from src.models.generation.metrics import GenerationMetrics

@pytest.fixture
def metrics_lib():
    return GenerationMetrics(output_dir="outputs/experiments/test_metrics/")

# --- N-Gram Exact Matching Tests ---

def test_perfect_prediction(metrics_lib):
    preds = ["the movie is very good"]
    refs = ["the movie is very good"]
    bleu = metrics_lib.compute_bleu(preds, refs)
    assert bleu > 0.99

def test_partially_correct_prediction(metrics_lib):
    preds = ["the movie is bad"]
    refs = ["the movie is very good"]
    bleu = metrics_lib.compute_bleu(preds, refs)
    assert 0.0 < bleu < 1.0

def test_completely_incorrect_prediction(metrics_lib):
    preds = ["cinema chala super undi"]
    refs = ["the movie is very good"]
    bleu = metrics_lib.compute_bleu(preds, refs)
    # No overlapping words, smoothing might give it a tiny non-zero, but should be near 0
    assert bleu < 0.1

# --- Diversity Tests ---

def test_repeated_outputs(metrics_lib):
    preds = ["ok ok ok ok", "ok ok ok ok"]
    d1, d2 = metrics_lib.compute_distinct_n(preds)
    # Total unigrams = 8. Unique = 1.
    assert d1 == 1/8
    
    self_bleu = metrics_lib.compute_self_bleu(preds)
    assert self_bleu > 0.99 # Identical sentences

def test_high_diversity_outputs(metrics_lib):
    preds = ["the movie is good", "cinema super undi", "nenu vellanu"]
    d1, d2 = metrics_lib.compute_distinct_n(preds)
    # High distinctness
    assert d1 == 1.0 # Every word is completely unique across the dataset
    
    self_bleu = metrics_lib.compute_self_bleu(preds)
    assert self_bleu < 0.1 # Very low syntactic overlap

# --- CMI Tests ---

def test_monolingual_cmi(metrics_lib):
    # Monolingual English: Max lang = 4, Total = 4
    labels = [["EN", "EN", "EN", "EN"]]
    results = metrics_lib.compute_cmi(labels)
    assert results["avg_cmi"] == 0.0
    assert results["dataset_cmi"] == 0.0

def test_balanced_code_mixed_cmi(metrics_lib):
    # Balanced code-mixed: 2 EN, 2 TE
    labels = [["EN", "TE", "EN", "TE"]]
    results = metrics_lib.compute_cmi(labels)
    # Max = 2, Total = 4. CMI = 100 * (1 - 2/4) = 50.0
    assert results["avg_cmi"] == 50.0
    assert results["dataset_cmi"] == 50.0

def test_cmi_with_punctuation(metrics_lib):
    # 2 EN, 2 TE, 2 PUNCT
    labels = [["EN", "PUNCT", "TE", "EN", "PUNCT", "TE"]]
    results = metrics_lib.compute_cmi(labels)
    # Punct ignored. Lang total = 4, Max = 2. CMI = 50.0
    assert results["avg_cmi"] == 50.0

def test_full_evaluation_pipeline_handling_missing_inputs(metrics_lib):
    # Test evaluation when labels are missing
    preds = ["the movie is good"]
    refs = ["the movie is good"]
    results = metrics_lib.evaluate(predictions=preds, references=refs, token_labels=None)
    
    assert "bleu" in results
    assert "distinct_1" in results
    assert "avg_cmi" not in results # Should be skipped gracefully
