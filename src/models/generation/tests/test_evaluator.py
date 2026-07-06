import pytest
from src.models.generation.evaluator import GenerationEvaluator

class MockLID:
    def predict_tags(self, text):
        return ["EN", "TE", "EN"]

@pytest.fixture
def evaluator():
    return GenerationEvaluator(lid_model=MockLID(), output_dir="outputs/experiments/test_eval/")

def test_evaluator_missing_references(evaluator):
    prompts = ["translate this"]
    results = evaluator.evaluate_batch(prompts, references=None)
    
    assert "avg_cmi" in results
    assert "bleu" not in results

def test_evaluator_missing_lid():
    evaluator_no_lid = GenerationEvaluator(lid_model=None, output_dir="outputs/experiments/test_eval/")
    prompts = ["translate this"]
    results = evaluator_no_lid.evaluate_batch(prompts, references=["translate this"])
    
    assert "bleu" in results
    assert "avg_cmi" not in results

def test_evaluator_empty_batch(evaluator):
    results = evaluator.evaluate_batch([], [])
    assert results == {}

def test_evaluator_error_analysis(evaluator):
    # Test monolingual collapse detection
    prompts = ["test 1", "test 2"]
    # We pass prompts as predictions via the fallback mock
    evaluator.evaluate_batch(prompts)
    
    # Check that error analysis file was created
    assert (evaluator.output_dir.parent / "error_analysis_generation.md").exists()
    assert (evaluator.output_dir / "predictions.csv").exists()
    assert (evaluator.output_dir / "detailed_examples.md").exists()
