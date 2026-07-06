import pytest
from backend.services.lid_service import LIDService

@pytest.fixture
def lid():
    service = LIDService()
    # Force mock mode for tests to avoid loading models
    service.is_mock_mode = True
    service.initialized = True
    # Ensure dictionaries are loaded
    service._load_configs()
    return service

def test_normalize_text(lid):
    assert lid.normalize_text("bagundi.") == "bagundi "
    assert lid.normalize_text("Hello, world!") == "Hello  world "

def test_dictionary_classification(lid):
    # Depending on what's in romanized_telugu.yaml
    label, reason, tier = lid.classify_token("bagundi")
    assert label == "TE"
    assert "Dictionary" in tier

def test_suffix_classification(lid):
    # 'vellanu' has '-anu' suffix
    label, reason, tier = lid.classify_token("vellanu")
    assert label == "TE"
    # Even if vellanu is in dictionary, let's test a fake word with the suffix
    label, reason, tier = lid.classify_token("randomwordthunnanu")
    assert label == "TE"
    assert "Suffix" in tier

def test_english_classification(lid):
    label, reason, tier = lid.classify_token("movie")
    assert label == "EN"
    assert "English" in tier

def test_other_classification(lid):
    label, reason, tier = lid.classify_token("xyzqwe")
    assert label == "OTHER"
    assert "OTHER" in tier

def test_tag_text_mixed(lid):
    result = lid.tag_text("Movie chala bagundi bro.")
    tokens = result["tokens"]
    labels = result["labels"]
    
    assert tokens == ["Movie", "chala", "bagundi", "bro."]
    assert labels == ["EN", "TE", "TE", "EN"]
