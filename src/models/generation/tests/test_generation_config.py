import pytest
from src.models.generation.generation_config import GenerationConfigManager

@pytest.fixture(scope="module")
def config_mgr():
    return GenerationConfigManager(config_path="configs/generation.yaml")

def test_load_greedy_strategy(config_mgr):
    conf = config_mgr.get_strategy_config("greedy")
    assert conf.do_sample is False
    assert conf.num_beams == 1
    assert conf.max_new_tokens == 128

def test_load_beam_search_strategy(config_mgr):
    conf = config_mgr.get_strategy_config("beam_search")
    assert conf.do_sample is False
    assert conf.num_beams == 4
    assert conf.early_stopping is True
    assert conf.no_repeat_ngram_size == 3

def test_load_nucleus_sampling(config_mgr):
    conf = config_mgr.get_strategy_config("nucleus_sampling")
    assert conf.do_sample is True
    assert conf.top_p == 0.92
    assert conf.temperature == 0.8
    assert conf.repetition_penalty == 1.2

def test_invalid_strategy(config_mgr):
    with pytest.raises(ValueError, match="not found in configuration"):
        config_mgr.get_strategy_config("magical_decoding")
