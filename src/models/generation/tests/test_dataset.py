import pytest
import pandas as pd
from pathlib import Path
from src.models.generation.dataset import TriMixGenerationDataset

@pytest.fixture
def dummy_data_dir(tmp_path):
    # Create dummy parquet files for each dataset type
    alpaca_df = pd.DataFrame({
        "instruction": ["Translate to Telugu", "Explain physics"],
        "input": ["Hello world", ""],
        "output": ["Namaskaram prapancham", "Physics ante..."]
    })
    alpaca_path = tmp_path / "alpaca.parquet"
    alpaca_df.to_parquet(alpaca_path)
    
    hold_df = pd.DataFrame({
        "context": ["Hi how are you?", "Movie chusava?"],
        "response": ["I am fine bro", "Ha chusa, super undi"]
    })
    hold_path = tmp_path / "hold.parquet"
    hold_df.to_parquet(hold_path)
    
    sentiment_df = pd.DataFrame({
        "label": ["positive", "negative"],
        "text": ["cinema keka", "worst movie ever"]
    })
    sentiment_path = tmp_path / "sentiment.parquet"
    sentiment_df.to_parquet(sentiment_path)
    
    return {
        "alpaca": alpaca_path,
        "hold": hold_path,
        "sentiment": sentiment_path
    }

def test_alpaca_dataset_loading(dummy_data_dir):
    dataset = TriMixGenerationDataset(dummy_data_dir["alpaca"], dataset_type="alpaca")
    assert len(dataset) == 2
    assert dataset[0]["instruction"] == "Translate to Telugu"
    assert dataset[0]["input"] == "Hello world"
    assert dataset[0]["target"] == "Namaskaram prapancham"
    assert dataset[0]["dataset_type"] == "alpaca"
    
def test_hold_dataset_loading(dummy_data_dir):
    dataset = TriMixGenerationDataset(dummy_data_dir["hold"], dataset_type="hold")
    assert len(dataset) == 2
    assert dataset[1]["context"] == "Movie chusava?"
    assert dataset[1]["target"] == "Ha chusa, super undi"
    assert dataset[1]["dataset_type"] == "hold"
    
def test_sentiment_dataset_loading(dummy_data_dir):
    dataset = TriMixGenerationDataset(dummy_data_dir["sentiment"], dataset_type="sentiment")
    assert len(dataset) == 2
    assert dataset[0]["label"] == "positive"
    assert dataset[0]["target"] == "cinema keka"
    assert dataset[0]["dataset_type"] == "sentiment"

def test_empty_target_filtering(tmp_path):
    # Create a df with some empty targets
    df = pd.DataFrame({
        "instruction": ["Valid", "Empty target", "NaN target", "Spaces target"],
        "input": ["", "", "", ""],
        "output": ["Valid output", "", float("nan"), "   "]
    })
    path = tmp_path / "dirty_alpaca.csv"
    df.to_csv(path, index=False)
    
    dataset = TriMixGenerationDataset(path, dataset_type="alpaca")
    assert len(dataset) == 1
    assert dataset[0]["instruction"] == "Valid"
    assert dataset[0]["target"] == "Valid output"
