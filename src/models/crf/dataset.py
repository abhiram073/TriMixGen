import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List

from src.models.crf.features import extract_sentence_features

def load_pseudo_labeled_data(data_dirs: List[Path], max_samples: int = None) -> List[Tuple[List[str], List[str]]]:
    """Loads all pseudo-labeled data from parquet files and reconstructs sentences/labels."""
    sentences = []
    
    for directory in data_dirs:
        for file in directory.glob("*.parquet"):
            if "gold" in str(file): continue # Skip gold data
            
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].tolist()
                labels = row['labels'] if isinstance(row['labels'], list) else row['labels'].tolist()
                sentences.append((tokens, labels))
                
                if max_samples and len(sentences) >= max_samples:
                    return sentences
    return sentences

def load_gold_data(gold_path: Path) -> List[Tuple[List[str], List[str]]]:
    """Loads the Gold Standard evaluation dataset."""
    df = pd.read_parquet(gold_path)
    sentences = []
    for _, row in df.iterrows():
        tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].tolist()
        labels = row['labels'] if isinstance(row['labels'], list) else row['labels'].tolist()
        sentences.append((tokens, labels))
    return sentences

def prepare_crf_data(sentences: List[Tuple[List[str], List[str]]]):
    """Extracts X (features) and y (labels) from token-label pairs."""
    X = [extract_sentence_features(s[0]) for s in sentences]
    y = [s[1] for s in sentences]
    return X, y
