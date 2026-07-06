import sys
import numpy as np
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))
from src.models.crf.dataset import load_pseudo_labeled_data

def analyze_lengths():
    print("Loading datasets...")
    data_dirs = [Path("data/processed")]
    # Load up to 25000 to match the BiLSTM training size exactly
    all_sentences_labels = load_pseudo_labeled_data(data_dirs, max_samples=25000)
    
    lengths = [len(s[0]) for s in all_sentences_labels]
    
    if not lengths:
        print("No sentences found.")
        return
        
    lengths = np.array(lengths)
    
    stats = {
        "min": int(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(np.max(lengths))
    }
    
    print(json.dumps(stats, indent=4))
    
    # Justify max_length
    # Usually p99 is a very safe max_length.
    chosen_max = stats["p99"]
    truncated_seqs = np.sum(lengths > chosen_max)
    percent_truncated = (truncated_seqs / len(lengths)) * 100
    
    print(f"\nRecommended max_length: {chosen_max}")
    print(f"Percentage of sequences truncated: {percent_truncated:.2f}%")
    
if __name__ == "__main__":
    analyze_lengths()
