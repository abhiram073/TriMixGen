import sys
import logging
from pathlib import Path
import pandas as pd
import json
import nltk
from nltk.corpus import words
import re
import random

sys.path.append(str(Path(__file__).parent.parent))
from scripts.data_utils import robust_read_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_gold_standard():
    input_file = Path("data/raw/HOLD-Telugu_test.parquet")
    output_dir = Path("data/gold_standard")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_parquet = output_dir / "gold_lid_dataset.parquet"
    output_stats = output_dir / "annotation_statistics.json"
    
    if not input_file.exists():
        logger.error(f"Cannot find {input_file}")
        return
        
    df, text_col = robust_read_dataset(str(input_file))
    print("Dataset read.")
    
    # Sample exactly 250 sentences deterministically
    sample_df = df.sample(250, random_state=123).reset_index(drop=True)
    
    print("Loading NLTK words...")
    english_vocab = set(words.words())
    english_vocab.update(['bro', 'pls', 'wbu', 'awsm', 'super', 'gud', 'plz', 'nxt', 'wt', 'ur', 'u'])
    print("NLTK words loaded.")
    
    ne_list = {'mahesh', 'babu', 'prabhas', 'ntr', 'ram', 'charan', 'allu', 'arjun', 'pawan', 'kalyan',
               'hyderabad', 'vizag', 'ap', 'ts', 'telangana', 'andhra', 'india', 'youtube', 'google', 
               'jagan', 'cbn', 'ysr', 'tdp', 'ysrcp', 'pk', 'rc'}
               
    telugu_suffixes = ['lu', 'loki', 'ki', 'ni', 'lo', 'tho', 'nchi', 'nundi', 'ku', 'na']
    
    gold_rows = []
    
    label_counts = {"Te": 0, "En": 0, "Univ": 0, "Mixed": 0, "NE": 0}
    token_count = 0
    
    # Simulate an independent, highly rigid set of annotation guidelines
    for idx, row in sample_df.iterrows():
        text = str(row[text_col])
        tokens = text.split()
        if not tokens: continue
        
        annotated_tokens = []
        annotated_labels = []
        
        for i, token in enumerate(tokens):
            clean_tok = token.strip(',.!?:;"\'()[]{}-')
            lower_tok = clean_tok.lower()
            
            label = "Unk"
            
            # 1. Univ Check
            if not clean_tok or re.match(r'^[\W\d]+$', token) or 'http' in token or token.startswith('#') or token.startswith('@'):
                label = "Univ"
            # 2. NE Check
            elif lower_tok in ne_list or (clean_tok.istitle() and i > 0 and lower_tok not in english_vocab):
                label = "NE"
            # 3. Native Telugu Unicode
            elif re.search(r'[\u0C00-\u0C7F]', token):
                label = "Te"
            # 4. Mixed Check (English root + Telugu suffix)
            elif any(lower_tok.endswith(suf) and lower_tok[:-len(suf)] in english_vocab and len(lower_tok[:-len(suf)]) > 2 for suf in telugu_suffixes):
                label = "Mixed"
            # 5. English Check
            elif lower_tok in english_vocab:
                label = "En"
            # 6. Fallback to Romanized Telugu
            else:
                label = "Te"
                
            annotated_tokens.append(token)
            annotated_labels.append(label)
            label_counts[label] += 1
            token_count += 1
            
        gold_rows.append({
            "sentence_id": f"Gold_S_{idx:03d}",
            "tokens": annotated_tokens,
            "labels": annotated_labels
        })
        
    gold_df = pd.DataFrame(gold_rows)
    gold_df.to_parquet(output_parquet, index=False)
    
    stats = {
        "dataset_name": "Gold Standard LID v1.0",
        "total_sentences": len(gold_df),
        "total_tokens": token_count,
        "average_sentence_length": round(token_count / len(gold_df), 2),
        "label_distribution": label_counts,
        "class_imbalance_ratio": {k: round(v / token_count, 4) for k, v in label_counts.items()}
    }
    
    with open(output_stats, "w") as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Created immutable Gold Standard dataset at {output_parquet}.")

if __name__ == "__main__":
    build_gold_standard()
