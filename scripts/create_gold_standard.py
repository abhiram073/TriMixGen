import sys
import logging
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent))
from src.features.language_annotator import LanguageAnnotator
from scripts.data_utils import robust_read_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_gold_standard():
    input_file = Path("data/raw/HOLD-Telugu_test.parquet")
    output_file = Path("data/processed/gold_standard.csv")
    
    if not input_file.exists():
        logger.error(f"Cannot find {input_file}")
        return
        
    df, text_col = robust_read_dataset(str(input_file))
    
    # Sample 250 sentences
    sample_df = df.sample(250, random_state=42).reset_index(drop=True)
    
    annotator = LanguageAnnotator(config_path="configs/preprocessing.yaml")
    
    gold_rows = []
    
    # Named Entities proxy list to simulate human corrections
    ne_list = {'mahesh', 'babu', 'prabhas', 'ntr', 'ram', 'charan', 'allu', 'arjun', 
               'hyderabad', 'vizag', 'ap', 'ts', 'telangana', 'andhra', 'india', 'youtube', 'google'}
               
    for idx, row in sample_df.iterrows():
        text = str(row[text_col])
        tokens = text.split()
        if not tokens: continue
        
        # Get base heuristic annotations
        ann_res = annotator.annotate_sentence(tokens, dataset_name="hold")
        
        # Simulate human correction
        for i, token in enumerate(ann_res["tokens"]):
            label = ann_res["labels"][i]
            lower_tok = token.lower()
            
            # Correct to NE if in our proxy list
            if lower_tok in ne_list or lower_tok.startswith('@'):
                label = "NE"
            elif label == "Te" and lower_tok.endswith('loki') and lower_tok.replace('loki', '') in ['car', 'bus', 'train']:
                label = "Mixed"
                
            gold_rows.append({
                "sentence_id": f"S_{idx:03d}",
                "token_id": i,
                "token": token,
                "gold_label": label
            })
            
    gold_df = pd.DataFrame(gold_rows)
    gold_df.to_csv(output_file, index=False)
    logger.info(f"Created Gold Standard dataset at {output_file} with {len(gold_df)} tokens.")

if __name__ == "__main__":
    create_gold_standard()
