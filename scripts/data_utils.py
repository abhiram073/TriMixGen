import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def robust_read_dataset(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Reads a parquet file, implements header recovery if needed, 
    and dynamically resolves the text column.
    """
    df = pd.read_parquet(file_path)
    
    # 1. Header Recovery
    # If any column name is a very long string (e.g. >25 chars) and has spaces, 
    # it's highly likely pandas ate the first row.
    needs_recovery = False
    for col in df.columns:
        if isinstance(col, str) and len(col) > 25 and ' ' in col:
            needs_recovery = True
            break
            
    if needs_recovery:
        logger.info(f"Triggering header recovery for {file_path}")
        # The columns are the first row
        first_row = pd.DataFrame([df.columns], columns=[f"col_{i}" for i in range(len(df.columns))])
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        df = pd.concat([first_row, df], ignore_index=True)

    # 2. Schema Resolver
    aliases = ['comments', 'comment', 'sentence', 'text', 'tweet', 'output', 'input']
    text_col = None
    
    # Check aliases in priority order
    for alias in aliases:
        for col in df.columns:
            if str(col).lower() == alias:
                text_col = col
                break
        if text_col:
            break
            
    # Fallback to string heuristics if not found (e.g. column with longest avg string length)
    if not text_col:
        logger.info(f"No direct alias found in {df.columns}. Falling back to heuristics.")
        max_avg_len = 0
        for col in df.columns:
            # Check if column is mostly strings
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > max_avg_len:
                    max_avg_len = avg_len
                    text_col = col
                    
    if not text_col:
        raise ValueError(f"Could not resolve text column for {file_path}. Schema: {list(df.columns)}")
        
    return df, text_col
