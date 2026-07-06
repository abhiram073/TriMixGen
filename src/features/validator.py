import pandas as pd
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DatasetValidator:
    """
    Validates dataset schemas, checks for missing values, and enforces
    minimum length constraints on textual data.
    """
    def __init__(self, required_columns: List[str], text_column: str, min_words: int = 1):
        self.required_columns = required_columns
        self.text_column = text_column
        self.min_words = min_words

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Checks if all required columns exist in the DataFrame."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects and drops rows with missing values in the text column."""
        initial_len = len(df)
        df_clean = df.dropna(subset=[self.text_column]).copy()
        
        # Also drop rows where the text is completely empty or just whitespace
        df_clean = df_clean[df_clean[self.text_column].str.strip() != '']
        
        dropped = initial_len - len(df_clean)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing or empty text.")
        return df_clean

    def filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out rows where the word count is strictly less than min_words."""
        if self.min_words <= 0:
            return df
            
        initial_len = len(df)
        # Using simple whitespace split for validation; tokenizers handle the rest later
        mask = df[self.text_column].apply(lambda x: len(str(x).split()) >= self.min_words)
        df_filtered = df[mask].copy()
        
        dropped = initial_len - len(df_filtered)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows containing fewer than {self.min_words} words.")
        return df_filtered

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the complete validation pipeline on a DataFrame.
        Raises ValueError if schema validation fails.
        """
        logger.info("Starting dataset validation...")
        if not self.validate_schema(df):
            raise ValueError("Schema validation failed. Check required columns.")
            
        df = self.handle_missing_values(df)
        df = self.filter_by_length(df)
        
        logger.info(f"Dataset validation complete. Final row count: {len(df)}")
        return df
