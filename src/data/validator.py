import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatasetValidator:
    """
    Validates datasets for schema correctness, missing values, duplicates, and encoding.
    """
    
    @staticmethod
    def validate_generation_schema(df: pd.DataFrame) -> Dict[str, Any]:
        required_cols = {"id", "language_pair", "instruction", "target"}
        return DatasetValidator._validate_schema(df, required_cols)
        
    @staticmethod
    def validate_lid_schema(df: pd.DataFrame) -> Dict[str, Any]:
        required_cols = {"id", "language_pair", "tokens", "labels"}
        report = DatasetValidator._validate_schema(df, required_cols)
        
        # specific LID validation (tokens and labels length must match)
        if report["is_valid"]:
            try:
                mismatch_count = df.apply(lambda row: len(row['tokens']) != len(row['labels']), axis=1).sum()
                if mismatch_count > 0:
                    report["is_valid"] = False
                    report["errors"].append(f"Found {mismatch_count} rows where tokens length != labels length.")
            except Exception as e:
                report["is_valid"] = False
                report["errors"].append(f"Failed token/label length validation: {e}")
                
        return report

    @staticmethod
    def _validate_schema(df: pd.DataFrame, required_cols: set) -> Dict[str, Any]:
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "missing_values": {},
            "duplicate_ids": 0
        }
        
        # 1. Check Columns
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            report["is_valid"] = False
            report["errors"].append(f"Missing required columns: {missing_cols}")
            return report # Early exit if missing critical columns
            
        # 2. Check Missing Values
        missing_counts = df[list(required_cols)].isnull().sum().to_dict()
        report["missing_values"] = missing_counts
        
        if sum(missing_counts.values()) > 0:
            report["warnings"].append("Dataset contains missing values in required columns.")
            
        # 3. Check Duplicates
        if "id" in df.columns:
            dup_count = df.duplicated(subset=["id"]).sum()
            report["duplicate_ids"] = int(dup_count)
            if dup_count > 0:
                report["warnings"].append(f"Found {dup_count} duplicate IDs.")
                
        return report
