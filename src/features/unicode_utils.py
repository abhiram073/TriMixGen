import unicodedata
import pandas as pd
from typing import Dict, Any

class UnicodeUtils:
    """
    Utility class for Unicode validation, script detection, and encoding checks.
    """

    @staticmethod
    def is_telugu_char(char: str) -> bool:
        """Checks if a character falls within the Telugu Unicode block."""
        return '\u0c00' <= char <= '\u0c7f'
        
    @staticmethod
    def is_latin_char(char: str) -> bool:
        """Checks if a character is a standard Latin alphabet letter."""
        # A-Z (0041-005A), a-z (0061-007A)
        return ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a')

    @staticmethod
    def validate_encoding(text: str) -> bool:
        """
        Detects malformed Unicode sequences. 
        In Python, invalid bytes decoded with 'replace' result in U+FFFD.
        Returns False if the replacement character is found.
        """
        if not isinstance(text, str):
            return False
        return '\ufffd' not in text

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Detects the primary script of the text.
        Returns one of: 'Telugu', 'Latin', 'Mixed', 'Unknown'
        """
        if not text or not isinstance(text, str):
            return "Unknown"
        
        has_telugu = False
        has_latin = False
        
        for char in text:
            if UnicodeUtils.is_telugu_char(char):
                has_telugu = True
            elif UnicodeUtils.is_latin_char(char):
                has_latin = True
                
            if has_telugu and has_latin:
                return "Mixed"
                
        if has_telugu:
            return "Telugu"
        elif has_latin:
            return "Latin"
        else:
            return "Unknown"

    @staticmethod
    def analyze_proportions(text: str) -> Dict[str, float]:
        """
        Calculates the proportion of characters belonging to specific categories.
        Ignores whitespace.
        """
        counts = {"telugu": 0, "latin": 0, "digits": 0, "punctuation": 0, "other": 0}
        
        if not text or not isinstance(text, str):
            return {k: 0.0 for k in counts}
            
        for char in text:
            if UnicodeUtils.is_telugu_char(char):
                counts["telugu"] += 1
            elif UnicodeUtils.is_latin_char(char):
                counts["latin"] += 1
            else:
                cat = unicodedata.category(char)
                if cat.startswith('Z'):  # Skip whitespace/separators
                    continue
                elif cat.startswith('N'):  # Numbers
                    counts["digits"] += 1
                elif cat.startswith('P'):  # Punctuation
                    counts["punctuation"] += 1
                else:  # Emojis, symbols, etc.
                    counts["other"] += 1
                    
        total = sum(counts.values())
        if total == 0:
            return {k: 0.0 for k in counts}
            
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def generate_report(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """
        Generates a comprehensive Unicode analysis report for a dataset.
        """
        if df.empty or text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found or DataFrame is empty.")
            
        total_samples = len(df)
        texts = df[text_column].astype(str)
        
        # 1. Script Distribution
        scripts = texts.apply(UnicodeUtils.detect_script)
        script_counts = scripts.value_counts().to_dict()
        
        # 2. Malformed Encodings
        malformed_count = int((~texts.apply(UnicodeUtils.validate_encoding)).sum())
        
        return {
            "total_samples": total_samples,
            "malformed_samples": malformed_count,
            "script_distribution": {k: v / total_samples for k, v in script_counts.items()}
        }
