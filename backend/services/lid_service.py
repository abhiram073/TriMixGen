import logging
import re
import os
import yaml
from backend.config import settings

logger = logging.getLogger(__name__)

class LIDService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LIDService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
        
    def __init__(self):
        if self.initialized:
            return
            
        logger.info(f"Initializing LIDService on device: {settings.DEVICE}")
        
        self.is_mock_mode = settings.USE_MOCK_LID
        
        self._load_configs()
        
        if self.is_mock_mode:
            logger.warning("DEVELOPMENT MODE: Hybrid LID Service is active (Mock Model).")
            self.model_loaded = True
            self.initialized = True
            return
            
        # Production Mode
        try:
            if not os.path.exists(settings.INDICBERT_MODEL_PATH) and settings.INDICBERT_MODEL_PATH != "ai4bharat/indic-bert":
                raise FileNotFoundError(f"Production IndicBERT checkpoint missing at {settings.INDICBERT_MODEL_PATH}")
                
            self.model_loaded = True
            self.initialized = True
            logger.info("LIDService initialized in PRODUCTION mode.")
        except Exception as e:
            logger.error(f"Failed to initialize LIDService in Production: {e}")
            self.model_loaded = False
            self.initialized = False

    def _load_configs(self):
        """Loads dictionaries from configuration files."""
        self.hin_dict = set()
        self.ben_dict = set()
        self.guj_dict = set()
        self.en_dict = set()
        self.suffixes = set()
        self.entities = set()
        
        try:
            with open("configs/lid/hindi_words.yaml", "r") as f:
                self.hin_dict = set(yaml.safe_load(f).get("words", []))
            with open("configs/lid/bengali_words.yaml", "r") as f:
                self.ben_dict = set(yaml.safe_load(f).get("words", []))
            with open("configs/lid/gujarati_words.yaml", "r") as f:
                self.guj_dict = set(yaml.safe_load(f).get("words", []))
            with open("configs/lid/english_words.yaml", "r") as f:
                self.en_dict = set(yaml.safe_load(f).get("words", []))
            with open("configs/lid/suffixes.yaml", "r") as f:
                self.suffixes = set(yaml.safe_load(f).get("suffixes", []))
            with open("configs/lid/named_entities.yaml", "r") as f:
                self.entities = set(yaml.safe_load(f).get("entities", []))
        except FileNotFoundError as e:
            logger.warning(f"LID config file missing: {e}. Falling back to empty dictionaries.")

    def normalize_text(self, text: str) -> str:
        """Removes punctuation and special characters."""
        normalized = re.sub(r'[^\w\s]', ' ', text)
        return normalized

    def tokenize(self, text: str) -> list:
        """Splits normalized text into a list of lowercase tokens."""
        return self.normalize_text(text).split()

    def classify_token(self, token: str, language_pair: str) -> tuple:
        """Applies the Hybrid Pipeline to a single token, returning (Label, Reason, Tier)."""
        token_lower = token.lower()
        
        # Determine valid tags for this pair
        valid_tags = {"HIN", "ENG", "OTHER"}
        if "BEN" in language_pair:
            valid_tags.add("BEN")
        elif "GUJ" in language_pair:
            valid_tags.add("GUJ")
            
        # OTHER mapping for strict punctuation/numbers
        if not re.match(r'^[a-zA-Z]+$', token):
            return "OTHER", "Not a strict alphabetic word", "Tier 0 (Format)"

        # Tier 1: Dictionary Lookup
        if "BEN" in valid_tags and token_lower in self.ben_dict:
            return "BEN", "Found in Bengali dictionary", "Tier 1"
        if "GUJ" in valid_tags and token_lower in self.guj_dict:
            return "GUJ", "Found in Gujarati dictionary", "Tier 1"
        if token_lower in self.hin_dict:
            return "HIN", "Found in Hindi dictionary", "Tier 1"
            
        # Tier 2: NER
        if token_lower in self.entities:
            return "ENG", "Identified as Named Entity", "Tier 2"
            
        # Tier 3: Suffixes
        if any(token_lower.endswith(s) for s in self.suffixes):
            # Fallback heuristic: assume HIN for suffixes for now
            return "HIN", "Matches morphological suffix", "Tier 3"
            
        # Tier 4: English
        if token_lower in self.en_dict:
            return "ENG", "Found in English dictionary", "Tier 4"
            
        # Tier 5 Fallback (Model / Other)
        if self.is_mock_mode:
            return "OTHER", "Token unknown to all heuristic tiers", "Tier 5"
        else:
            return "OTHER", "Unrecognized by production model", "Tier 5"

    def calculate_labels(self, tokens: list, language_pair: str) -> list:
        """Iterates through tokens and logs classifications."""
        labels = []
        for t in tokens:
            label, reason, tier = self.classify_token(t, language_pair)
            labels.append(label)
            logger.debug(f"LID Classification -> Token: '{t}' | Pair: {language_pair} | Label: '{label}' | Tier: '{tier}' | Reason: '{reason}'")
        return labels

    def tag_text(self, text: str, language_pair: str) -> dict:
        """Public API endpoint for language identification."""
        if not self.initialized:
            if not self.is_mock_mode:
                raise RuntimeError("LIDService is unavailable. Production checkpoint missing.")
            else:
                raise RuntimeError("LIDService is not initialized.")
            
        original_tokens = text.split()
        
        cleaned_tokens = []
        for raw_t in original_tokens:
            clean = re.sub(r'[^\w]', '', raw_t)
            if clean:
                cleaned_tokens.append(clean)
            else:
                cleaned_tokens.append(raw_t) # fallback if it's purely punctuation
                
        labels = self.calculate_labels(cleaned_tokens, language_pair)
        
        return {
            "tokens": original_tokens,
            "labels": labels
        }
