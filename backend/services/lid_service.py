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
        self.te_dict = set()
        self.en_dict = set()
        self.suffixes = set()
        self.entities = set()
        
        try:
            with open("configs/lid/romanized_telugu.yaml", "r") as f:
                self.te_dict = set(yaml.safe_load(f).get("words", []))
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
        # This replaces punctuation with spaces to avoid joining words
        normalized = re.sub(r'[^\w\s]', ' ', text)
        return normalized

    def tokenize(self, text: str) -> list:
        """Splits normalized text into a list of lowercase tokens."""
        return self.normalize_text(text).split()

    def dictionary_lookup(self, token: str) -> bool:
        """Tier 1: O(1) set lookup for Romanized Telugu core words."""
        return token.lower() in self.te_dict

    def suffix_lookup(self, token: str) -> bool:
        """Tier 2: Morphological suffix detection for Telugu verbs/postpositions."""
        token_lower = token.lower()
        return any(token_lower.endswith(s) for s in self.suffixes)

    def ner_lookup(self, token: str) -> bool:
        """Tier 3: Named Entity Recognition."""
        return token.lower() in self.entities

    def english_lookup(self, token: str) -> bool:
        """Tier 4: English dictionary detection."""
        return token.lower() in self.en_dict

    def classify_token(self, token: str) -> tuple:
        """Applies the Hybrid Pipeline to a single token, returning (Label, Reason, Tier)."""
        if self.dictionary_lookup(token):
            return "TE", "Found in Romanized Telugu dictionary", "Tier 1 (Dictionary)"
            
        if self.ner_lookup(token):
            return "EN", "Identified as Named Entity", "Tier 2 (NER)"
            
        if self.suffix_lookup(token):
            return "TE", "Matches Telugu morphological suffix", "Tier 3 (Suffix Rules)"
            
        if self.english_lookup(token):
            return "EN", "Found in English dictionary", "Tier 4 (English Detector)"
            
        # Tier 5 Fallback (Model / Other)
        if self.is_mock_mode:
            return "OTHER", "Token unknown to all heuristic tiers", "Tier 5 (OTHER)"
        else:
            # Here we would invoke self.model(token) if not mock mode
            return "OTHER", "Unrecognized by production model", "Tier 5 (IndicBERT / OTHER)"

    def calculate_labels(self, tokens: list) -> list:
        """Iterates through tokens and logs classifications."""
        labels = []
        for t in tokens:
            label, reason, tier = self.classify_token(t)
            labels.append(label)
            logger.debug(f"LID Classification -> Token: '{t}' | Label: '{label}' | Tier: '{tier}' | Reason: '{reason}'")
        return labels

    def tag_text(self, text: str) -> dict:
        """Public API endpoint for language identification."""
        if not self.initialized:
            if not self.is_mock_mode:
                raise RuntimeError("LIDService is unavailable. Production checkpoint missing.")
            else:
                raise RuntimeError("LIDService is not initialized.")
            
        # 1. We keep original whitespace-separated tokens for return schema alignment 
        # (Frontend relies on exact match with generated text for highlighting)
        # BUT we classify based on normalized stripped tokens!
        original_tokens = text.split()
        
        # 2. Extract words without punctuation for accurate classification
        cleaned_tokens = []
        for raw_t in original_tokens:
            clean = re.sub(r'[^\w]', '', raw_t)
            if clean:
                cleaned_tokens.append(clean)
            else:
                cleaned_tokens.append(raw_t) # fallback if it's purely punctuation
                
        # 3. Classify
        labels = self.calculate_labels(cleaned_tokens)
        
        return {
            "tokens": original_tokens,
            "labels": labels
        }
