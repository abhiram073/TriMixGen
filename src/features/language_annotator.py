import re
import yaml
from typing import List, Dict, Tuple, Any, Set
from src.features.unicode_utils import UnicodeUtils

class LanguageAnnotator:
    """
    Core Heuristic Language Annotator for TriMixGen.
    Generates Pseudo-LID tags (Te, En, Univ, Mixed, NE) using a 7-tier cascading logic.
    """

    def __init__(self, english_dict: Set[str] = None, config_path: str = "configs/preprocessing.yaml"):
        """
        Args:
            english_dict: A set of valid English words for Tier 5 lookup.
            config_path: Path to YAML config for confidence thresholds.
        """
        # Load configuration
        self.config = {
            'tier1_universal': 1.0,
            'tier2_unicode': 1.0,
            'tier3_ne': 0.7,
            'tier4_mixed': 0.8,
            'tier5_dict': 0.9,
            'tier6_romanized': 0.7,
            'tier7_ambiguous': 0.5,
            'dataset_thresholds': {'default': 0.75}
        }
        
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg and 'language_annotator' in cfg:
                    if 'confidence_thresholds' in cfg['language_annotator']:
                        self.config.update(cfg['language_annotator']['confidence_thresholds'])
                    if 'dataset_thresholds' in cfg['language_annotator']:
                        self.config['dataset_thresholds'] = cfg['language_annotator']['dataset_thresholds']
                    elif 'sentence_confidence_threshold' in cfg['language_annotator']:
                        self.config['dataset_thresholds']['default'] = cfg['language_annotator']['sentence_confidence_threshold']
        except Exception:
            pass # Fallback to defaults

        # Setup basic English dictionary if none provided
        self.english_dict = english_dict or {"this", "is", "a", "test", "movie", "good", "bad", "awesome", "hello", "hi"}
        
        # Pre-compile regexes
        self.univ_re = re.compile(r'^(\d+|[^\w\s]+|<URL>|<USER>|<emoji_[^>]+>)$')
        self.mixed_re = re.compile(r'^[a-zA-Z]+(lo|ki|ni|lu|nunchi|to)$', re.IGNORECASE)

    def _is_universal(self, token: str) -> bool:
        return bool(self.univ_re.match(token))

    def _is_named_entity(self, token: str, is_first: bool) -> bool:
        # Simplistic heuristic: capitalized and not the first word, or very long and capitalized
        if not token.isalpha() or not token[0].isupper():
            return False
        if not is_first:
            return True
        return False

    def annotate_token(self, token: str, is_first: bool, prev_tag: str = None, next_token: str = None) -> Tuple[str, float, str]:
        """
        Annotates a single token via the 7-tier logic.
        Returns: (Tag, Confidence, Heuristic_Tier)
        """
        # Tier 1: Universal
        if self._is_universal(token):
            return ('Univ', self.config['tier1_universal'], 'Tier_1_Univ')
            
        # Tier 2: Unicode Detection
        if any(UnicodeUtils.is_telugu_char(c) for c in token):
            return ('Te', self.config['tier2_unicode'], 'Tier_2_Unicode')
            
        # Tier 3: Named Entity
        if self._is_named_entity(token, is_first):
            return ('NE', self.config['tier3_ne'], 'Tier_3_NE')
            
        # Tier 4: Mixed Morphology
        if self.mixed_re.match(token):
            return ('Mixed', self.config['tier4_mixed'], 'Tier_4_Mixed')
            
        # Tier 5: English Dictionary Lookup
        clean_token = token.lower().strip(r'[^\w\s]')
        if clean_token in self.english_dict:
            return ('En', self.config['tier5_dict'], 'Tier_5_Dict')
            
        # Tier 6: Romanized Telugu Fallback
        if len(token) > 2:
            return ('Te', self.config['tier6_romanized'], 'Tier_6_Romanized')
            
        # Tier 7: Ambiguous Contextual Resolution
        # If length <= 2 and we have no strong signal, rely on context.
        tag = 'Unk'
        if prev_tag == 'En': tag = 'En'
        elif prev_tag == 'Te': tag = 'Te'
        else:
            tag = 'En' # Default base if no context
        return (tag, self.config['tier7_ambiguous'], 'Tier_7_Ambiguous')

    def annotate_sentence(self, tokens: List[str], dataset_name: str = "") -> Dict[str, Any]:
        """
        Annotates a list of tokens representing a sentence.
        Returns a rich object containing tags, confidences, traces, and sentence validity.
        """
        labels, confidences, traces = [], [], []
        
        # Resolve threshold
        threshold = self.config['dataset_thresholds'].get('default', 0.75)
        d_name_lower = dataset_name.lower()
        for k, v in self.config['dataset_thresholds'].items():
            if k != 'default' and k in d_name_lower:
                threshold = v
                break
        
        for i, token in enumerate(tokens):
            prev = labels[i-1] if i > 0 else None
            nxt = tokens[i+1] if i < len(tokens)-1 else None
            is_first = (i == 0)
            
            tag, conf, trace = self.annotate_token(token, is_first, prev, nxt)
            labels.append(tag)
            confidences.append(conf)
            traces.append(trace)
            
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        is_high_conf = avg_conf >= threshold
        
        return {
            "tokens": tokens,
            "labels": labels,
            "confidences": confidences,
            "traces": traces,
            "avg_confidence": avg_conf,
            "is_high_confidence": is_high_conf
        }

    def process_dataset(self, tokenized_sentences: List[List[str]], dataset_name: str = "") -> Dict[str, Any]:
        """
        Processes an entire corpus. Segregates into high-confidence (training set)
        and low-confidence (manual review set). Generates comprehensive report.
        """
        high_conf_set = []
        low_conf_set = []
        
        stats = {
            "label_distribution": {"Te": 0, "En": 0, "Univ": 0, "Mixed": 0, "NE": 0, "Unk": 0},
            "trace_distribution": {},
            "total_tokens": 0,
            "total_sentences": len(tokenized_sentences)
        }
        
        for tokens in tokenized_sentences:
            if not tokens: continue
            
            res = self.annotate_sentence(tokens, dataset_name=dataset_name)
            if res["is_high_confidence"]:
                high_conf_set.append(res)
            else:
                low_conf_set.append(res)
                
            for tag in res["labels"]:
                stats["label_distribution"][tag] += 1
            for trace in res["traces"]:
                stats["trace_distribution"][trace] = stats["trace_distribution"].get(trace, 0) + 1
            stats["total_tokens"] += len(tokens)
            
        return {
            "high_confidence_set": high_conf_set,
            "manual_review_set": low_conf_set,
            "report": stats
        }
