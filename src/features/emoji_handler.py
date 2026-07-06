import emoji
import re
import yaml
from typing import Dict, Any, List

class EmojiHandler:
    """
    Handles emojis and emoticons in text without performing broader normalization.
    Configurable via configs/preprocessing.yaml.
    """
    
    # Regex to match ASCII emoticons like :), :(, :D, ;), etc.
    EMOTICON_PATTERN = re.compile(
        r'(?::|;|=)(?:-)?(?:\)|\(|D|P|p|O|o|/|\\|\|)'
    )
    
    # Direct mapping for ASCII emoticons to standard Unicode Emojis
    EMOTICON_TO_EMOJI = {
        ':)': '🙂', ':-)': '🙂', '=)': '🙂',
        ':(': '☹️', ':-(': '☹️', '=(':'☹️',
        ':D': '😃', ':-D': '😃', '=D': '😃',
        ';)': '😉', ';-)': '😉',
        ':P': '😛', ':-P': '😛', ':p': '😛', ':-p': '😛',
        ':O': '😮', ':-O': '😮', ':o': '😮', ':-o': '😮'
    }

    # Sentiments for classifying emojis (subset for social media context)
    SENTIMENT_MAP = {
        # Positive / Laughter / Love
        '👍': 'positive', '❤️': 'love', '😍': 'love', '🙂': 'positive', '😃': 'laughter', 
        '😂': 'laughter', '🤣': 'laughter', '😊': 'positive', '🥰': 'love', '😘': 'love',
        '🙏': 'positive', '👏': 'positive', '🎉': 'positive', '🙌': 'positive',
        # Negative / Anger / Sadness
        '👎': 'negative', '😡': 'anger', '🤬': 'anger', '😠': 'anger',
        '😭': 'sadness', '😢': 'sadness', '☹️': 'negative', '😞': 'sadness', '😔': 'sadness',
        '💔': 'negative', '🤦‍♂️': 'negative', '🤦‍♀️': 'negative', '🙄': 'negative',
        # Surprise
        '😮': 'surprise', '😱': 'surprise', '😲': 'surprise', '😳': 'surprise'
    }

    def __init__(self, strategy: str = None, config_path: str = "configs/preprocessing.yaml"):
        """
        Args:
            strategy: Overrides config file if provided. 
                      Options: 'preserve', 'remove', 'replace_descriptive', 'replace_sentiment'
            config_path: Path to YAML config file.
        """
        self.strategy = strategy
        if not self.strategy:
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if config and 'emoji_handler' in config:
                        self.strategy = config['emoji_handler'].get('strategy', 'preserve')
                    else:
                        self.strategy = 'preserve'
            except Exception:
                self.strategy = 'preserve'
                
        valid_strategies = ['preserve', 'remove', 'replace_descriptive', 'replace_sentiment']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")
            
        # Normalize sentiment map keys by removing variant selectors to ensure robust matching
        self.normalized_sentiment_map = {k.replace('\ufe0f', ''): v for k, v in self.SENTIMENT_MAP.items()}

    def _replace_emoticons(self, text: str) -> str:
        """Converts ASCII emoticons to Unicode emojis for uniform processing."""
        def replace(match):
            emoticon = match.group(0)
            return self.EMOTICON_TO_EMOJI.get(emoticon, emoticon)
        return self.EMOTICON_PATTERN.sub(replace, text)

    def handle(self, text: str) -> str:
        """
        Applies the configured strategy to text containing emojis and emoticons.
        Handles mixed emoji sequences natively via the emoji package.
        """
        if not text:
            return ""
            
        # First, convert emoticons to emojis
        text = self._replace_emoticons(text)
        
        if self.strategy == 'preserve':
            return text
            
        elif self.strategy == 'remove':
            return emoji.replace_emoji(text, replace='')
            
        elif self.strategy == 'replace_descriptive':
            # e.g. "😂" -> "<emoji_face_with_tears_of_joy>"
            def replace_desc(e, _):
                desc = emoji.demojize(e, language='en').strip(':')
                return f" <emoji_{desc}> "
            # Replace multiple spaces with single space after replacement
            processed = emoji.replace_emoji(text, replace=replace_desc)
            return re.sub(r'\s+', ' ', processed).strip()
            
        elif self.strategy == 'replace_sentiment':
            # e.g. "😂" -> "<emoji_laughter>"
            def replace_sent(e, _):
                # We strip variant selectors if present before lookup
                clean_e = e.replace('\ufe0f', '')
                sent = self.normalized_sentiment_map.get(clean_e, 'miscellaneous')
                return f" <emoji_{sent}> "
            processed = emoji.replace_emoji(text, replace=replace_sent)
            return re.sub(r'\s+', ' ', processed).strip()
            
        return text

    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Calculates dataset-level emoji classification statistics."""
        stats = {
            "total_emojis_found": 0,
            "classes": {
                "positive": 0, "negative": 0, "neutral": 0, 
                "laughter": 0, "surprise": 0, "sadness": 0, 
                "anger": 0, "love": 0, "miscellaneous": 0
            }
        }
        
        for text in texts:
            if not text: continue
            text = self._replace_emoticons(text)
            
            for d in emoji.emoji_list(text):
                stats["total_emojis_found"] += 1
                e_char = d['emoji'].replace('\ufe0f', '')
                cls = self.normalized_sentiment_map.get(e_char, "miscellaneous")
                stats["classes"][cls] += 1
                
        return stats
