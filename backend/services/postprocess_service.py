import re

class PostprocessService:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans generated text by normalizing whitespace, 
        fixing punctuation spacing, and removing artifacts.
        """
        if not text:
            return ""
            
        # Remove URLs (http/https and domains ending with .com, .nz, etc)
        text = re.sub(r'https?:\/\/[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.(com|org|net|io|in|co|uk|nz|us|edu)(\/[^\s]*)?', '', text)
        
        # Remove HTML and XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove Markdown backticks
        text = re.sub(r'[`]', '', text)
        
        # Remove standalone special tokens if any leaked
        text = re.sub(r'<extra_id_\d+>', '', text)
        
        # Remove repeated punctuation (e.g., !!? -> !?)
        text = re.sub(r'([.?!])\1+', r'\1', text)
        
        # Remove broken unicode or weird artifacts like .):
        text = re.sub(r'\.\):?', '', text)
        text = re.sub(r'[^\w\s.,!?:;\'-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing (e.g., "word , word" -> "word, word")
        text = re.sub(r'\s+([,.:;?!])', r'\1', text)
        
        return text.strip()
