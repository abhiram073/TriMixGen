import re
import html
import unicodedata
from typing import Optional

class TextNormalizer:
    """
    A configurable text normalization pipeline for NLP.
    Processes raw text by removing HTML, standardizing Unicode,
    replacing entities (URLs, mentions), and normalizing characters.
    """
    def __init__(
        self,
        collapse_whitespace: bool = True,
        lowercase_latin: bool = True,
        max_repeated_chars: int = 2,
        normalize_repeated_punct: bool = True,
        url_placeholder: Optional[str] = "<URL>",
        mention_placeholder: Optional[str] = "<USER>",
        remove_hashtag_symbol: bool = True,
        clean_html: bool = True,
        unicode_form: Optional[str] = "NFC"
    ):
        """
        Args:
            collapse_whitespace: Collapses multiple spaces into a single space and trims edges.
            lowercase_latin: Lowercases Latin alphabet characters (leaves Indic scripts unchanged).
            max_repeated_chars: Threshold for repeated letters (e.g., 2 converts "baaaagundi" to "baagundi").
            normalize_repeated_punct: Collapses repeated punctuation (e.g., "!!!" to "!").
            url_placeholder: String to replace URLs. If None, URLs are left intact.
            mention_placeholder: String to replace @mentions. If None, mentions are left intact.
            remove_hashtag_symbol: Removes '#' but preserves the tag text.
            clean_html: Unescapes HTML entities (e.g., &amp; -> &).
            unicode_form: Form to normalize Unicode (NFC recommended for Indic scripts).
        """
        self.collapse_whitespace = collapse_whitespace
        self.lowercase_latin = lowercase_latin
        self.max_repeated_chars = max_repeated_chars
        self.normalize_repeated_punct = normalize_repeated_punct
        self.url_placeholder = url_placeholder
        self.mention_placeholder = mention_placeholder
        self.remove_hashtag_symbol = remove_hashtag_symbol
        self.clean_html = clean_html
        self.unicode_form = unicode_form
        
        # Pre-compile regex patterns for efficiency
        self.url_re = re.compile(r'https?://\S+|www\.\S+')
        self.mention_re = re.compile(r'@\w+')
        self.hashtag_re = re.compile(r'#(\w+)')
        self.whitespace_re = re.compile(r'\s+')
        
        # Repeated punctuation (matches 2+ occurrences of standard sentence boundaries/separators)
        self.punct_re = re.compile(r'([.?!,-])\1+')

    def normalize(self, text: str) -> str:
        """Executes the normalization pipeline on a string."""
        if not text or not isinstance(text, str):
            return ""
            
        # 1. HTML Cleanup
        if self.clean_html:
            text = html.unescape(text)
            
        # 2. Case Normalization (Done early to ensure placeholders remain uppercase)
        # Python's lower() affects Latin characters but has no effect on Telugu 
        # characters as they lack casing, perfectly satisfying our requirement.
        if self.lowercase_latin:
            text = text.lower()
            
        # 3. URL Normalization
        if self.url_placeholder is not None:
            text = self.url_re.sub(self.url_placeholder, text)
            
        # 4. Mention Normalization
        if self.mention_placeholder is not None:
            text = self.mention_re.sub(self.mention_placeholder, text)
            
        # 4. Hashtag Handling
        if self.remove_hashtag_symbol:
            text = self.hashtag_re.sub(r'\1', text)
            
        # 6. Unicode Normalization
        # NFC (Canonical Composition) is highly recommended for Telugu to ensure
        # that distinct vowels/consonants are composed into single standard codepoints
        # where applicable, avoiding rendering and tokenization fragmentation.
        if self.unicode_form:
            text = unicodedata.normalize(self.unicode_form, text)
            
        # 7. Repeated Punctuation Normalization
        if self.normalize_repeated_punct:
            # Replaces '!!!' with '!'
            text = self.punct_re.sub(r'\1', text)
            
        # 8. Repeated Character Normalization
        if self.max_repeated_chars > 0:
            # Matches any non-whitespace character repeated more than max_repeated_chars times.
            # Replaces the entire match with exactly max_repeated_chars of that character.
            # Example for max=2: "soooo" -> "soo", "baaaaagundi" -> "baagundi"
            pattern = r'([^\s])\1{' + str(self.max_repeated_chars) + r',}'
            text = re.sub(pattern, lambda m: m.group(1) * self.max_repeated_chars, text)
            
        # 9. Whitespace Normalization
        if self.collapse_whitespace:
            text = self.whitespace_re.sub(' ', text).strip()
            
        return text
