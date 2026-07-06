import re

def is_telugu_script(token: str) -> bool:
    """Check if token contains Telugu Unicode characters."""
    return bool(re.search(r'[\u0C00-\u0C7F]', token))

def has_english_suffix(token: str) -> bool:
    """Check if token ends with common English suffixes (basic heuristic)."""
    suffixes = ('ing', 'ed', 'ly', 's', 'es', 'tion', 'ness', 'ment')
    return token.lower().endswith(suffixes)

def has_telugu_suffix(token: str) -> bool:
    """Check if Romanized token ends with common Telugu morphological suffixes."""
    suffixes = ('lu', 'loki', 'ki', 'ni', 'lo', 'tho', 'nchi', 'nundi', 'ku', 'na', 'ga', 'di')
    return any(token.lower().endswith(s) and len(token) > len(s) for s in suffixes)

def contains_emoji(token: str) -> bool:
    """Check if token contains emojis."""
    return bool(re.search(r'[\U00010000-\U0010ffff]', token))

def extract_word_features(sentence, i):
    """
    Extract features for a specific word in a sentence.
    
    Includes:
    - Current token properties (lowercase, length, casing, digits, punctuation)
    - Affixes (prefixes and suffixes of length 2 and 3)
    - Unicode scripts (Telugu, emoji)
    - Code-mixing heuristics (Romanized Telugu suffix matches)
    - Contextual features (Previous word, Next word)
    """
    word = sentence[i]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length()': len(word),
        
        # Affixes
        'word.prefix-2': word[:2].lower(),
        'word.prefix-3': word[:3].lower(),
        'word.suffix-2': word[-2:].lower(),
        'word.suffix-3': word[-3:].lower(),
        
        # Script and Char Properties
        'word.is_telugu_script': is_telugu_script(word),
        'word.has_telugu_suffix': has_telugu_suffix(word),
        'word.has_english_suffix': has_english_suffix(word),
        'word.contains_emoji': contains_emoji(word),
        'word.is_punctuation': bool(re.match(r'^[\W]+$', word)),
        'word.is_hashtag': word.startswith('#'),
        'word.is_mention': word.startswith('@'),
        'word.is_url': 'http' in word.lower() or 'www' in word.lower(),
    }
    
    # Previous word context
    if i > 0:
        word1 = sentence[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.is_telugu_script': is_telugu_script(word1),
        })
    else:
        features['BOS'] = True # Beginning of sentence

    # Next word context
    if i < len(sentence)-1:
        word1 = sentence[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.is_telugu_script': is_telugu_script(word1),
        })
    else:
        features['EOS'] = True # End of sentence

    return features

def extract_sentence_features(sentence):
    """Apply feature extraction to all words in a sentence."""
    return [extract_word_features(sentence, i) for i in range(len(sentence))]
