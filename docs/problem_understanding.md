# Phase 1: Problem Understanding - Code-Mixed NLP

### 1. Objective
Establish a rigorous theoretical foundation for Telugu-English code-mixing, understand the unique challenges of Indic NLP, and identify the motivations for building the TriMixGen system.

### 2. Theory
**What is Code-Mixing?**
Code-mixing is the phenomenon where multilingual speakers alternate between two or more languages within a single conversation or utterance. It is pervasive in multilingual societies (e.g., India) and heavily present in social media, messaging, and informal web text. 

**Types of Code-Mixing:**
1. **Intra-sentential:** Switching languages within a single sentence. (e.g., "Nenu repu *market ki* velthanu, *buy some* apples.")
2. **Inter-sentential:** Switching languages between sentences.
3. **Word-level (Intra-word):** Morphological blending of two languages (e.g., "Google-*chesava*?" - attaching a Telugu verb suffix to an English noun).

**Why Indic NLP is Difficult:**
- **Script Variation:** Indic languages are often written in the Latin script (Romanization/Transliteration) on social media, lacking standardized spelling conventions.
- **Morphological Richness:** Languages like Telugu are highly agglutinative, meaning words are formed by stringing together morphemes, resulting in a massive vocabulary size.
- **Resource Scarcity:** Compared to English, annotated datasets for Indic code-mixing are limited in size and quality.
- **Contextual Ambiguity:** Words like "ante" can be English (a poker bet) or Telugu (meaning "that is").

### 3. Implementation Plan
For this specific task (Task 1.1), the implementation plan involves synthesizing knowledge into this foundational document. Moving forward, this theoretical understanding will dictate our preprocessing (handling transliteration), tokenization (subword models for agglutinative words), and modeling (multilingual representations) strategies.

### 4. Folder Structure
```text
TriMixGen/
└── docs/
    └── problem_understanding.md  <-- (This file)
```

### 5. Code
*Since this is a theoretical task, there is no direct functional code. However, here is a conceptual representation of how we will structure our NLP pipeline later.*

```python
# Conceptual pipeline representation
from typing import List

def process_code_mixed_text(text: str) -> List[str]:
    """
    Conceptual pipeline for handling Telugu-English text.
    """
    # 1. Normalize transliterated text (e.g., spelling variations)
    normalized = normalize_text(text)
    
    # 2. Tokenize using a sub-word tokenizer (e.g., SentencePiece)
    tokens = tokenize(normalized)
    
    # 3. Predict Language ID for each token
    tags = identify_language(tokens)
    
    return tags
```

### 6. Explanation
The conceptual pipeline highlights our upcoming challenges. We cannot use standard whitespace tokenization because of the morphological richness of Telugu. Sub-word tokenizers (like WordPiece or SentencePiece) are mandatory. Furthermore, because spelling is non-standard in Romanized Telugu, the `normalize_text` step will be our first major engineering hurdle in Phase 4.

### 7. Common Mistakes
- **Using Monolingual Pre-trained Models:** Attempting to use a standard English BERT or a pure Telugu model on code-mixed data usually fails. The model gets confused by the unseen syntax and vocabulary. Multilingual models (mBERT, XLM-R) are essential.
- **Ignoring Transliteration:** Treating Romanized Telugu as English vocabulary.
- **Assuming Whitespace is Sufficient:** Telugu suffixes appended to English words ("car-lo") will ruin standard English embeddings if not properly tokenized into subwords (`car`, `##lo`).

### 8. Improvements
- **Future Integration:** We should consider using phonetic hashing (like Soundex modified for Indic languages) during our data cleaning phase to handle the immense spelling variations (e.g., "chudaledu", "choodaledu", "chudale").

### 9. Research References
1. *Sitaram, S., et al. (2019).* "A Survey of Code-mixed Data and Models in NLP."
2. *Chandu, K., et al. (2018).* "Language Identification in Code-Mixed Text using Word Embeddings."
3. *Khanuja, S., et al. (2020).* "GLUECoS: An Evaluation Benchmark for Code-Switched NLP."

### 10. Next Step
Proceed to **Task 1.2: Literature Review**. We will review 5-7 specific State-of-the-Art papers covering both Word-Level Language Identification (LID) and Code-Mixed Text Generation to inform our architecture choices.
