# Classical to Transformer Transition

This document outlines the theoretical and empirical justifications for transitioning from classical machine learning models and recurrent neural networks (RNNs) to pre-trained Transformer architectures for the task of Word-Level Language Identification (LID) in code-mixed Telugu-English text.

## 1. Why Classical Models Reached Their Performance Ceiling

Over the course of Experiments 001A through 004, we systematically tested the limits of classical NLP techniques. We discovered that each architectural enhancement solved one problem while exposing a deeper, insurmountable bottleneck:

*   **Rule-Based (Heuristics):** Achieved perfect precision for native scripts (Unicode boundaries) but suffered from catastrophic recall failures due to dictionary overlap. It proved that code-mixing cannot be solved by deterministic lookup tables.
*   **CRF (Sequence Labeling):** Improved robustness by learning local orthographic transition probabilities. However, it was strictly limited to a fixed Markov window, meaning it could not leverage long-range syntactic dependencies. It completely failed on Named Entities (NE) and Out-Of-Vocabulary (OOV) tokens.
*   **BiLSTM (Recurrence):** Captured full-sentence bidirectional context, slightly boosting absolute accuracy. However, because it was trained from scratch on heavily imbalanced data without prior semantic knowledge, it lazily collapsed into a majority-class predictor (tagging almost everything as Telugu).
*   **BiLSTM + Attention (Weighted Loss):** We solved the context-decay problem using Self-Attention and the majority-collapse problem using inverse-frequency loss weighting. This successfully isolated English morphology. Yet, the model still failed entirely on Named Entities (F1 = 0.00).

**The Final Ceiling:** Classical models trained from scratch on small, domain-specific datasets (even 25,000 sentences) simply do not possess enough linguistic exposure to learn the deep semantic differences between Romanized Telugu ("bagundi") and English proper nouns ("Hyderabad" / "John"). Because they are orthographically identical, and the models lack a pre-trained semantic worldview, they hit an absolute performance ceiling.

## 2. Why Transformers Will Overcome These Limitations

Pre-trained multilingual Transformers (such as mBERT and XLM-R) fundamentally alter the learning paradigm through two major mechanisms:

### A. Subword Tokenization (WordPiece)
Classical models use word-level tokenization. If a word was not seen in the 25k training sentences, its embedding is a random vector (`<UNK>`). Transformers use subword tokenization (e.g., WordPiece). Unknown words are deterministically split into known morphological sub-units (e.g., `playing` -> `play`, `##ing`). This mathematically eliminates the OOV problem and allows the model to leverage morphological similarity for unseen code-mixed variations.

### B. Massive Pre-Trained Semantic Priors
mBERT was pre-trained on Wikipedia dumps of 104 languages using Masked Language Modeling (MLM). Before we even begin fine-tuning for our LID task, mBERT already *knows* what English looks like, what Named Entities represent distributionally, and how syntax flows. We are no longer asking a model to learn a language from scratch; we are merely asking it to map its existing, vast semantic knowledge to our specific 5 class labels.

## 3. Hypotheses Tested in Experiment 005 (mBERT)

By fine-tuning mBERT for token classification, we are testing the following hypotheses:

1.  **The NE Hypothesis:** mBERT will achieve a non-zero F1 score on the Named Entity (`NE`) class, which all classical models failed at, because its pre-trained embeddings already capture the distributional semantics of proper nouns.
2.  **The OOV Hypothesis:** Subword tokenization will significantly boost recall on rare `Mixed` tokens and previously unseen English insertions compared to word-level RNNs.
3.  **The Contextual Disambiguation Hypothesis:** By projecting tokens into a deeply contextualized embedding space, mBERT will successfully disambiguate ambiguous tokens (e.g., "bus" vs romanized Telugu slang) better than local Attention over random embeddings.
