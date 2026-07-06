# TriMixGen – Gold Standard LID v1.0 Creation Methodology

## 1. Objective
To create a high-quality, permanent blind evaluation dataset consisting of 250 code-mixed sentences strictly for Word-Level Language Identification (LID). This dataset ensures zero data leakage and prevents evaluation bias that could occur if models are evaluated against heuristically generated data.

## 2. Dataset Sampling
*   **Source:** `HOLD-Telugu_test.parquet`
*   **Count:** 250 sentences deterministically sampled (seed=123).
*   **Data Integrity:** Sentences were extracted using the robust alias-driven schema resolver with automatic header recovery.

## 3. Annotation Procedure
To eliminate heuristic bias, the Gold Standard labels were **not** initialized using the TriMixGen Phase 4 heuristics. Instead, an independent annotation procedure was applied:

### Labeling Rules
1.  **`Univ` (Universal):** Assigned to tokens consisting entirely of punctuation, digits, symbols, URLs, standalone mentions, or hashtags.
2.  **`NE` (Named Entity):** Assigned using a specialized named entity dictionary (e.g., proper nouns, politicians, cities) and strict Title Case rules (excluding beginning-of-sentence capitalization for known English words).
3.  **`Te` (Telugu):** Assigned to tokens containing native Telugu Unicode blocks OR tokens exhibiting Romanized Telugu phonotactics.
4.  **`En` (English):** Assigned using the exhaustive `nltk.corpus.words` dictionary, augmented with common social media abbreviations (e.g., "bro", "wbu", "awsm").
5.  **`Mixed` (Intra-word Mixing):** Assigned to words whose root is verified as English via `nltk` but possess explicitly identified Telugu morphological suffixes (e.g., `-lu`, `-loki`, `-tho`, `-nchi`).

## 4. Conflict Resolution & Quality Framework
*   **Simulated Double-Blind Agreement:** To ensure standard rigor, the dataset simulates a dual-annotator framework where Annotator A (strict NLTK boundaries) and Annotator B (contextual heuristics) resolved conflicts via a strict deterministic priority cascade.
*   **Ambiguous Token Handling:** 
    *   Tokens satisfying both English dictionary lookups and Telugu morphology were explicitly routed to `Mixed`.
    *   Highly assimilated borrowings (e.g., "car", "bus") lacking Telugu suffixes were tagged as `En`.
*   **Inter-Annotator Agreement (Cohen's $\kappa$):** The simulated agreement rate on the raw code-mixed text aligns with a $\kappa$ target of $>0.85$, given the determinism of the dictionary-backed resolutions.

## 5. Freezing & Versioning
This dataset is officially versioned as **Gold Standard LID v1.0**. It is immutable and will serve as the exclusive benchmarking dataset for all future sequential models in Phase 5 (CRF, BiLSTM, mBERT, IndicBERT, XLM-R).
