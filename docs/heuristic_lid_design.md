# TriMixGen – Heuristic Language Annotation Methodology

This document outlines the theoretical framework and decision logic for the **Pseudo-LID Tagging Engine**, the core component responsible for automatically generating word-level language identification labels for the `HOLD-Telugu` corpus. This heuristically labeled data will serve as the training set for our XLM-R Token Classifier.

---

## 1. Label Definitions

Every token evaluated by the annotator must be assigned exactly one of the following tags:

*   **`Te` (Telugu):** Words in native Telugu Unicode OR Latin-script words representing Telugu vocabulary (Romanized).
*   **`En` (English):** Standard English words, abbreviations, and common internet slang.
*   **`Univ` (Universal):** Punctuation, numbers, emojis, URLs, `<USER>` mentions, and standalone symbols.
*   **`Mixed` (Intra-word Mixing):** Tokens formed by combining English roots with Telugu morphological suffixes (e.g., "car-lo").
*   **`NE` (Named Entity):** Proper nouns, locations, and brand names, agnostic of linguistic origin.

---

## 2. Decision Hierarchy & Conflict Resolution

The annotator evaluates tokens using a rigid cascading waterfall logic. If a token satisfies a rule at a higher tier, it is tagged immediately, and lower tiers are skipped. This resolves conflicts deterministically.

1.  **Tier 1: Universal Tokens (Regex)**
    *   *Logic:* Matches `\d+`, punctuation, emojis, `<URL>`, `<USER>`.
    *   *Action:* Tag `Univ`.
2.  **Tier 2: Native Unicode Detection**
    *   *Logic:* Token contains characters in the `\u0C00-\u0C7F` block.
    *   *Action:* Tag `Te`.
3.  **Tier 3: Named Entity Recognition (NER)**
    *   *Logic:* Capitalized tokens not at the start of a sentence, or matched against a Spacy/Indic NER heuristic.
    *   *Action:* Tag `NE`.
4.  **Tier 4: Mixed Morphology Detection**
    *   *Logic:* Regex match for English word roots followed by common Telugu agglutinative suffixes (`lo`, `ki`, `ni`, `lu`, `nunchi`). e.g., `^[a-zA-Z]+(lo|ki|ni|lu)$`.
    *   *Action:* Tag `Mixed`.
5.  **Tier 5: English Dictionary Lookup**
    *   *Logic:* Query token against a local English dictionary (NLTK `words` + custom slang).
    *   *Action:* Tag `En`.
6.  **Tier 6: Romanized Telugu Fallback**
    *   *Logic:* If the token is Latin script but failed the English dictionary lookup, it is highly probable to be Romanized Telugu.
    *   *Action:* Tag `Te`.
7.  **Tier 7: Ambiguous Tokens**
    *   *Logic:* Short tokens (len $\leq$ 2) like "a", "aa", "O" that collide in both languages.
    *   *Action:* Contextual resolution. If adjacent to `En`, tag `En`. If adjacent to `Te`, tag `Te`.

---

## 3. Confidence Scoring

Each tag is assigned a confidence score $[0.0, 1.0]$. This allows us to weight the XLM-R loss function during training, penalizing the model less for disagreeing with low-confidence heuristic tags.

*   **1.0**: Native Telugu Unicode (`Te`), Numbers/Punctuation/Tags (`Univ`).
*   **0.9**: Standard English words matched in dictionary (`En`).
*   **0.8**: Mixed morphology regex matches (`Mixed`).
*   **0.7**: Romanized Telugu fallback (`Te`) and Named Entities (`NE`).
*   **0.5**: Ambiguous short tokens.

---

## 4. Examples by Label

| Token | Heuristic Tier | Tag | Confidence | Reason |
| :--- | :--- | :--- | :--- | :--- |
| `బాగుంది` | Tier 2 (Unicode) | `Te` | 1.0 | Native script. |
| `bagundi` | Tier 6 (Fallback) | `Te` | 0.7 | Latin script, not in English dict. |
| `awesome` | Tier 5 (Dict) | `En` | 0.9 | Matches English dictionary. |
| `carlo` | Tier 4 (Mixed) | `Mixed` | 0.8 | "car" (root) + "lo" (suffix). |
| `Hyderabad` | Tier 3 (NER) | `NE` | 0.7 | Capitalized proper noun. |
| `<URL>` | Tier 1 (Universal) | `Univ` | 1.0 | Special placeholder token. |

---

## 5. Failure Cases & Limitations

### Expected Precision and Recall Limitations
*   **Failure Case 1: Dictionary Collisions.** Romanized Telugu words that happen to be valid English words (e.g., "no" meaning 'no' in En, but "no" could be a typo for something else; "do", "me").
    *   *Impact:* False positives for `En`. Lowers `Te` recall.
*   **Failure Case 2: Code-Mixed Slang.** Aggressive internet slang ("lmao", "fr") might not be in our English dictionary.
    *   *Impact:* Defaults to Tier 6 (Fallback) and incorrectly tags as `Te`.
*   **Failure Case 3: Start-of-Sentence NER.** A named entity at the start of a sentence (capitalized) might be missed if we rely solely on casing heuristics.
    *   *Impact:* Tagged as `En` or `Te` instead of `NE`.

---

## 6. Comparison with Alternative Approaches

| Approach | Pros | Cons | Why TriMixGen avoids it |
| :--- | :--- | :--- | :--- |
| **fastText LID** | Excellent for sentence-level. | Fails completely on single-word Romanized inputs. | Too inaccurate for Word-Level token classification. |
| **API LLM (GPT-4)** | Near human accuracy. | Expensive, slow, violates local/open-source constraints. | We need a fast, offline, reproducible pipeline. |
| **TriMixGen Heuristic** | Fast, offline, interpretable. | Susceptible to dictionary collisions. | It perfectly balances our constraints while providing a "good enough" signal for XLM-R to generalize from. |

---

## 7. Rules for Creating the Pseudo-Labeled Training Set

1.  The `language_annotator.py` module will process the training split of `HOLD-Telugu` (~3,500 sentences).
2.  Any sentence where the average token confidence score falls below **0.75** will be discarded from the training set to maintain data quality.
3.  The resulting dataset will be saved in CoNLL-2003 format (`Token \t Tag`) suitable for Hugging Face Token Classification pipelines.

---

## 8. Validation Strategy Against Gold Dataset

The heuristic engine is **not** the final model; it is simply a label generator. To ensure it is generating valid data:
1.  We will run the Heuristic Annotator over the 500-sentence **Gold Annotated Test Set** (created in Step 4).
2.  We will calculate Precision, Recall, and F1-score for the Heuristic Annotator against human ground truth.
3.  If the Heuristic Annotator achieves an F1-score $> 0.80$, the generated training set is deemed high-quality enough to train the XLM-R neural network.

---

### User Review Required
Please review the complete heuristic methodology and decision hierarchy. If approved, I will implement `src/features/language_annotator.py` which executes this exact mathematical logic!
