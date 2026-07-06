# Gold Annotation Plan and Guidelines

## 1. Objective
To create a high-quality Gold Standard Evaluation Dataset (250–500 sentences) for Word-Level Language Identification (LID) in Telugu-English code-mixed text. This dataset will only be used for evaluation, never for training.

## 2. Label Definitions
Every token (separated by whitespace or tokenized boundary) must be assigned exactly one of the following tags:
*   **`Te` (Telugu):** Words written in native Telugu script OR Latin script (Romanized) that represent Telugu vocabulary (e.g., "bagundi", "బాగుంది"). Includes regional slang.
*   **`En` (English):** Standard English words, regardless of capitalization, minor typos, or common abbreviations (e.g., "good", "awsm", "pls", "wbu").
*   **`Univ` (Universal):** Punctuation (`.`, `,`, `!`), numbers (`100`, `2.5`), symbols, emojis (`😂`), URLs, standalone hashtags, and mentions (`@username`).
*   **`Mixed` (Intra-word Mixing):** Tokens that combine roots from one language with morphology from another (e.g., "carloki" $\rightarrow$ car [En] + loki [Te]).
*   **`NE` (Named Entity):** Proper nouns (people, places, organizations, products), regardless of linguistic origin, to prevent penalizing unknown names (e.g., "Mahesh", "Hyderabad", "iPhone").

## 3. Ambiguous Token Rules
*   **Demonstratives vs Articles:** "a", "O", "aa".
    *   *Rule:* Context dictates the tag. If used as an English article ("a car"), tag `En`. If used as a Telugu demonstrative ("aa cinema" = that movie), tag `Te`.
*   **Borrowings:** Highly assimilated English words (e.g., "bus", "train") that are used daily in Telugu.
    *   *Rule:* If it is purely the English root ("bus"), tag `En`. If it has Telugu morphology ("buslu", "trainloki"), tag `Mixed`.

## 4. Quality Control & Validation Process
*   **Double-Blind Annotation:** Two independent annotators will label the same 500-sentence subset sampled randomly from the `HOLD-Telugu` corpus.
*   **Kappa Statistic:** Cohen's Kappa ($\kappa$) will be calculated to measure inter-annotator agreement. We target a $\kappa > 0.85$ (Strong agreement).
*   **Conflict Resolution:** 
    *   Any token where Annotator A and Annotator B disagree will be flagged.
    *   A third senior annotator (or joint consensus meeting) will review flagged tokens and make the final deterministic ruling based on the guidelines above.
*   **Sanity Checks:** Automated scripts will verify that URLs, mentions, and punctuation are consistently tagged as `Univ`.
