# Word-Level LID Annotation Guidelines

## 1. Objective
Create a Gold Standard Evaluation Dataset (250–500 sentences) for Word-Level Language Identification (LID) in Telugu-English code-mixed text. This dataset will **only** be used for evaluation, never for training.

## 2. Tagset Definition
Every token separated by whitespace must be assigned exactly one of the following tags:
*   **`Te`**: Telugu (Native or Romanized)
*   **`En`**: English
*   **`Univ`**: Universal (Punctuation, Numbers, Symbols, Emojis, URLs, Hashtags, Mentions)
*   **`Mixed`**: Intra-word code-mixing (e.g., English root with Telugu suffix)
*   **`NE`**: Named Entities (Proper nouns)

## 3. Detailed Labeling Rules

### Rule 1: English (`En`)
*   Tokens that are standard English words, regardless of capitalization or minor typos (e.g., "good", "awsm", "Gud", "morning").
*   Common English abbreviations (e.g., "pls", "wbu", "idk").

### Rule 2: Telugu (`Te`)
*   Tokens written in native Telugu script (e.g., "బాగుంది").
*   Tokens written in Latin script (Romanized) that represent Telugu words (e.g., "bagundi", "chala", "andaru").
*   Regional slang and colloquialisms that originate from Telugu.

### Rule 3: Universal (`Univ`)
*   **Punctuation & Symbols:** `.`, `,`, `?`, `!`, `"`, `-`, `_`, `&`, `%`.
*   **Numbers:** Integers, decimals, and dates (e.g., `100`, `2.5`, `2024`).
*   **Emojis/Emoticons:** `:-)`, `😂`, `❤️`.
*   **URLs & Mentions:** `https://t.co/...`, `@username`.
*   **Standalone Hashtags:** `#TriMixGen`. (Note: If the hashtag contains a clear word, the word inside can be evaluated separately for complex tasks, but for base LID, tag as `Univ`).

### Rule 4: Named Entities (`NE`)
*   Names of people, places, organizations, or products (e.g., "Mahesh", "Hyderabad", "iPhone", "Netflix").
*   Even if a name has a Telugu or English origin, if it functions as a proper noun entity, tag it as `NE`. This prevents the LID model from penalizing unknown names as "incorrect language."

### Rule 5: Intra-Word Code-Mixing (`Mixed`)
*   Tokens that combine roots from one language with morphology (suffixes/prefixes) from another.
*   *Example:* "carloki" (car [En] + loki [Te, meaning 'into']).
*   *Example:* "voteveyyandi" (vote [En] + veyyandi [Te]).
*   If in doubt, and the root is primarily English but the conjugation is Telugu, tag as `Mixed`.

### Rule 6: Ambiguous Tokens
*   Tokens like "a", "O", "aa" that could be English articles or Telugu demonstratives.
*   **Resolution:** Context dictates the tag. If "a" is used as an article ("a car"), tag `En`. If "aa" is used as a demonstrative ("aa cinema", that movie), tag `Te`.

## 4. Annotation Process
1. A 500-sentence subset will be randomly sampled from the `HOLD-Telugu` test set.
2. Annotators will tag tokens strictly adhering to these rules.
3. Cross-validation will resolve disagreements on `Mixed` and `Ambiguous` tags.
