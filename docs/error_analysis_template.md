# Error Analysis Framework: [Model Name]

*Evaluate the performance of [Model Name] on the Gold Standard dataset across the following core code-mixing failure modes.*

## 1. Romanized Telugu
- **Description:** Cases where Romanized Telugu is misclassified (often as English due to Latin script vocabulary overlap).
- **Analysis:** [Insert observations]

## 2. Native Telugu
- **Description:** Cases where native Telugu script `[\u0C00-\u0C7F]` is misclassified.
- **Analysis:** [Insert observations]

## 3. Mixed Tokens
- **Description:** Intra-word mixing (e.g., English root + Telugu suffix like "car-loki", "college-ki").
- **Analysis:** [Insert observations]

## 4. Named Entities (NE)
- **Description:** Proper nouns classified incorrectly (usually as Te or En instead of NE).
- **Analysis:** [Insert observations]

## 5. Ambiguous Tokens
- **Description:** Words that exist in both languages (e.g., "bus", "train", "a", "O") misclassified based on a lack of context.
- **Analysis:** [Insert observations]

## 6. Morphological Mixing
- **Description:** Errors arising from heavily inflected words.
- **Analysis:** [Insert observations]

## 7. Unknown Words (OOV)
- **Description:** Words not seen during training or missing from pre-trained vocabularies.
- **Analysis:** [Insert observations]

## 8. Emoji-related Errors
- **Description:** Errors caused by adjacent emojis or failure to tag emojis as Univ.
- **Analysis:** [Insert observations]
