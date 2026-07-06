# GEN_002: Dataset Strategy

## 1. Overview
The dataset for GEN_002 is derived from the **HOLD-Telugu** corpus, which consists of natural social media interactions in Romanized Telugu-English. Unlike the clean, instruction-tuned Alpaca dataset used in GEN_001, HOLD-Telugu is highly noisy, colloquial, and unstructured. Our data preparation pipeline is critical for ensuring the model learns high-quality conversational code-mixing.

## 2. Preprocessing & Filtering Strategy

### Filtering Strategy
Raw social media data contains significant amounts of unusable text. We will apply strict rule-based filtering to discard:
1. **Length Outliers**: Sentences containing fewer than 3 words or more than 100 words.
2. **Language Outliers**: Sentences that are 100% English or 100% Telugu. We will use a lightweight heuristic (or IndicBERT LID) to ensure only sentences containing at least one word from both languages are retained.
3. **URL & Artifact Removal**: Removal of `http://` links, `@mentions`, HTML tags, and excessive punctuation (e.g., `!!!!!` or `....`).

### Toxicity and Quality Filtering
Instead of simply removing all toxicity (which may strip out authentic, harmless colloquialisms), we will retain natural conversational code-mixed examples. We will remove only samples that are fundamentally unsuitable for generation (e.g., severe hate speech, explicit illegal content, spam). Every filtering rule applied will be thoroughly documented.

### Filtering Summary
A strict logging protocol will track the pipeline. We will generate a comprehensive filtering summary report that includes:
* Original dataset size
* Number of removed samples
* Number of retained samples
* Removal percentage
* Specific filtering reasons per category

### Conversational Sample Selection
We want the model to act as a conversational assistant. Therefore, we will filter the dataset to prioritize pairs that resemble Q&A or Instruction->Response formats. We will look for interrogative markers (e.g., "ela", "enti", "enduku", "?") to form the input contexts, and map the subsequent conversational replies as the target outputs.

### Duplicate Detection
Social media datasets suffer from massive duplication (e.g., repeated memes, standard greetings). We will apply MinHash / LSH (Locality Sensitive Hashing) to remove near-duplicate pairs. A Jaccard similarity threshold of `0.85` will be used to discard redundant samples, preventing the model from memorizing highly frequent phrases.

### Preprocessing
All text will undergo standardized normalization:
* **Lowercasing**: All text will be cast to lowercase to unify vocabulary (e.g., "Nenu" -> "nenu").
* **Spacing**: Extra whitespaces will be collapsed into a single space.
* **Emoji Stripping**: While emojis convey emotion, they confuse the sub-word tokenization of Romanized Telugu. All emojis will be stripped.

## 3. Train / Validation / Test Splits

We must maintain the strict evaluation protocols established in GEN_001 to ensure comparability across the curriculum learning stages.

* **Split Ratios**: We will reuse the exact fixed split ratios from GEN_001.
  * **Train**: 80%
  * **Validation**: 10%
  * **Test**: 10%
* **Reproducibility**: The split will be deterministic, controlled by `random_seed = 42`.
* **Static Evaluation Sets**: Once generated, the validation and test splits will be frozen as `data/processed/gen_002/valid.parquet` and `test.parquet`. These exact datasets **must be reused in GEN_003** so all curriculum stages remain directly comparable without data leakage or distribution shift.
