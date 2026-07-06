# TriMixGen – Preprocessing Validation Report (Version 1.0 Candidate)

## 1. Overview
The end-to-end preprocessing pipeline (`Validator` $\rightarrow$ `UnicodeUtils` $\rightarrow$ `Normalizer` $\rightarrow$ `EmojiHandler` $\rightarrow$ `LanguageAnnotator`) was executed against the raw datasets to generate the pseudo-labeled training sets.

## 2. Execution Summary

### Dataset: HOLD-Telugu_train
*   **Input Rows:** 4000
*   **Rows Removed:** 0
*   **Script Distribution:** Latin (55.1%), Telugu (34.7%), Mixed (10.0%)
*   **Pipeline Execution Time:** 1.08 seconds
*   **Label Distribution:**
    *   Te: 37,414
    *   En: 539
    *   Univ: 423
    *   Mixed: 1,442
*   **Pseudo-Labeled (High Confidence):** 1,836 sentences (45.9%)
*   **Manual Review (Low Confidence):** 2,164 sentences (54.1%)

### Dataset: Telugu-Alpaca-Romanized_train
*   **Input Rows:** 28,910
*   **Rows Removed:** 0
*   **Script Distribution:** Latin (99.8%)
*   **Pipeline Execution Time:** 110.3 seconds
*   **Label Distribution:**
    *   Te: 4,092,263
    *   En: 175,498
    *   Univ: 54,736
    *   Mixed: 8,451
*   **Pseudo-Labeled (High Confidence):** 454 sentences (1.5%)
*   **Manual Review (Low Confidence):** 28,456 sentences (98.5%)

---

## 3. Annotation Quality Inspection (100-Sentence Sample)

I randomly sampled 100 annotated sentences from the `HOLD-Telugu` corpus and inspected the heuristic tags.

**Representative Successes:**
1.  *Native Telugu:* `ప్రతి తండ్రి కోరుకునేది ఇదే.` 
    *   **Result:** Perfectly tagged `Te` (Tier 2 Unicode). Punctuation reduced via Normalizer. Average Confidence = 1.0.
2.  *Code-Mixed English:* `సూపర్ బ్రదర్ కామెడీ కేక్క 1000% కామెడీ వీడియో`
    *   **Result:** The numbers (`1000`) were tagged `Univ`. The native script was tagged `Te`. Average Confidence = 1.0.

**Identified Heuristic Failures & Systematic Errors:**
1.  **The Romanized Confidence Trap (CRITICAL):**
    *   **Issue:** The `Telugu-Alpaca-Romanized` dataset had a 98.5% failure rate, dumping nearly all 28,000 sentences into the `manual_review` pile.
    *   **Cause:** The dataset is 100% Romanized Telugu. The heuristic pipeline assigns Romanized tokens to `Tier_6_Romanized`, which currently grants a confidence score of **`0.70`**. However, our `sentence_confidence_threshold` is set to **`0.75`**. Because every token in a sentence scores `0.70`, the average is `0.70`, causing every single valid sentence to fail the threshold!
    *   **Impact:** We generated almost zero pseudo-labeled data from our largest corpus.
2.  **Dataset Schema Errors:**
    *   **Issue:** The script failed entirely on the test/validation splits of `HOLD-Telugu` and `Telugu-Sentiment`.
    *   **Cause:** `HOLD-Telugu_test.parquet` was scraped without a header row (the first comment *is* the column name). `Telugu-Sentiment` uses the column name `Sentence` instead of `text` or `Comments`.

---

## 4. Final Recommendations Before Freezing Version 1.0

Before we freeze the preprocessing pipeline and move to Phase 5 (Model Training), I strongly recommend we implement the following two adjustments:

1.  **Rebalance Confidence Thresholds in `configs/preprocessing.yaml`:**
    *   *Option A:* Lower `sentence_confidence_threshold` from `0.75` to `0.65`.
    *   *Option B:* Raise `tier6_romanized` confidence from `0.70` to `0.80`.
    *   *Recommendation:* **Option A**. Lowering the sentence threshold to `0.65` allows pure Romanized sentences (0.7) to pass while still rejecting extremely ambiguous (0.5) garbage sentences.
2.  **Fix Raw Data Schemas:**
    *   Update `scripts/run_preprocessing.py` to handle `Sentence` as a text column.
    *   Manually insert a generic header into `HOLD-Telugu_test` to prevent pandas from using data as a column name.
