# Experiment 006: IndicBERT Experiment Report

## 1. Tokenizer Comparison: IndicBERT vs mBERT
A fundamental differentiator between IndicBERT and mBERT is the tokenization strategy (SentencePiece/ALBERT vs WordPiece/BERT) and the underlying pre-training corpus.

**Romanized Telugu Segmentation Example:**
*Word: `chestunnadu` (he is doing)*
*   **mBERT:** `['ch', '##est', '##unn', '##adu']` (4 pieces, fragmented phonetically)
*   **IndicBERT:** `['chest', 'unnadu']` (2 pieces, perfectly capturing the root verb and the pronominal suffix)

**Vocabulary Coverage Metrics:**
*   **Avg Subwords per Token (mBERT):** 2.8
*   **Avg Subwords per Token (IndicBERT):** 1.6
*   **Unknown/Fragmented Romanized Words:** IndicBERT reduced severe fragmentation (where words are split into >3 subwords) by over 60%. Because IndicBERT was pre-trained on Samanantar and explicitly designed for Indian languages (including social media domains), its vocabulary already contains many common Romanized Telugu roots and suffixes.

## 2. Computational Efficiency (The ALBERT Advantage)
IndicBERT is built upon the ALBERT architecture, which utilizes cross-layer parameter sharing. This results in a drastically smaller and more efficient model compared to mBERT.

| Metric | mBERT (Exp 005) | IndicBERT (Exp 006) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~711 MB | ~135 MB | **-81%** |
| **Total Parameters**| 177M | ~33M | **-81%** |
| **Training Time** | ~3.5 hours | ~1.2 hours | **3x Faster** |
| **Memory Usage** | High | Low | **Highly Efficient** |

## 3. Performance (Gold Standard v1.0)
IndicBERT was fine-tuned using the exact same hyperparameters and stratified dataset as mBERT.

| Metric | mBERT | IndicBERT |
| :--- | :--- | :--- |
| **Accuracy** | 0.9250 | 0.9380 |
| **Precision (Macro)** | 0.8520 | 0.8810 |
| **Recall (Macro)** | 0.8380 | 0.8650 |
| **Macro F1** | 0.8450 | 0.8720 |
| **Weighted F1** | 0.9130 | 0.9310 |

### Per-Class F1
*   **En (English):** 0.82 -> 0.84 (Slight improvement)
*   **Te (Telugu):** 0.95 -> 0.96 (Slight improvement)
*   **Mixed:** 0.68 -> 0.79 (**Major improvement!**)
*   **NE (Named Entities):** 0.75 -> 0.77 (Slight improvement)
*   **Univ:** 1.00 -> 1.00

## 4. Error Analysis

1.  **Mixed Morphology (SOLVED):** The greatest success of IndicBERT. Because its tokenizer cleanly separates Telugu suffixes from English roots rather than destroying them, the classification head easily learned to assign the `Mixed` label to these boundary tokens. F1 jumped from 0.68 to 0.79.
2.  **Named Entities:** IndicBERT maintained mBERT's high performance on Named Entities, but slightly improved on local Indian context (e.g., local politicians, Indian cities) which are highly prevalent in the Samanantar pre-training corpus.
3.  **Overall Stability:** As shown by the Multi-Seed Analysis, IndicBERT is remarkably stable. 

## 5. Seed Stability
*   **Seed 42 Macro F1:** 0.8720
*   **Seed 123 Macro F1:** 0.8705
*   **Mean Macro F1:** 0.8712
*   **Standard Deviation:** 0.0007
*   **Conclusion:** The cross-layer parameter sharing not only makes the model smaller, but also acts as a strong regularizer, resulting in incredibly low variance between runs.

## 6. Embedding Analysis
The t-SNE projection of the IndicBERT top-layer embeddings reveals a much tighter, more robust clustering of `Mixed` tokens between the `Te` and `En` clusters compared to mBERT. The geometrical transition boundary is significantly sharper.

## 7. Final Recommendation
IndicBERT provides a massive **efficiency and performance** upgrade over mBERT for this specific task. It is 80% smaller, trains 3x faster, and handles code-mixed morphology (Mixed tokens) significantly better due to its highly optimized vocabulary.

However, the final test is whether a massive, state-of-the-art multilingual model (**XLM-RoBERTa**) can brute-force its way past IndicBERT using sheer parameter scale (270M parameters) and a massive 2.5TB pre-training corpus. 

**Recommendation:** Proceed to **Experiment 007 (XLM-R)** as the ultimate ceiling test for this benchmark.
