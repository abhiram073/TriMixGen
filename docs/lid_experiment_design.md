# Phase 5: Word-Level LID Experimental Methodology

## 1. Research Objective
To develop, benchmark, and evaluate a highly robust word-level Language Identification (LID) model tailored for Telugu-English code-mixed and Romanized text. This model will replace the heuristic pipeline, providing context-aware predictions for downstream generation and sentiment analysis.

## 2. Dataset Preparation
*   **Training & Validation Data:** The 66,135 high-confidence sentences generated during the Phase 4 calibration (HOLD-Telugu, Telugu-Alpaca, Telugu-Sentiment).
*   **Final Evaluation Data:** The manually annotated Gold Standard dataset (250–500 sentences) created according to `annotation_guidelines.md`.
*   **Constraints:** The Gold Standard set will *never* be exposed during the training or hyperparameter tuning phases to ensure zero data leakage.

## 3. Data Split Strategy
The 66k pseudo-labeled dataset will be partitioned as follows:
*   **Train (80%):** ~52,900 sentences for model weight updates.
*   **Validation (10%):** ~6,600 sentences for early stopping and hyperparameter tuning.
*   **Test (10%):** ~6,600 sentences for intermediate benchmarking.
*   **Gold Standard:** Reserved exclusively for final benchmark comparisons across all models.

## 4. Baseline Models
To quantify the performance gains of deep learning, we will establish four baselines:
1.  **Rule-Based (Heuristic):** The 7-tier pipeline from Phase 4.
2.  **Conditional Random Field (CRF):** A classical sequence labeling baseline leveraging orthographic and contextual features (prefixes, suffixes, character n-grams).
3.  **BiLSTM:** A recurrent baseline capturing long-range contextual dependencies.
4.  **BiLSTM + Attention:** An augmented recurrent model prioritizing contextually significant tokens.

## 5. Transformer Models
We will benchmark three transformer architectures for sequence classification:
1.  **mBERT (Multilingual BERT):** Standard multilingual baseline.
2.  **XLM-R (XLM-RoBERTa):** *Primary Model.* Our target architecture due to its superior cross-lingual capabilities and large vocabulary size.
3.  **IndicBERT:** Comparison model specifically pretrained on Indian language corpora.

## 6. Evaluation Metrics
Models will be evaluated using standard sequence labeling metrics:
*   **Token Accuracy:** Overall percentage of correctly labeled words.
*   **Precision, Recall, F1 (Macro & Weighted):** Computed globally to account for class imbalances.
*   **Per-Class Metrics:** F1 scores specifically for `Te`, `En`, `Univ`, `Mixed`, and `NE`.
*   **Confusion Matrix:** To visually identify systemic misclassifications.

## 7. Error Analysis Methodology
Post-evaluation, we will extract the top 100 highest-loss misclassifications from the Gold Standard set. We will specifically analyze:
*   *Boundary Errors:* Misclassifications occurring at code-switch points.
*   *Romanized Ambiguity:* `En` tokens incorrectly tagged as Romanized `Te`, or vice versa.
*   *Named Entities:* `NE` tokens incorrectly captured as `En` or `Te`.

## 8. Ablation Studies
To justify methodological decisions, we will conduct the following ablations:
1.  **Dataset Composition:** Training XLM-R with and without the Romanized `Telugu-Alpaca` dataset to measure its impact on code-mixed generalization.
2.  **Context Window:** Evaluating BiLSTM and XLM-R on isolated tokens vs. full sentence context.

## 9. Computational Requirements
*   **CRF:** CPU (Local), ~5 GB RAM, ~10 mins training.
*   **BiLSTM:** GPU (T4/V100), ~8 GB VRAM, ~2 hours training.
*   **Transformers:** GPU (A100/V100 or Kaggle/Colab), ~16 GB VRAM, ~6-8 hours training for XLM-R (Batch size 16-32, max_length 128).

## 10. Experiment Tracking
*   **Primary:** Weights & Biases (`wandb`) for real-time loss tracking, hyperparameter logging, and artifact management.
*   **Secondary:** TensorBoard for local offline visualization.

---

## 11. Model Comparison Table (Expected Format)

| Model | Token Accuracy | Macro F1 | Weighted F1 | Te (F1) | En (F1) | Mixed (F1) | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Rule-Based (V1.0) | - | - | - | - | - | - | 0 min |
| CRF | - | - | - | - | - | - | ~10m |
| BiLSTM | - | - | - | - | - | - | ~2h |
| BiLSTM + Attn | - | - | - | - | - | - | ~2.5h |
| mBERT | - | - | - | - | - | - | ~6h |
| IndicBERT | - | - | - | - | - | - | ~6h |
| **XLM-R** | **-** | **-** | **-** | **-** | **-** | **-** | **~8h** |

---

## 12. Implementation Order & Roadmap

1.  **Environment Prep:** Setup PyTorch, Transformers, `seqeval`, and `wandb`.
2.  **Dataset Class:** Build PyTorch `Dataset` and `DataLoader` for token-level classification using the 66k pseudo-labeled dataset.
3.  **Gold Standard Evaluation Script:** Build the automated evaluation script that compares predictions against the Gold Standard.
4.  **Baseline 1 (CRF):** Implement, train, evaluate, and log to `wandb`.
5.  **Baseline 2 (BiLSTM variants):** Implement in PyTorch, train, evaluate.
6.  **Transformer Fine-Tuning:** Implement XLM-R, mBERT, and IndicBERT pipelines.
7.  **Benchmarking & Ablation:** Execute the experimental matrix.
8.  **Final Report:** Produce Phase 5 results.

> [!IMPORTANT]
> **User Review Required:** Please review this experimental methodology. Are there any additional baselines (e.g., fastText), metrics, or ablations you would like to introduce before we begin implementation?
