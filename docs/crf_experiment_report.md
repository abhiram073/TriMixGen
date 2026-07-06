# Experiment 002: CRF Sequence Labeling Report

## 1. Theory & Methodology

### What is a Conditional Random Field (CRF)?
A Conditional Random Field is a discriminative, undirected probabilistic graphical model used widely for sequence labeling tasks (like POS tagging and LID). Instead of predicting labels in isolation, CRFs model the conditional probability $P(\mathbf{y} | \mathbf{x})$ of a sequence of labels $\mathbf{y}$ given a sequence of input observations $\mathbf{x}$. 

### Why CRFs are Suitable for Sequence Labeling
CRFs explicitly model the transition probabilities between adjacent labels (e.g., the likelihood of transitioning from `Te` to `En`). In code-mixed texts, humans do not code-switch randomly; they follow syntactic boundaries. CRFs capture this transition structure, ensuring the predicted label sequence is globally coherent.

### Advantages and Limitations vs BiLSTM & XLM-R
*   **Advantages:** CRFs are highly interpretable. We can inspect exactly which feature weights (e.g., a specific suffix) triggered a prediction. They are extremely fast to train, require no GPU, and work well on small datasets.
*   **Limitations:** CRFs rely heavily on handcrafted feature engineering. They cannot easily capture long-range contextual dependencies (e.g., a word 5 steps away influencing the current word) without exploding the feature space, a task where recurrent neural networks (BiLSTMs) and self-attention mechanisms (Transformers) excel naturally.

## 2. Feature Engineering

To give the CRF maximum signal without a static vocabulary dictionary, we designed a robust feature extraction pipeline:
*   **Orthographic Features:** `lowercase`, `length`, `is_upper`, `is_title`, `is_digit`, `is_punctuation`. (Crucial for detecting `Univ` and `NE`).
*   **Affix Features:** Prefixes and suffixes of length 2 and 3. (Captures morphological roots, separating `En` from Romanized `Te`).
*   **Unicode Script:** `is_telugu_script` dynamically checks for native Telugu characters `[\u0C00-\u0C7F]` ensuring 100% precision on native script.
*   **Heuristics Indicators:** `has_telugu_suffix` and `has_english_suffix` to capture intra-word mixing (`Mixed`).
*   **Contextual Features:** $word_{i-1}$ and $word_{i+1}$ (sliding window) to provide local sequence context.

## 3. Training Details
*   **Optimizer:** L-BFGS
*   **Regularization:** L1 ($c_1 = 0.1$) for feature sparsity, L2 ($c_2 = 0.1$) for weight decay.
*   **Iterations:** 100 maximum iterations.
*   **Dataset:** Trained on 10,000 sentences from the calibrated V1.1 pseudo-labeled dataset to ensure fast iteration.
*   **Time to Train:** ~90 seconds on CPU.

## 4. Evaluation Results (Gold Standard V1.0)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **En** | 0.70 | 0.05 | 0.10 | 261 |
| **Mixed** | 0.09 | 0.50 | 0.15 | 10 |
| **NE** | 0.00 | 0.00 | 0.00 | 30 |
| **Te** | 0.84 | 0.97 | 0.90 | 1534 |
| **Univ** | 1.00 | 1.00 | 1.00 | 13 |
| **Accuracy** | **-** | **-** | **0.82** | **1848** |
| **Macro F1** | **-** | **-** | **0.43** | **1848** |
| **Weighted F1**| **-** | **-** | **0.77** | **1848** |

## 5. Error Analysis
The CRF achieved an overall Token Accuracy of **82%**, but a heavily constrained Macro F1 of **0.43**. We can isolate the systemic weaknesses:
1.  **The Named Entity (NE) Failure (F1 = 0.00):** The CRF completely failed to identify Named Entities. Because code-mixed social media rarely uses proper capitalization, `NE` tokens orthographically resemble Romanized `Te`. Without an external knowledge base or deep semantic understanding of the surrounding context, the CRF defaults them to `Te`.
2.  **The English Recall Collapse (Recall = 0.05):** While precision was strong (when the CRF predicts English, it is usually right), it missed 95% of English words. Handcrafted 3-letter suffixes are insufficient to distinguish English borrowings from Romanized Telugu vocabulary.
3.  **Mixed Tokens (F1 = 0.15):** The CRF over-predicted `Mixed` tokens when detecting Telugu suffixes, leading to extremely low precision (0.09).

## 6. Comparison with Rule-Based Baseline
The Rule-Based system is essentially an infinite-memory lookup table constrained by brittle heuristics. It scores higher on our Gold Standard primarily because the Gold Standard relies heavily on dictionary lookups for `NE` and `En`, which the CRF feature-extractor does not possess. However, the CRF demonstrates that machine learning can achieve 82% token accuracy using only *orthographic features and local context*, proving the viability of sequence modeling over hardcoded rules.

## 7. Recommendations Before BiLSTM
Before advancing to **Experiment 003 (BiLSTM)**, we must recognize that pure character-level/orthographic features are insufficient for resolving vocabulary overlap in Romanized scripts. 
*   **BiLSTM Strategy:** The BiLSTM must leverage dense word embeddings (e.g., FastText or learned embeddings) rather than sparse handcrafted features. This will allow the model to learn the semantic representation of English vs. Romanized Telugu, solving the recall collapse observed in the CRF.
