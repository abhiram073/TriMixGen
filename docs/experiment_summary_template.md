# Experiment [ID] Results Summary: [Model Name]

## Core Metrics
*   **Accuracy:** [Accuracy]
*   **Macro F1:** [Macro F1]
*   **Weighted F1:** [Weighted F1]

## Training Profile
*   **Epochs Run:** [Epochs]
*   **Best Validation Loss:** [Val Loss]
*   **Training Time:** [Time]
*   **Hardware:** [CPU/GPU]

## Recommendation Framework
### 1. Baseline Comparison
*Did this model significantly improve upon the previous model (e.g., CRF)?*
[Analysis of Macro F1 Delta, specifically looking at English and Named Entity recall]

### 2. Failure Identification
*Where does this model systematically fail?*
[Analysis of confusion matrix hotspots]

### 3. Justification for Next Architecture
*Why do we need the next model in the sequence? (e.g., Attention or Transformer)*
[Explanation of why recurrent context is insufficient, or why pre-trained weights are necessary]

### 4. Proceed Decision
**Recommendation:** [Proceed to Experiment X / Re-tune hyperparameters / Add Features]
