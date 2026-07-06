# Word-Level LID Experiment Management Framework

## 1. Global Experiment Settings

To ensure strict reproducibility and comparability, all experiments will adhere to the following global constants:

*   **Dataset Version:** V1.1 (Calibrated 66k Pseudo-Labeled Data)
*   **Preprocessing Version:** V1.1 (Config: HOLD=0.70, Alpaca=0.65, Sentiment=0.75)
*   **Gold Standard Version:** Gold Standard LID v1.0
*   **Global Random Seed:** 42 (Applied to PyTorch, NumPy, Python Random, and Hugging Face)

## 2. Standardized Output Structure

Every model implementation will generate outputs isolated within its own experiment directory to prevent logging overlap.

```text
data/
└── outputs/
    └── experiments/
        ├── exp_001A_heuristic_consistency/
        ├── exp_002_crf/
        ├── exp_003_bilstm/
        ├── exp_004_bilstm_attn/
        ├── exp_005_mbert/
        ├── exp_006_indicbert/
        └── exp_007_xlm_r/
            ├── config.yaml               # Frozen hyperparameters & environment state
            ├── train.log                 # Epoch-wise training loss and val metrics
            ├── metrics.json              # Final test/gold evaluation scores
            ├── predictions.csv           # Sentence, Ground Truth, Prediction, Confidence
            ├── confusion_matrix.png      # Matplotlib output of class predictions
            ├── classification_report.txt # Per-class precision, recall, F1
            ├── model_summary.txt         # Architecture dump (layers, param count)
            └── checkpoints/              # Saved model weights
```

## 3. Implementation Order & Success Criteria

Experiments must be executed sequentially. Progression to the next model is contingent on meeting the architectural success criteria defined below.

### Exp 001A: Rule-Based (Heuristics Consistency)
*   **Objective:** Evaluate heuristic consistency on a non-benchmark dataset. This is NOT the official benchmark.
*   **Success Criteria:** Generates predictions and `classification_report.txt` on the heuristic-initialized evaluation set. No training required.

### Exp 002: Conditional Random Field (CRF)
*   **Objective:** Establish a classical machine learning sequence baseline.
*   **Hyperparameters:** L1 Penalty = 0.1, L2 Penalty = 0.1, Max Iterations = 100.
*   **Features:** Orthographic (is_upper, is_digit), Affixes (prefix-3, suffix-3), Context (word-1, word+1).
*   **Success Criteria:** Successfully trains without OOM, outperforms Rule-Based macro F1, and produces the standardized output directory.

### Exp 003: BiLSTM
*   **Objective:** Establish a recurrent neural network baseline to capture long-range sequence context.
*   **Hyperparameters:**
    *   *Optimizer:* AdamW (LR: 1e-3, Weight Decay: 1e-4)
    *   *Scheduler:* ReduceLROnPlateau
    *   *Batch Size:* 64
    *   *Epochs:* 10
    *   *Evaluation Frequency:* End of each epoch
    *   *Checkpoint Policy:* Save `best_model.pt` based on Validation Macro F1.
*   **Success Criteria:** Model converges (training loss decreases stably) and inference generates the confusion matrix correctly.

### Exp 004: BiLSTM + Attention
*   **Objective:** Evaluate the impact of additive/dot-product attention on recurrent states.
*   **Hyperparameters:** Same as Exp 003, with an added Attention layer configuration.
*   **Success Criteria:** Training completes; validation F1 equals or slightly exceeds the standard BiLSTM.

### Exp 005: mBERT
*   **Objective:** Multilingual transformer baseline.
*   **Hyperparameters:**
    *   *Optimizer:* AdamW (LR: 2e-5)
    *   *Scheduler:* Linear warmup with decay (warmup_ratio=0.1)
    *   *Batch Size:* 16 or 32 (memory dependent)
    *   *Epochs:* 3-5
    *   *Evaluation Frequency:* Every 500 steps and end of epoch.
    *   *Checkpoint Policy:* Top-1 validation F1.
*   **Success Criteria:** Successfully integrates with Hugging Face `Trainer` or PyTorch lightning, outputs standard predictions, and clearly outperforms all recurrent models.

### Exp 006: IndicBERT
*   **Objective:** Compare against a transformer explicitly pre-trained on Indian corpora.
*   **Hyperparameters:** Identical to mBERT.
*   **Success Criteria:** Training completes stably.

### Exp 007: XLM-R (Primary Model)
*   **Objective:** Train our ultimate model target.
*   **Hyperparameters:** Identical to mBERT, adjusting batch size for XLM-R memory requirements (e.g., gradient accumulation steps if VRAM is exceeded).
*   **Success Criteria:** Achieves the highest F1 score on the Gold Standard evaluation set, finalizing the TriMixGen LID component.

---
> [!IMPORTANT]
> **Review Checkpoint**
> This experiment tracker will govern the entire development of Phase 5. Please confirm the configuration architecture, or propose any modifications to hyperparameters and the execution order.
