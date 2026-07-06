# TriMixGen – Preprocessing Calibration Report

## 1. Executive Summary
Following the discovery of heuristic bottlenecks in Preprocessing Version 1.0 (specifically the "Romanized Confidence Trap" and dynamic schema failures), a formal Calibration Phase was executed.

We implemented:
*   Dataset-specific confidence thresholds mapped in `configs/preprocessing.yaml`.
*   A robust, alias-driven schema resolver (`data_utils.py`).
*   An automatic header-recovery engine to rescue improperly scraped datasets (e.g., `HOLD-Telugu_test`).

This calibration completely resolved the data retention bottlenecks without sacrificing the scientific rigor of our code-mixed heuristic engine.

---

## 2. Threshold Calibration Experiments

To empirically derive the highest quality thresholds for each dataset, we processed the entire corpora through the heuristic engine and measured retention at multiple intervals.

### 2.1 Code-Mixed Dataset (HOLD-Telugu)
*   *Nature:* Highly noisy, natural social media text mixing English (0.9), Universal (1.0), and Romanized (0.7).
*   *Experiment:*
    *   `0.60`: 99.9% retention (too permissive, admits garbage)
    *   `0.65`: 96.2% retention
    *   **`0.70`**: 71.9% retention (optimal balance of noise reduction and data volume)
    *   `0.75`: 45.9% retention (too strict)
*   **Calibrated Threshold:** `0.70`

### 2.2 Romanized Generation Dataset (Telugu-Alpaca)
*   *Nature:* 100% Romanized Telugu instructions and outputs. (Base token confidence = 0.70).
*   *Experiment:*
    *   `0.60`: 99.9% retention
    *   **`0.65`**: 95.8% retention (optimal: allows pure Romanized text, drops ambiguous noise)
    *   `0.70`: 10.7% retention (The Romanized Trap: strict equality failure)
    *   `0.75`: 1.5% retention
*   **Calibrated Threshold:** `0.65`

### 2.3 Native Script Dataset (Telugu-Sentiment)
*   *Nature:* Clean, native Telugu script (Base token confidence = 1.0).
*   *Experiment:*
    *   `0.60 - 0.75`: 100.0% retention
*   **Calibrated Threshold:** `0.75` (Default strict fallback)

---

## 3. Final Preprocessing Pipeline Results (Version 1.1)

After hardcoding the empirical thresholds into `configs/preprocessing.yaml`, the end-to-end pipeline was executed.

| Dataset | Total Rows | Valid Rows | Schema Action | Retained (High Conf) | Manual Review | Retention % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HOLD-Telugu_train** | 4,000 | 4,000 | Direct Alias | 2,876 | 1,124 | 71.9% |
| **HOLD-Telugu_test** | 500 | 500 | **Header Recovery** | 395 | 105 | 79.0% |
| **Telugu-Alpaca-Romanized**| 28,910 | 28,910 | Direct Alias | 27,722 | 1,188 | 95.8% |
| **Telugu-Sentiment (All)** | 35,142 | 35,142 | Direct Alias | 35,142 | 0 | 100.0% |

**Total Pseudo-Labeled Sentences Generated:** `66,135`

### Conclusion
The calibration phase successfully rescued 27,000+ Romanized Alpaca instruction pairs from erroneous rejection and dynamically recovered the headless HOLD-Telugu test set. 

**Status:** Preprocessing Version 1.0 is successfully calibrated, validated, and frozen. We are ready to proceed to Phase 5.
