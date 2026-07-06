# Final Master Benchmark: Word-Level LID for Telugu-English Code-Mixed Text

This document represents the finalized evaluation of all models tested for the Language Identification (LID) module of the TriMixGen pipeline. All models were evaluated against the frozen **Gold Standard LID v1.0** dataset.

## 1. Master Performance Table

| Rank | Model Architecture | Macro F1 | Accuracy | Mixed F1 | NE F1 | Inference Latency | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (Prod)** | **IndicBERT** | **0.8720** | 0.9380 | **0.79** | 0.77 | **~12 ms** | **135 MB** |
| 2 | XLM-RoBERTa | 0.8780 | 0.9410 | 0.77 | **0.80** | ~105 ms | 1.1 GB |
| 3 | mBERT | 0.8450 | 0.9250 | 0.68 | 0.75 | ~15 ms | 711 MB |
| 4 | CRF | 0.4300 | 0.8400 | 0.02 | 0.00 | ~2 ms | <5 MB |
| 5 | BiLSTM + Attention | 0.2471 | 0.7800 | 0.00 | 0.00 | ~8 ms | 15 MB |
| 6 | BiLSTM | 0.1814 | 0.7100 | 0.00 | 0.00 | ~5 ms | 12 MB |
| 7 | Rule-Based (Heuristics)| N/A | N/A | 0.00 | 0.00 | <1 ms | 0 MB |

## 2. Model Evolution Summary

1.  **Rule-Based:** Established that deterministic dictionaries cannot solve code-mixing due to heavy lexical overlap in Romanized scripts.
2.  **CRF:** Proved that local sequence transitions (Markov window) help, but completely failed on Out-Of-Vocabulary (OOV) tokens and Named Entities (NE).
3.  **BiLSTM:** Attempted bidirectional global context but collapsed into majority-class prediction due to class imbalance.
4.  **BiLSTM + Attention:** Solved the context decay and majority collapse using Attention and Weighted Loss, successfully isolating English morphology, but still failed entirely on Named Entities because it lacked semantic priors.
5.  **mBERT:** The breakthrough. Introduced WordPiece tokenization (solving OOV) and massive pre-trained priors (solving NEs). Macro F1 jumped by +0.60.
6.  **IndicBERT (Production Model):** Solved the final hurdle: Morphological Code-Switching (`Mixed` tokens). Its ALBERT-based architecture provided a 3x speedup, while its Indic-specific SentencePiece vocabulary perfectly tokenized Romanized Telugu roots and suffixes.
7.  **XLM-RoBERTa:** Proved the ultimate performance ceiling. While it scored marginally higher in overall Accuracy/Macro F1 due to 270M parameters, it was rejected for production due to its 9x slower inference speed and slight degradation on `Mixed` tokens compared to IndicBERT.

## 3. Production Deployment Decision
**IndicBERT** is officially designated as the production Word-Level LID tagger for TriMixGen. It provides the optimal balance of state-of-the-art semantic accuracy and high-throughput computational efficiency required for large-scale dataset generation.
