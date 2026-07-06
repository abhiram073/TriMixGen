# TriMixGen LID Model Benchmark Comparison

This document serves as the master tracking table for all Word-Level Language Identification models evaluated on the frozen Gold Standard LID v1.0 dataset.

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | Training Time | Inference Speed | Parameters | Model Size | Memory Usage | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule-Based (V1.1)** | | | | | | 0 min | Fast | 0 | 0 MB | Low | Absolute transparency, perfect script ID | Fails on context-dependent ambiguity |
| **CRF** | | | | | | ~1.5 min | Fast | ~200k | ~10 MB | Low | Captures transition probabilities | Fails on Named Entities and English OOV |
| **BiLSTM** | | | | | | | | | | | | |
| **BiLSTM + Attention** | | | | | | | | | | | | |
| **mBERT** | | | | | | | | | | | | |
| **IndicBERT** | | | | | | | | | | | | |
| **XLM-R (Target)** | | | | | | | | | | | | |
