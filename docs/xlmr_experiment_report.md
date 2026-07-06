# Experiment 007: XLM-RoBERTa Experiment Report

## 1. Tokenizer Comparison: WordPiece vs SentencePiece

We compared how the three transformer tokenizers segment heavily code-mixed text.

| Feature | mBERT (WordPiece) | IndicBERT (SentencePiece) | XLM-R (SentencePiece) |
| :--- | :--- | :--- | :--- |
| **Vocab Size** | 119K | 200K (Indic specific) | 250K (Global) |
| **Avg Subwords/Token**| 2.8 | 1.6 | 1.9 |
| **Romanized Te Seg.**| Poor (Heavy fragmentation) | Excellent (Preserves morphology) | Good (Moderate fragmentation) |
| **Named Entities** | Excellent | Excellent | Excellent |
| **Mixed Tokens** | Poor | Excellent | Good |

*Analysis:* While XLM-R has a massive 250K vocabulary, it is distributed across 100 global languages. Thus, its coverage of Romanized Telugu slang is slightly worse than IndicBERT (which dedicated its vocabulary purely to Indic scripts and their Romanized counterparts).

## 2. Computational Analysis
XLM-RoBERTa is a behemoth compared to the other models.

| Metric | IndicBERT | XLM-R Base | Cost Increase |
| :--- | :--- | :--- | :--- |
| **Parameters** | 33M | 278M | **8.4x** |
| **Model Size** | 135 MB | 1.1 GB | **8.1x** |
| **CPU RAM Usage** | ~1.5 GB | ~4.5 GB | **3.0x** |
| **Training Time** | ~1.2 hrs | ~6.5 hrs | **5.4x** |
| **Inference Latency** | ~12 ms/seq | ~105 ms/seq | **8.7x** |

## 3. Performance (Gold Standard v1.0)

| Metric | mBERT | IndicBERT | XLM-RoBERTa |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.9250 | 0.9380 | **0.9410** |
| **Macro F1** | 0.8450 | 0.8720 | **0.8780** |
| **Weighted F1** | 0.9130 | 0.9310 | **0.9350** |
| **Mixed F1** | 0.68 | **0.79** | 0.77 |
| **NE F1** | 0.75 | 0.77 | **0.80** |

*Analysis:* XLM-R narrowly beats IndicBERT in overall Macro F1 and Accuracy due to its sheer parameter scale and massive semantic understanding of global Named Entities (NE F1 = 0.80). However, IndicBERT still outperformed XLM-R on `Mixed` tokens because its tokenizer cleanly isolated the Telugu suffixes that XLM-R fragmented.

## 4. Embedding Analysis
The t-SNE projection of XLM-R embeddings shows beautiful, distinct manifolds for English, Telugu, and Named Entities. However, the `Mixed` tokens are slightly more dispersed across the decision boundary compared to IndicBERT's incredibly tight clustering, reflecting the slight drop in Mixed F1.

## 5. Production Recommendation

**Selected Production Model: IndicBERT**

**Justification:**
While XLM-RoBERTa technically achieved a +0.006 increase in Macro F1 (0.878 vs 0.872), deploying it in a production data pipeline is unjustifiable from an engineering perspective:
1.  **Latency:** XLM-R is nearly 9x slower at inference. TriMixGen will need to process millions of tokens to generate the final code-mixed corpus. Using XLM-R would bottleneck the entire generation pipeline.
2.  **Resource Cost:** XLM-R requires 1.1 GB of VRAM/RAM compared to IndicBERT's 135 MB.
3.  **Task Alignment:** IndicBERT actually outperformed XLM-R on the most difficult structural class (`Mixed` tokens) due to its specialized tokenizer. 

Therefore, **IndicBERT** provides 99% of the performance of the world's best cross-lingual model at 12% of the computational cost. It is the perfect production model for Word-Level LID.
