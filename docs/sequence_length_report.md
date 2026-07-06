# Sequence Length Analysis Report

Before optimizing the training pipeline for the BiLSTM, we analyzed the sequence lengths (in tokens) of our V1.1 pseudo-labeled code-mixed dataset (25,000 sentences).

## Empirical Distribution
*   **Minimum:** 1 token
*   **Mean:** 128.91 tokens
*   **Median:** 111.0 tokens
*   **90th Percentile (p90):** 281 tokens
*   **95th Percentile (p95):** 322 tokens
*   **99th Percentile (p99):** 396 tokens
*   **Maximum:** 507 tokens

## Recommendation & Expected Impact

We recommend enforcing a **`max_length = 256`** for the BiLSTM training pipeline. 

### Why 256?
1.  **Percentage of Sequences Truncated:** At 256 tokens, we are slightly below the 90th percentile. This means ~85% of sequences will remain completely untouched. For the 15% of sequences that exceed 256 tokens, we will truncate the tail end (tokens 257+).
2.  **Expected Reduction in Training Time:** Because recurrent models (LSTMs) must process tokens sequentially, and the PyTorch DataLoader pads batches to the maximum sequence length *in that batch*, an outlier of 507 tokens forces the CPU to compute hundreds of unnecessary hidden states per sequence. Enforcing a hard cutoff at 256 will cut the maximum recurrent depth in half, reducing our training time per epoch from ~14 minutes to **~6-8 minutes**, allowing the full experiment to complete in a reasonable timeframe.
3.  **Expected Impact on Model Performance:** Negligible to slightly positive. Since Word-Level LID is highly localized (context usually matters most within a 5-10 word window), dropping the tail end of massive 300+ token paragraphs does not rob the model of meaningful sequence transition signals. In fact, keeping the sequences shorter prevents the BiLSTM's hidden state from accumulating too much noise over extremely long code-mixed blocks.

## Actions Taken
1.  Updated `SequenceDataset` to accept `max_length`.
2.  Updated `config.yaml` and `scripts/run_exp_003_bilstm.py` to utilize `max_length=256`.
3.  Restarted Experiment 003 from scratch using the optimized pipeline.
