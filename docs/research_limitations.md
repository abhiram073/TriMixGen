# Research Limitations & Assumptions

## 1. Dataset Limitations
*   **Size Constraints:** Our generative dataset relies on a 52k Romanized Telugu corpus and a highly localized 4k code-mixed dataset (`HOLD-Telugu`). State-of-the-art LLMs typically train on millions of such tokens. Consequently, the zero-shot capability for novel slang outside the `HOLD-Telugu` distribution may be limited.
*   **Bias in Sentiment & Hate Speech Data:** The `HOLD-Telugu` dataset is specifically designed for Hate and Offensive Language Detection. This introduces a skewed semantic distribution (heavily weighted towards toxic, aggressive, or highly polarized YouTube comments). The generative model may inherit this aggressive tone if style-adaptation (LoRA Phase 2) is not heavily regularized.
*   **Lack of Canonical LID Data:** Since official word-level LID datasets (e.g., GLUECoS) are currently offline/gated, our training data for LID relies on a programmatic Pseudo-LID heuristic.

## 2. Annotation Assumptions (LID)
*   Our pseudo-LID training pipeline assumes that Unicode blocks cleanly separate native Telugu from English.
*   It assumes `fastText` can accurately distinguish Romanized Telugu from English at the word level, which introduces inherent noise. We assume the noise is random and that XLM-R can still learn the latent structure of code-mixing.
*   The Gold Standard LID evaluation relies on manual annotation of ~250-500 sentences. We assume this sample size is statistically significant enough to evaluate model performance, though a larger set (3k+) would be ideal.

## 3. Hardware Constraints
*   **Compute:** This project is designed to be executable on a local laptop environment (consumer-grade GPUs or MPS).
*   **Parameter-Efficient Fine-Tuning (PEFT):** To adhere to hardware constraints, we exclusively use QLoRA (4-bit quantization + Low-Rank Adaptation) for generative training. This caps our ability to update all model weights (Full Fine-Tuning), which might slightly reduce the absolute ceiling of model fluency.
*   **Context Window:** Due to VRAM limitations, the max sequence length for generation is capped at 256 tokens.

## 4. Future Improvements
*   **Scaling Data:** Integrate larger datasets like AI4Bharat's `IndicCMix` once gated access is approved.
*   **Full Fine-Tuning:** Execute full fine-tuning on A100/H100 clusters to compare against the LoRA baselines.
*   **Expanded Annotation:** Crowdsource a 10,000+ sentence word-level LID dataset to eliminate reliance on pseudo-LID heuristics.
