# TriMixGen Dataset Selection & Benchmark Report (v2 - Final)

## 1. Objective
Perform a fresh, verified discovery of publicly accessible datasets for TriMixGen. Due to the unavailability and gated nature of previous AI4Bharat datasets, we have devised a **Multi-Dataset Strategy** combining official Hugging Face datasets and GitHub-hosted DravidianLangTech shared task data. 

## 2. Dataset Benchmark Table

| Dataset Name | Official Source | Task | Language Pair | Size | License | Format | HF Availability | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HOLD-Telugu (DravidianLangTech)** | GitHub: `Salman1804102/DravidianLangTech-EACL-2024-HOLD` | Hate/Offensive Detection | Te-En (Code-Mixed) | ~4k samples | Academic/MIT | CSV | No | **Primary** |
| **Telugu Alpaca Romanized** | HF: `Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized` | Generation / Instruct | En -> Te (Romanized) | ~52k samples | Apache 2.0 | Parquet | Yes | **Primary** |
| **Telugu_Sentiment** | HF: `mounikaiiith/Telugu_Sentiment` | Sentiment Analysis | Te-En (Code-Mixed) | ~35k samples | MIT | Parquet | Yes | **Secondary** |

## 3. Multi-Dataset Strategy

No single verified dataset provides parallel generation data, word-level LID, and natural code-mixed text simultaneously. Therefore, we will distribute the objectives across three verified sources:

### Dataset A -> Natural Code-Mixed Text (DravidianLangTech)
*   **Source:** `Salman1804102/DravidianLangTech-EACL-2024-HOLD` (GitHub)
*   **Role:** This fulfills the requirement for FIRE/DravidianLangTech data. It contains raw, natural YouTube comments written in highly code-mixed Telugu-English. 
*   **Suitability:** We will use the text from this dataset for both downstream classification and Unsupervised Domain Adaptation.

### Dataset B -> Code-Mixed Text Generation (Alpaca Romanized)
*   **Source:** `Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized` (Hugging Face)
*   **Role:** Because `ai4bharat/IndicCMix` was gated, this is the best verified public alternative. It provides instruction-response pairs where the output is transliterated (Romanized) Telugu, exactly mirroring code-mixed conversational structures.
*   **Suitability:** Primary dataset for fine-tuning our sequence-to-sequence generation model (mT5/Llama).

### Dataset C -> Domain Adaptation & Sentiment
*   **Source:** `mounikaiiith/Telugu_Sentiment` (Hugging Face)
*   **Role:** An incredibly robust, easily accessible dataset containing ~35k samples of Telugu text.
*   **Suitability:** Will be used to pre-train/adapt our models to the Telugu vocabulary distribution before fine-tuning on the smaller `HOLD-Telugu` dataset.

### Dataset D -> Word-Level Language Identification (LID)
*   **Strategy:** True word-level LID datasets for Telugu (like GLUECoS) are currently offline. As a standard research practice, we will programmatically generate a **Pseudo-LID Dataset** during our Preprocessing Phase using Dataset A & C. We will tokenize the text and use Unicode-block heuristics (Telugu vs. Latin) combined with a fastText language detector to assign word-level tags (`En`, `Te`, `Univ`) to create our own high-quality LID training set.

## 4. Hardware Consideration & Scaling
*   **Initial Experiments (Laptop):** We will use the `HOLD-Telugu` dataset (~4k rows) for rapid prototyping, pipeline building, and LID modeling.
*   **Medium Validation:** `Telugu_Sentiment` (~35k rows) will validate our domain adaptation.
*   **Future Scaling:** `Telugu Alpaca Romanized` (~52k rows) will be used for LoRA generative fine-tuning.

## 5. Next Steps
1. Review this finalized selection.
2. I have updated `configs/datasets.yaml` and the download script to pull from both GitHub (raw CSVs) and Hugging Face (Parquet).
3. Upon your approval, we will execute the download and perform the EDA (Phase 3).
