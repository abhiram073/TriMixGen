# Phase 2: Dataset Collection & Benchmarking

### 1. Objective
Identify, evaluate, and acquire publicly available Telugu-English code-mixed datasets. Establish a benchmark to compare these datasets across size, quality, and specific tasks (Sentiment, LID, Generative constraints).

### 2. Theory
Code-mixed datasets are notoriously difficult to construct. They require bilingual annotators and are often scraped from social media (YouTube comments, Twitter). Because they come from informal sources, they are highly noisy. For TriMixGen, we need datasets that provide word-level language identification tags or at least parallel text for generation tasks.

**Prominent Dataset Sources:**
- **DravidianCodeMix (FIRE / DravidianLangTech):** Massive collection of YouTube comments annotated for sentiment and offensive language. While primarily sentence-level, they provide excellent raw code-mixed text for unsupervised pre-training or generation fine-tuning.
- **GLUECoS:** While primarily focused on English-Hindi/Spanish, the methodology sets the benchmark.
- **AI4Bharat (IndicCorp):** Provides massive monolingual corpora, but we can extract code-mixed instances using heuristics.

### 3. Implementation Plan
1. **Benchmark Table:** Create a reference table of the top 3 datasets for our use case.
2. **Download Script:** Write a robust Python script using the `datasets` library from Hugging Face to pull these into our `data/raw/` folder.
3. **Storage:** All datasets will be saved locally as JSONL or Parquet files for fast I/O during the pandas-based EDA phase.

### 4. Folder Structure
```text
TriMixGen/
├── data/
│   └── raw/               <-- (Datasets will be downloaded here)
├── docs/
│   └── dataset_benchmark.md <-- (This file)
└── scripts/
    └── download_data.py   <-- (Script to execute the download)
```

### 5. Code (Benchmark & Script)

#### Dataset Benchmark

| Dataset Name | Task | Size | License | Strengths | Weaknesses | Intended Use in TriMixGen |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DravidianCodeMix (Telugu)** | Sentiment/Offensive | ~54k comments | CC-BY 4.0 | Real-world YouTube data, highly natural code-mixing. | Highly noisy, spelling errors, missing word-level LID tags. | Generative fine-tuning and unsupervised domain adaptation. |
| **CMTET (Kusampudi et al.)** | Sentiment Analysis | ~5k sentences | Academic | Curated specifically for Telugu-English. | Small size. | Validation set for generation. |
| **LID-Telugu-English (Custom/Scraped)** | Language ID (Word) | ~10k words | Academic | Specifically tagged per token (En, Te, Univ). | Hard to find publicly, often requires manual scraping. | Primary training dataset for Phase 5 (LID). |

*Note: For the purpose of this project, we will simulate the acquisition of a word-level tagged dataset for LID, while relying on the Hugging Face `DravidianCodeMix` for bulk generative text.*

#### Download Script (`scripts/download_data.py`)
```python
import os
import pandas as pd
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

def download_dravidian_codemix(output_dir: str):
    """
    Downloads the DravidianCodeMix Telugu dataset from HuggingFace.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Downloading DravidianCodeMix Telugu dataset...")
    
    try:
        # Note: Using a popular community upload of the DravidianLangTech dataset
        dataset = load_dataset("appalaraju/telugu_english_codemixed_sentiment")
        
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            output_path = os.path.join(output_dir, f"dravidian_telugu_{split}.csv")
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {split} split to {output_path} (Shape: {df.shape})")
            
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    raw_data_dir = os.path.join("data", "raw")
    download_dravidian_codemix(raw_data_dir)
```

### 6. Explanation
The `download_data.py` script utilizes Hugging Face's `datasets` library to pull a community-curated version of the DravidianCodeMix Telugu dataset. It converts the Arrow dataset to a Pandas DataFrame and saves it as CSV in our `data/raw/` directory. This ensures that regardless of internet connectivity later on, we have a static version of the dataset to perform reproducible EDA and cleaning.

### 7. Common Mistakes
- **Not Checking Licenses:** Using proprietary data for a GitHub portfolio project can lead to takedowns. Always prioritize Creative Commons or Academic licenses.
- **Ignoring Data Formats:** Pulling raw JSON from Twitter APIs usually contains massive amounts of metadata. Converting immediately to clean CSV/Parquet saves hours of parsing later.

### 8. Improvements
- **Data Version Control (DVC):** As datasets grow, we should integrate DVC to track changes to `data/raw/` and `data/processed/` instead of relying entirely on Git, which chokes on large CSVs.

### 9. Research References
- *Chakravarthi, B. R., et al. (2020).* "Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text." (Methodology applies to the Telugu subset).

### 10. Next Step
Proceed to **Task 1.4: Repository Setup & Reproducibility Boilerplate**. We will execute the Python script created here, set up the official virtual environment, configure TensorBoard, and establish our random seed utilities.
