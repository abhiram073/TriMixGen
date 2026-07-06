# GEN_003: Style-Controlled Generation Design

## 1. Research Objective

### Why GEN_003 is Needed
The final stage of the curriculum learning pipeline, **GEN_003**, evolves the model from a passive conversational respondent into an **active, style-controlled generator**. While GEN_002 successfully taught the model the probability distribution of natural social media code-mixing, it lacks explicit semantic steering. GEN_003 introduces conditioning constraints, allowing users to direct the emotional tone (sentiment), formality, and lexical density (English vs. Telugu bias) of the generated output.

### The Curriculum Learning Ordering
Style control is a high-level semantic constraint. If attempted during GEN_001 (when the model is still learning Romanized Telugu syntax) or GEN_002 (when it is learning the chaotic boundaries of code-switching), the optimization landscape becomes too complex, leading to mode collapse or severe hallucinations. By introducing style control only *after* the foundational grammar and code-mixing distributions are stabilized, the model can dedicate its parameter updates purely to mapping control vectors (prompts) to established linguistic structures.

---

## 2. Dataset Strategy

### The Telugu Sentiment Dataset
GEN_003 utilizes the `Telugu-Sentiment` dataset (which contains `train`, `validation`, and `test` splits). This dataset consists of code-mixed reviews and social media comments categorized by sentiment (Positive, Negative, Neutral).

### Preprocessing & Quality Control
* **Duplicate Removal**: We will implement exact string matching and MinHash LSH to drop redundant samples.
* **Length Filtering**: Dropping extremely short (<3 words) or extremely long (>100 words) samples to maintain consistent generative targets.
* **Sentiment Balancing**: Highly skewed datasets cause unconditional bias (e.g., generating positive text even when prompted for negative). We will aggressively downsample the majority classes to achieve a perfectly balanced 1:1:1 ratio between Positive, Negative, and Neutral samples in the training split.

### Train / Validation / Test Splitting
We will ingest the raw `Telugu-Sentiment` splits, process them, and output standardized `train.parquet`, `valid.parquet`, and `test.parquet` files. These splits will remain statically frozen to allow perfectly reproducible evaluation, mirroring the rigor of GEN_001 and GEN_002.

---

## 3. Fine-Tuning Strategy

### Continuation from GEN_002
GEN_003 will initialize by merging the best LoRA adapter from GEN_002 into the base `mt5-small` weights. This gives the model its fully realized conversational code-mixing foundation. We will then inject a **new, fresh LoRA adapter** to learn the style-conditioning mappings.

### Catastrophic Forgetting Prevention & Replay Strategy
To prevent the model from overfitting to sentiment data and forgetting how to act as a general instruction-following assistant or conversational respondent, we will employ a **Dual-Replay Strategy**:
* **10% GEN_001 Replay**: Samples from the Alpaca dataset to preserve instructional logic.
* **10% GEN_002 Replay**: Samples from the HOLD-Telugu dataset to preserve unconstrained conversational fluidity.
* **80% GEN_003 Data**: The new style-controlled sentiment dataset.

### Learning-Rate Scheduling
* **Peak LR**: `1e-4` (Lower than previous stages, as we are applying highly targeted semantic conditioning rather than broad language modeling).
* **Schedule**: Cosine decay with a 10% warmup to gently align the new LoRA weights to the embedded conversational representations.

---

## 4. Deployment Export

Upon successful completion of the GEN_003 stage and evaluation benchmarks, the model must be prepared for backend integration in the TriMixGen web application. We will execute an automated deployment pipeline to export the following unified artifacts:
1. **Merged Production Model**: Base `google/mt5-small` merged with both GEN_001 and GEN_002 adapters, and finally with the new GEN_003 LoRA adapter.
2. **LoRA Adapter**: Independent serialization of the final GEN_003 adapters if dynamic loading is preferred.
3. **Generation Configuration**: Freezing optimal decoding strategies (top-p, temperature, repetition penalty).
4. **Prompt Templates**: Serialized JSON/YAML containing the multi-attribute templates designed for GEN_003.
5. **Tokenizer Configuration**: Ensuring all special tokens are accurately exported.
6. **Inference Configuration**: Standardizing batched inference endpoints.

A **Deployment Manifest** will be generated documenting the exact MD5 hashes and file paths for all exported artifacts to ensure a seamless transition to the production engineering team.
