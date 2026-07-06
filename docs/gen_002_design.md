# GEN_002: Natural Code-Mixing Design Document

## 1. Research Objective

### Why GEN_002 Exists
**GEN_002** represents the second stage of our curriculum learning pipeline for TriMixGen. While GEN_001 focused on teaching the base `google/mt5-small` model the foundational grammar, syntax, and vocabulary of Romanized Telugu, GEN_002 aims to elevate this by introducing natural, conversational code-mixing. The goal is to transition the model from generating formal or monotonic Romanized Telugu to producing fluid, colloquial Telugu-English (Hinglish/Tenglish style) text that mirrors actual social media communication.

### The Necessity of a Second Stage
Curriculum learning dictates that neural models learn complex behaviors better when introduced sequentially. Natural code-mixing is highly complex, requiring the model to maintain syntactic coherence while dynamically switching between languages (Telugu morphology with English stems, and vice-versa). Attempting to learn Romanized script representation, Telugu grammar, and complex language switching simultaneously leads to poor convergence and high rates of script hallucination. Stage 1 (GEN_001) isolated the Romanization task; Stage 2 (GEN_002) isolates the code-mixing distribution task.

### Why Direct Training on HOLD-Telugu is Insufficient
Directly fine-tuning `mT5` on HOLD-Telugu without the GEN_001 Alpaca pre-training phase typically results in monolingual collapse (the model favors generating purely in English) or severe script hallucination (reverting to native Telugu script). The HOLD-Telugu dataset is highly diverse and noisy; without a strong Romanized semantic foundation established in GEN_001, the model struggles to map the code-mixed semantic intent to the correct Romanized tokens.

---

## 2. Fine-Tuning Strategy

### Continuing from GEN_001
GEN_002 will initialize its weights using the final merged checkpoint from GEN_001. This ensures that the model retains its strong understanding of Romanized Telugu semantics. 

### LoRA Reuse
We will re-initialize a fresh set of LoRA adapters (Rank=8, Alpha=16) for GEN_002 on top of the merged GEN_001 base model. This isolates the code-mixing adaptation to the new adapters, allowing us to easily evaluate the specific delta introduced by the conversational HOLD-Telugu dataset.

### Learning-Rate Schedule
We will utilize a **cosine decay with warmup** learning rate schedule. 
* **Peak LR**: `2e-4` (slightly lower than GEN_001 to prevent destructive updates to the embedded Romanized knowledge).
* **Warmup Ratio**: `0.1` (to gently introduce the noisy gradients of social media data).

### Catastrophic Forgetting Prevention and Monitoring
To prevent the model from forgetting the formal structural rules learned in GEN_001, we will implement **Dataset Replay**. Approximately 15% of the GEN_002 training batches will contain high-quality samples from the GEN_001 Alpaca dataset.

Additionally, after every training epoch, we will evaluate the model on both the **GEN_001 validation set** and the **GEN_002 validation set**. We will explicitly track whether the adaptation to natural code-mixing degrades its ability to generate pure Romanized Telugu. This dual-validation analysis will be included in the final experiment report.

### Early Stopping
Training will be heavily regularized using Early Stopping based on the validation loss and validation CMI variance, with a patience of 3 epochs. Overfitting on HOLD-Telugu often manifests as generating the exact conversational quirks of the dataset rather than generalizing the code-mixing behavior.

---

## 3. Code-Mixing Analysis

### Expected CMI Distribution
The HOLD-Telugu dataset contains highly variable code-mixing. We expect the target CMI (Code Mixing Index) distribution to center around 15-30 for typical sentences. The model must learn to navigate this distribution, avoiding the extremes of purely monolingual outputs (CMI = 0).

### Code Mixing Index Tracking
We will rigorously measure and log the model's Code Mixing Index across the entire pipeline:
1. **Baseline**: CMI of the GEN_001 model before GEN_002 training.
2. **Epoch-wise**: CMI evaluated after every training epoch.
3. **Final Model**: CMI of the fully fine-tuned GEN_002 model.

We will visualize the progression of the CMI across training to ensure the model successfully interpolates from formal Romanized Telugu to the conversational code-mixed distribution.

### Language Switching Behavior
The model must learn natural transition points (e.g., inter-sentential and intra-sentential switching). Natural switches in Telugu-English often occur at noun phrases, verbs, or conjunctions (e.g., "nenu *work* chestunnanu" -> "I am working").

### Romanized Telugu Morphology
A key behavior to analyze is the attachment of Telugu morphological suffixes to English stems. For example, "car-lu" (cars), "update-chesa" (updated). The model must accurately predict these hybrid sub-word tokens without breaking the grammatical rules of Telugu.

### English Insertion Patterns
English insertions are typically nouns, technical terms, or common social media phrases. The model must learn that structural grammar usually remains Telugu, while the vocabulary dynamically samples from English based on context.
