# Phase 1: Literature Review - Code-Mixed NLP

### 1. Objective
Review the State-of-the-Art (SOTA) research models and architectures in code-mixed Natural Language Processing, specifically focusing on Language Identification (LID) and Text Generation. Evaluate the trade-offs of existing approaches.

### 2. Theory & Paper Summaries
We focus on 6 seminal and recent papers bridging the gap in code-mixed and Indic NLP:

**1. "A Survey of Code-mixed Data and Models in NLP" (Sitaram et al., 2019)**
* **Summary:** Provides a comprehensive overview of how code-mixing behaves linguistically and maps out the datasets available. 
* **Pros/Cons:** Great foundational knowledge; however, most models discussed are pre-transformer era (CRFs, standard LSTMs).

**2. "Language Identification in Code-Mixed Text using Word Embeddings" (Chandu et al., 2018)**
* **Summary:** Compares character-level and word-level CNNs and LSTMs for word-level LID in Indian languages.
* **Pros/Cons:** Proves that character-level information (or subwords) is crucial for Indic languages due to Romanized spelling variations, but doesn't leverage deep contextual embeddings like BERT.

**3. "GLUECoS: An Evaluation Benchmark for Code-Switched NLP" (Khanuja et al., 2020)**
* **Summary:** Introduces a benchmark suite containing LID, POS tagging, and NER for code-mixed English-Hindi and English-Spanish.
* **Pros/Cons:** Establishes standard evaluation practices. TriMixGen will borrow heavily from their evaluation metrics.

**4. "IndicBERT: A Multilingual ALBERT for Indic Languages" (Kakwani et al., 2020)**
* **Summary:** Evaluates a lightweight ALBERT model trained specifically on Indian languages.
* **Pros/Cons:** Highly efficient (good for our laptop constraints) but primarily trained on native scripts, not Romanized code-mixed text. It often struggles with English-Telugu transliteration.

**5. "GCM: A Toolkit for Generating Code-Mixed Text" (Gupta et al., 2021)**
* **Summary:** Presents a theoretical sequence-to-sequence model approach to generate syntactically valid code-mixed text using pointer-generator networks.
* **Pros/Cons:** Excellent theoretical constraints for generation, but lacks the raw generative power of modern Large Language Models (LLMs) like Llama or BLOOM.

**6. "mBART and mT5: Multilingual Sequence-to-Sequence Models" (Xue et al., 2021)**
* **Summary:** Introduces mT5, a massively multilingual model covering 101 languages, including Telugu.
* **Pros/Cons:** Extremely powerful for generation tasks. However, its massive parameter size requires Parameter-Efficient Fine-Tuning (PEFT) like LoRA to run on our constrained hardware.

### 3. Implementation Plan
Based on this review, our TriMixGen architecture choices are:
- **LID:** We will baseline with CRF/BiLSTM but aim for a fine-tuned `XLM-R` or `mBERT` due to their proven subword handling capabilities on Romanized text.
- **Generation:** We will utilize `mT5` or a small instruct-tuned `Llama 3 (8B)` (if hardware allows via QLoRA), as they offer the best balance of multilingual capability and generation fluency.

### 4. Folder Structure
```text
TriMixGen/
└── docs/
    ├── problem_understanding.md
    └── literature_review.md  <-- (This file)
```

### 5. Code
*Since this is a documentation task, here is a pseudo-code block demonstrating how our literature review guides our model selection strategy.*

```python
# Strategy for TriMixGen model selection based on literature
from transformers import AutoModelForTokenClassification, AutoModelForSeq2SeqLM

def load_models_based_on_literature():
    # For Language Identification: XLM-R is chosen for strong cross-lingual subword features
    lid_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base")
    
    # For Generation: mT5 is chosen for its strong sequence-to-sequence multilingual generation
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    
    return lid_model, gen_model
```

### 6. Explanation
The pseudo-code illustrates our ultimate modeling choices. By choosing `xlm-roberta-base` for LID, we leverage a model that natively understands multiple languages and handles subword tokenization well, mitigating the transliteration issues highlighted by Chandu et al. (2018). By selecting `mt5-small`, we get a generation model that supports Telugu natively and can be fine-tuned via LoRA.

### 7. Common Mistakes
- **Ignoring Model Vocabulary:** A common mistake in code-mixing research is choosing an English-centric LLM. If the tokenizer doesn't support Indic scripts or frequently breaks down Romanized Telugu into individual characters, the model's performance will crash.
- **Underestimating Computational Cost:** Attempting full fine-tuning on a sequence-to-sequence model like mT5-base or larger on a laptop will lead to Out of Memory (OOM) errors.

### 8. Improvements
- **Continual Pre-training:** To improve upon the SOTA, we could take XLM-R and perform domain adaptation (Masked Language Modeling) on a large corpus of unlabeled code-mixed Telugu-English text before fine-tuning it for LID.

### 9. Research References
All papers referenced in Section 2 serve as the core literature foundation for this project. 
- Links to standard papers can be found via Google Scholar or the ACL Anthology (e.g., *GLUECoS*).

### 10. Next Step
Proceed to **Task 1.3: Dataset Collection & Benchmarking**. We will transition from theory to practice by identifying and downloading the raw datasets needed to train these chosen architectures.
