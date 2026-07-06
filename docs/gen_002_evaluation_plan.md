# GEN_002: Evaluation Plan & Error Analysis

## 1. Evaluation Protocol

The evaluation pipeline for GEN_002 builds heavily upon the modular `GenerationMetrics` infrastructure established in GEN_001. We will evaluate the model's generated conversational text against the HOLD-Telugu references using the following comprehensive metric suite:

### Core Metrics
* **BLEU**: Measures precise n-gram overlap. While traditionally weak for conversational AI due to lexical diversity, it serves as a baseline for structural mapping.
* **ROUGE-L**: Evaluates the longest common subsequence, capturing semantic ordering and conversational flow.
* **BERTScore**: Computes deep contextual embeddings (using a multilingual encoder) to measure semantic similarity, highly crucial since code-mixed text can express the same intent using vast permutations of English and Telugu tokens.
* **Perplexity**: Derived directly from the language modeling loss, indicating the model's confidence in its own code-mixed probability distribution.

### Diversity Metrics
* **Distinct-1 & Distinct-2**: Measures the ratio of unique unigrams and bigrams, ensuring the model isn't memorizing generic, safe responses (e.g., "nenu fine").
* **Self-BLEU**: Quantifies the overlap *between* generated responses. A high Self-BLEU indicates severe mode collapse (the model is generating the same response for different prompts).

### Code Mixing Index (CMI)
GEN_002 fully activates our **Code Mixing Index (CMI)** evaluation.
1. We will route the generated predictions through our production **IndicBERT** LID (Language Identification) model.
2. The LID model will assign token-level language tags (e.g., `["TE", "EN", "EN", "TE"]`).
3. These tags will be passed into `GenerationMetrics.compute_cmi()` to calculate sentence-level CMI, average dataset CMI, and CMI variance.

---

## 2. Qualitative Analysis

To complement the mathematical metrics, the evaluation pipeline will automatically sample and generate three distinct groups of examples for human review:
1. **Best Generations**: Samples with the highest semantic overlap (BERTScore) and optimal CMI.
2. **Representative Generations**: Samples located at the median of the evaluation distribution.
3. **Failure Cases**: Samples that scored the lowest across metrics or exhibited severe errors (like monolingual collapse).

A short automated explanation will accompany each sampled group to provide context on why it was categorized as such.

---

## 3. Expected Error Analysis

Evaluating code-mixed generation requires classifying unique failure modes. We expect the following errors, which must be tracked during validation:

1. **Monolingual Collapse**: The model generates perfect sentences but defaults entirely to English, ignoring the code-mixed directive.
2. **Excessive English / Telugu**: The model skews too far into one language (e.g., generating 95% Telugu and 5% English), causing the CMI to plummet toward zero.
3. **Script Hallucination**: Despite the GEN_001 pre-training, noisy HOLD-Telugu data might cause the model to suddenly hallucinate native Telugu script (తెలుగు) instead of adhering to Romanization.
4. **Unnatural Switching**: Grammatically incorrect transition boundaries. (e.g., splitting a base noun and its English suffix incorrectly).
5. **Repeated Phrases**: The model gets stuck in a decoding loop (e.g., "nenu chestunnanu chestunnanu chestunnanu").
6. **Romanization Inconsistency**: The model spells the same Telugu word multiple different ways within the same sentence (e.g., "chala" vs "chaala").

---

## 3. Success Criteria

To declare GEN_002 a success and advance to GEN_003, the fine-tuned model must achieve the following quantitative thresholds on the HOLD-Telugu test set:

1. **BLEU Score**: `> 12.0` (Note: Conversational BLEU is typically low; 12.0 indicates reasonable structural alignment).
2. **BERTScore (F1)**: `> 0.85` (Indicating strong semantic preservation).
3. **Average CMI**: Between `15.0` and `30.0` (Indicating a healthy, natural balance of code-mixing, avoiding monolingual extremes).
4. **Self-BLEU**: `< 0.35` (Ensuring conversational diversity).
5. **Script Consistency**: `100%` Romanized script (0% native Telugu script generated).
