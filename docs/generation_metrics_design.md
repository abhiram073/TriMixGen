# Generation Metrics Design: Code-Mixed Evaluation

This document details the mathematical intuition behind the metrics implemented for TriMixGen and provides a final recommendation for research paper reporting.

## 1. N-Gram Overlap Metrics

### BLEU (Bilingual Evaluation Understudy)
*   **Intuition:** Counts how many n-grams (1 to 4 words) in the generated text exactly match n-grams in the reference text. Applies a brevity penalty.
*   **What it measures:** Lexical overlap and exact translation precision.
*   **Why it is useful:** Fast, industry standard, allows comparison with baseline papers.
*   **Where it fails:** Cannot understand synonyms. "Big" and "Large" score a 0%. In code-mixing, "movie" (EN) and "cinema" (loan word) are treated as entirely wrong despite identical semantics.
*   **Complexity:** $O(N)$ where $N$ is sequence length.
*   **Interpretation:** 0-100. >40 is highly fluent.
*   **Suitability for Code-Mixing:** Low. BLEU is overly rigid for colloquial generation where users fluidly swap vocabulary.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
*   **Intuition:** Finds the Longest Common Subsequence (LCS) between the generation and reference.
*   **What it measures:** Structural recall and sequence preservation.
*   **Why it is useful:** Good for determining if the overall "gist" of the sentence was retained.
*   **Where it fails:** Relies on exact lexical matching.
*   **Complexity:** $O(N \times M)$ where $N, M$ are lengths.
*   **Interpretation:** 0-100. Higher means better recall.
*   **Suitability for Code-Mixing:** Medium. Better than BLEU, but still struggles with synonyms.

## 2. Semantic Similarity Metrics

### BERTScore
*   **Intuition:** Uses contextual embeddings to compute cosine similarity between every generated word and reference word.
*   **What it measures:** Deep semantic equivalence.
*   **Why it is useful:** Understands synonyms and cross-lingual meaning.
*   **Where it fails:** Computationally expensive; requires an active LLM.
*   **Complexity:** $O(N^2 \times D)$ where $D$ is embedding dimension.
*   **Interpretation:** 0.0 - 1.0. >0.9 indicates high semantic equivalence.
*   **Suitability for Code-Mixing:** Extremely High. It is robust to lexical substitution.

## 3. Information Theoretic Metrics

### Perplexity (PPL)
*   **Intuition:** Measures how "surprised" a language model is by the text. (Exponentiated cross-entropy).
*   **What it measures:** Fluency and grammatical naturalness.
*   **Where it fails:** Highly model-dependent.
*   **Complexity:** Requires a full forward pass of the model per sequence.
*   **Interpretation:** Lower is better.
*   **Suitability for Code-Mixing:** Medium. Useful to ensure the model isn't outputting gibberish, but doesn't measure translation accuracy.

## 4. Diversity Metrics

### Distinct-1 / Distinct-2
*   **Intuition:** Ratio of unique unigrams/bigrams to total generated n-grams.
*   **What it measures:** Vocabulary diversity.
*   **Why it is useful:** Detects "safe response" collapse (e.g. model always replying "ok").
*   **Interpretation:** 0.0 - 1.0. Higher is better.
*   **Suitability:** High. Code-mixing models are prone to collapsing into safe monolingual phrases.

### Self-BLEU
*   **Intuition:** Computes BLEU of every generated sentence against all other generated sentences.
*   **What it measures:** Structural/Syntactic diversity.
*   **Interpretation:** Lower is better (we want less overlap between generated sentences).
*   **Suitability:** High.

## 5. Code-Mixing Index (CMI)
*   **Intuition:** $CMI = 100 \times [1 - \frac{max(w_i)}{N - u}]$ where $w_i$ is word count per language, $N$ is total words, $u$ is language-independent tokens.
*   **What it measures:** The degree of interleaving between English and Telugu.
*   **Why it is useful:** Mathematically proves the model is code-switching.
*   **Where it fails:** Fails if the underlying LID (Language Identification) model is inaccurate.
*   **Complexity:** $O(N)$.
*   **Interpretation:** 0-100. >30 usually indicates natural conversational code-mixing.
*   **Suitability:** Essential.

## 6. Research Paper Recommendations

*   **Primary Metrics (Research Paper):**
    *   **BERTScore:** To prove the semantic intent of the generation is preserved despite vocabulary swapping.
    *   **CMI:** To prove the generation successfully bridged the linguistic gap.
    *   **Human Evaluation:** The ultimate gold standard for naturalness.
*   **Secondary Metrics:**
    *   **BLEU / ROUGE:** Included solely for baseline comparisons against legacy papers, but explicitly caveated as inadequate for colloquial generation.
    *   **Distinct-1/2:** To prove the curriculum learning did not cause vocabulary collapse.
*   **Deployment Monitoring:**
    *   **Perplexity / CMI:** Can be tracked live on user interactions to detect model drift without requiring ground-truth references.
