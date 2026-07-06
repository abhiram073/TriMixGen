# GEN_003: Evaluation Plan and Error Analysis

## 1. Evaluation Protocol

To rigorously evaluate style-controlled generation, our evaluation suite must be expanded. We will continue to measure the foundational metrics from GEN_001 and GEN_002, while introducing new semantic classification metrics.

### Foundational Generation Metrics
* **BLEU, ROUGE, BERTScore**: To measure semantic and structural overlap with the reference text.
* **Perplexity**: To measure generation confidence.
* **Distinct-1, Distinct-2, Self-BLEU**: To verify that style conditioning does not lead to generic, repetitive template responses (mode collapse).
* **Code Mixing Index (CMI)**: Utilizing our IndicBERT LID model to ensure the output remains naturally code-mixed despite the sentiment constraints.

### Style and Prompt Adherence Metrics (Independent)
Because GEN_003 is explicitly controlled via prompts, we must measure how well the generated text adheres to those instructions. These must be **reported independently** and not aggregated into a single score:
* **Sentiment Accuracy**: Evaluated using a pre-trained sentiment classifier.
* **Formality Adherence**: Heuristic checking for formal ("andi", "garu") vs informal ("ra", "mama") markers based on the prompt.
* **English Usage Adherence**: Measuring the generated English token ratio vs. target prompt requirements.
* **Code Mixing Index (CMI) Adherence**: Ensuring the CMI shifts based on the specific Prompt request.
* **Prompt Adherence**: Overall compliance with the core instruction intent.

---

## 2. Error Analysis

In a controlled generation setting, the failure modes become highly specific. We will track the following expected errors during validation:

1. **Sentiment Drift**: The model generates a semantically correct sentence, but the sentiment contradicts the prompt (e.g., generating "Idi chala bagundi" (This is very good) when prompted for a *negative* review).
2. **Incorrect Emotional Tone**: The output technically satisfies the sentiment but uses unnatural phrasing (e.g., overly formal praise for a casual social media prompt).
3. **Code-Mixing Collapse**: The model satisfies the sentiment constraint but abandons code-mixing entirely, defaulting to 100% English or 100% Romanized Telugu.
4. **Repetition**: The model gets stuck repeating the exact words from the prompt instead of generating novel content.
5. **Script Hallucination**: The cognitive load of mapping sentiment to Romanized text causes the model to revert to native Telugu script (తెలుగు).
6. **Prompt Misunderstanding**: The model ignores the prompt entirely and acts as a passive conversational agent (reverting to GEN_002 behavior).
7. **Excessive English / Telugu**: Violating explicit lexical density prompts (e.g., generating 90% Telugu when asked for high English usage).

---

## 3. Success Criteria

To declare GEN_003 a success and validate the model for production deployment, the fine-tuned model must achieve the following rigorous thresholds on the test set:

1. **Sentiment Accuracy**: `> 85%` (The generated text must match the requested sentiment).
2. **Lexical Adherence**: `> 80%` compliance when prompted for high/low English usage.
3. **BERTScore (F1)**: `> 0.85` (Preserving the high semantic alignment of GEN_002).
4. **Average CMI**: Between `15.0` and `30.0` (Maintaining natural conversational boundaries).
5. **Script Consistency**: `100%` Romanized script (0% native Telugu script generated).
6. **Catastrophic Forgetting Limits**: The model must maintain a BERTScore `> 0.82` on both the GEN_001 and GEN_002 frozen evaluation sets to prove it hasn't forgotten its foundational behaviors.

---

## 4. Human Evaluation Protocol

Since automated metrics often fail to capture subtle linguistic nuances of code-mixed text, we will conduct a structured human evaluation. Evaluators will rate generated samples on a scale of 1–5 across six independent dimensions:

1. **Fluency**: (1 = Unreadable, 5 = Flawless natural flow)
2. **Naturalness**: (1 = Robotic/Machine-translated tone, 5 = Highly authentic social media tone)
3. **Sentiment Correctness**: (1 = Completely opposite sentiment, 5 = Perfect expression of the requested sentiment)
4. **Style Adherence**: (1 = Ignored formality/informality, 5 = Perfect execution of requested style)
5. **Code-Mixing Quality**: (1 = Jarring, abrupt language shifts, 5 = Seamless, native-like language blending)
6. **Grammar**: (1 = Completely broken grammar, 5 = Syntactically flawless within the bounds of Tenglish)

A detailed rubric will be provided to evaluators to ensure scoring consistency across reviewers.
