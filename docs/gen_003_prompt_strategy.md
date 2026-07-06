# GEN_003: Controlled Generation Prompt Strategy

## 1. Overview
In GEN_003, the model transitions from predicting the next most likely token in an uncontrolled sequence to generating text bounded by specific semantic and stylistic constraints. To achieve this, we translate the static categorical labels of the Telugu-Sentiment dataset into natural language instruction prompts.

## 2. Sentiment Control
We will map the dataset labels (Positive, Negative, Neutral) to explicit generation instructions. This creates a hard conditioning signal for the model's emotional tone.

### Positive Prompts
Used when the target output reflects praise, happiness, or agreement.
* `Write a positive Telugu-English review.`
* `Generate an enthusiastic code-mixed response.`
* `Express agreement in Telugu-English.`

### Negative Prompts
Used when the target output reflects anger, disappointment, or disagreement.
* `Write a negative Telugu-English review.`
* `Generate a critical code-mixed comment.`
* `Express disappointment in Tenglish.`

### Neutral Prompts
Used when the target output is informative, objective, or non-opinionated.
* `Write a neutral Telugu-English review.`
* `Provide an objective code-mixed statement.`
* `Generate a factual response in Telugu-English.`

---

## 3. Style and Lexical Control

Beyond basic sentiment, we want to establish secondary control vectors. Since we do not have explicit labels for formality or English density in the raw dataset, we will programmatically infer these labels during the dataset preparation phase and dynamically append them to the sentiment prompts.

### Formality Control
We will classify the target sentences based on grammatical structure and vocabulary:
* **Formal**: Uses respectful Telugu markers (e.g., "andi", "garu", "meeruu") and complete sentences.
  * *Appended Prompt*: `Use a formal and respectful tone.`
* **Informal**: Uses casual markers (e.g., "ra", "mama", "nuvvu") and slang.
  * *Appended Prompt*: `Use a casual, conversational tone.`

### Lexical Density (Code-Mixing Axis)
We will utilize our IndicBERT LID (Language Identification) token tags to compute the Code Mixing Index (CMI) or raw English word ratio of the target sentence during preprocessing.
* **High English Usage**: Sentences where English tokens comprise > 50% of the non-punctuation tokens.
  * *Appended Prompt*: `Use a high amount of English vocabulary.`
* **Low English Usage**: Sentences where English tokens comprise < 20% of the non-punctuation tokens, heavily favoring Telugu roots.
  * *Appended Prompt*: `Use predominantly Telugu vocabulary.`

### Multi-Attribute Controlled Generation
We will explicitly construct composite prompts that combine multiple control vectors simultaneously. Each combination will be mapped independently to ensure the model learns intersecting boundaries. Examples include:
* `Write a positive Telugu-English review. Use a casual, conversational tone.` (Positive + Informal)
* `Generate an enthusiastic code-mixed response. Use a high amount of English vocabulary.` (Positive + High English Usage)
* `Write a negative Telugu-English review. Use a formal and respectful tone.` (Negative + Formal)
* `Provide an objective code-mixed statement. Use predominantly Telugu vocabulary.` (Neutral + Low English Usage)

---

## 4. Generalization Benchmark

To ensure the model does not simply overfit to the explicit training domains of the `Telugu-Sentiment` and `HOLD-Telugu` datasets, we will create a held-out evaluation prompt set covering diverse, unseen conversational domains:
1. Movie Reviews (`Write a positive Telugu-English movie review.`)
2. Restaurant Reviews (`Write a negative Telugu-English restaurant review.`)
3. Technology Discussions (`Discuss the new smartphone in Telugu-English.`)
4. Sports Commentary (`Write an enthusiastic code-mixed response about the cricket match.`)
5. College Conversations (`Use a casual tone to discuss college exams in Telugu-English.`)
6. Social Media Replies (`Reply casually in code-mixed Telugu to this tweet.`)

The final GEN_003 model will be evaluated against these prompts in a zero-shot capacity without further fine-tuning.
