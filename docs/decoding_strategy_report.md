# Decoding Strategy Report: Code-Mixed Generation

This report analyzes the mathematical intuition behind different generation decoding strategies and their specific implications for colloquial Telugu-English code-mixed text generation.

## 1. Mathematical Intuition

### Greedy Search
*   **Intuition:** At every timestep $t$, the model simply selects the token with the highest absolute probability: $w_t = \arg\max P(w | w_{<t})$.
*   **Behavior:** It is highly deterministic. It often gets stuck in repetitive loops because it lacks the foresight to explore lower-probability tokens that might lead to a much higher overall sequence probability later on.

### Beam Search
*   **Intuition:** Instead of keeping just 1 path (greedy), it maintains $k$ active paths (beams) at every step. It calculates the joint probability of the entire sequence and keeps the top $k$ most likely sequences.
*   **Behavior:** It guarantees finding a sequence with a much higher overall likelihood than greedy search. However, highly probable sequences in LLMs tend to be safe, generic, and "boring".

### Top-k Sampling
*   **Intuition:** Instead of maximizing probability, the model samples from the distribution. However, to prevent sampling total gibberish from the long tail of the distribution, it first truncates the vocabulary to only the Top $k$ most likely tokens, redistributes the probability mass, and samples from those $k$.
*   **Behavior:** Introduces controlled randomness (diversity).

### Top-p (Nucleus) Sampling
*   **Intuition:** Instead of a fixed number $k$, it sorts tokens by probability and keeps adding tokens to the pool until their cumulative probability exceeds $p$ (e.g., $0.92$).
*   **Behavior:** Highly dynamic. If the model is very confident (steep distribution), it might only sample from 2 tokens. If it is uncertain (flat distribution), it might sample from 100 tokens. This is widely considered the state-of-the-art for natural language generation.

### Temperature Scaling
*   **Intuition:** Before applying the softmax function to the raw logits to convert them to probabilities, the logits are divided by a temperature parameter $T$.
    *   $T < 1.0$: Sharpens the distribution (makes the model more confident/greedy).
    *   $T > 1.0$: Flattens the distribution (makes the model more random/creative).

## 2. Trade-offs in Code-Mixed Generation

Code-mixing is inherently a high-entropy phenomenon. There are mathematically infinite ways to interleave English and Telugu (e.g., "Cinema super undi", "Movie chala bagundi", "Film keka").

| Strategy | Fluency | Code-Mixing Diversity (CMI) | Hallucination Risk |
| :--- | :--- | :--- | :--- |
| **Greedy** | Very High | Very Low (repetitive) | Low |
| **Beam** | Very High | Low (prefers safe monolingual text) | Low |
| **Top-k** | Medium | High | Medium |
| **Top-p (Nucleus)** | High | **Very High** | Medium-Low |

**The Code-Mixing Problem with Beam Search:**
Beam search rigorously optimizes for mathematical likelihood. Because pure English and pure Telugu are vastly more statistically common in the pre-training data than Code-Mixed variants, Beam Search often "collapses" into monolingual text. It avoids the statistical penalty of switching languages.

**The Code-Mixing Advantage of Nucleus Sampling:**
Nucleus sampling (Top-p) allows the model to take slightly lower-probability paths (like suddenly switching to an English noun in the middle of a Telugu sentence) without wandering into total grammatical noise. It perfectly mimics the natural unpredictability of human code-switching.

## 3. Production Recommendation
For TriMixGen's production deployment, **Nucleus Sampling (Top-p = 0.92, Temperature = 0.8)** is the optimal default strategy. It will produce highly natural, diverse, and grammatically sound code-mixed generations without collapsing into boring monolingual loops. Beam Search should be retained solely as a fallback for strict instruction-following tasks where creativity is not desired.
