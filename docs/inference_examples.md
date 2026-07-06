# Inference Examples: TriMixGen Generation

This document illustrates expected outputs from the `inference.py` module across different modes and prompting strategies.

## Example 1: Translation (English to Code-Mixed)
**Code:**
```python
output = generator.generate(
    instruction="Translate the following to colloquial Telugu-English:",
    context="The weather is very beautiful today, let's go outside.",
    template="english_to_codemixed"
)
```
**Expected Output Object:**
```json
{
    "generated_text": "eeroju weather chala beautiful ga undi, bayatiki veldam.",
    "raw_prompt": "Instruction: Translate the following to colloquial Telugu-English:\nContext: The weather is very beautiful today, let's go outside.",
    "generation_time_sec": 1.45,
    "token_count": 14,
    "decoding_strategy": "nucleus_sampling",
    "generation_parameters": {"top_p": 0.92, "temperature": 0.8}
}
```

## Example 2: Conversation Generation
**Code:**
```python
output = generator.generate(
    instruction="Reply to the user naturally:",
    context="User: Cinema ki veldama bro?",
    template="conversation"
)
```
**Expected Output Object:**
```json
{
    "generated_text": "ha bro veldam, ye movie ki veldam?",
    "raw_prompt": "Instruction: Reply to the user naturally:\nContext: User: Cinema ki veldama bro?",
    "generation_time_sec": 0.92,
    "token_count": 9,
    "decoding_strategy": "nucleus_sampling",
    "generation_parameters": {"top_p": 0.92, "temperature": 0.8}
}
```

## Example 3: Batch Inference (Deployment API)
**Code:**
```python
instructions = ["Translate to code-mixed:", "Translate to code-mixed:"]
contexts = ["I am eating food.", "I am reading a book."]
outputs = generator.generate_batch(
    instructions=instructions,
    contexts=contexts,
    template="english_to_codemixed"
)
```
**Expected Output Object:**
List of standard inference objects representing:
1. "nenu food tintunnanu."
2. "nenu oka book chaduvutunnanu."
