# TriMixGen Backend API Reference

Base URL: `http://localhost:8000/api/v1`

## 1. POST `/generate`
Generates code-mixed Telugu-English text based on a given prompt and style constraints.

**Request Body**:
```json
{
    "prompt": "Write a positive review of the new movie",
    "style": "positive",          // Options: positive, negative, neutral, formal, informal
    "english_usage": "high",      // Options: high, low, auto
    "temperature": 0.8
}
```

**Response**:
```json
{
    "generated_text": "Movie chala bagundi, it was an amazing experience!",
    "rendered_prompt": "Write a positive Telugu-English review. Use a high amount of English vocabulary.",
    "language_tags": ["EN", "TE", "TE", "EN", "EN", "EN", "EN"],
    "cmi": 42.85,
    "latency": 0.8523,
    "inference_config": {"temperature": 0.8},
    "token_count": 7
}
```

## 2. POST `/tag`
Tags an input sequence with word-level Language Identification (LID) and computes its CMI.

**Request Body**:
```json
{
    "text": "Idi chala bagundi andi"
}
```

**Response**:
```json
{
    "tokens": ["Idi", "chala", "bagundi", "andi"],
    "labels": ["TE", "TE", "TE", "TE"],
    "confidence": 1.0,
    "cmi": 0.0
}
```

## 3. GET `/health`
Returns the status of the backend services and loaded model components.

**Response**:
```json
{
    "status": "healthy",
    "generator_loaded": true,
    "indicbert_loaded": true,
    "tokenizer_loaded": true,
    "uptime": 142.5,
    "device": "cpu",
    "model_versions": { ... }
}
```

## 4. GET `/model-info`
Returns diagnostic metadata regarding the active checkpoint.
