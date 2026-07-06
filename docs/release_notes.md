# TriMixGen v1.0 Release Notes

We are thrilled to announce the v1.0 Release of **TriMixGen** - an advanced Curriculum-Trained Natural Code-Mixing Agent.

## Core Features
1. **Zero-Shot Style Steering**: Control the Sentiment (Positive/Negative/Neutral), English Lexical Density (High/Low), and Formality of the generated code-mixing on the fly using instruction prompts.
2. **Curriculum Learning Pipeline**: TriMixGen was trained in three phases: 
   - **GEN_001**: Semantics Alignment
   - **GEN_002**: Colloquial Code-Mixing Adaptation
   - **GEN_003**: Multi-Attribute Prompt Control
3. **Catastrophic Forgetting Prevention**: Embedded dual-replay mechanisms during training ensure the model seamlessly adheres to native-Telugu grammar rules while dynamically mixing English vocabulary.
4. **FastAPI Inference Server**: A production-ready backend utilizing lazy-loaded Singletons to map `google/mt5-small` LoRA adapters and `IndicBERT` LID components into memory for massive latency reduction.
5. **Modern Web UI**: A highly responsive Vite+React SPA, featuring Code-Mixing Index (CMI) visualizations, dynamic Recharts token distributions, and a zero-latency Prompt Gallery.

## Deployment Notes
- `USE_MOCK_LID=false` must be set in production to enforce strict initialization of the IndicBERT model.
- Requires 4GB+ RAM (8GB+ if operating entirely on CPU). CUDA and MPS are supported natively.
