from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from backend.config import settings
import re

class GenerateRequest(BaseModel):
    sentence_1: str = Field(..., description="Monolingual sentence in language 1 (e.g. Hindi)")
    sentence_2: str = Field(..., description="Monolingual sentence in language 2 (e.g. Bengali/Gujarati)")
    sentence_3: str = Field(..., description="Monolingual sentence in language 3 (e.g. English)")
    language_pair: str = Field(..., description="Requested language combination: HIN-BEN-ENG or HIN-GUJ-ENG")
    style: str = Field(default="neutral", description="Requested style: positive, negative, neutral, formal, informal.")
    english_usage: str = Field(default="auto", description="Requested English density: high, low, auto.")
    temperature: float = Field(default=0.8, ge=0.1, le=1.2)
    
    @field_validator("sentence_1", "sentence_2", "sentence_3")
    def validate_sentences(cls, v):
        if not v.strip():
            raise ValueError("Sentence cannot be empty.")
        if len(v) > settings.MAX_PROMPT_LENGTH:
            raise ValueError(f"Sentence exceeds maximum length of {settings.MAX_PROMPT_LENGTH} characters.")
        
        # Basic sanitization
        v = re.sub(r'[^\w\s.,!?\'"-]', '', v)
        return v.strip()

class GenerateResponse(BaseModel):
    generated_text: str
    rendered_prompt: str
    language_tags: List[str]
    cmi: float
    latency: float
    inference_config: Dict[str, float]
    token_count: int

class TagRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language_pair: str = Field(..., description="Requested language combination: HIN-BEN-ENG or HIN-GUJ-ENG")

class TagResponse(BaseModel):
    tokens: List[str]
    labels: List[str]
    confidence: Optional[float] = 1.0 # Mocked for now until probabilities are surfaced
    cmi: float

class HealthResponse(BaseModel):
    status: str
    generator_loaded: bool
    indicbert_loaded: bool
    tokenizer_loaded: bool
    lid_mock_mode: bool
    uptime: float
    device: str
    model_versions: Dict[str, str]

class ModelInfoResponse(BaseModel):
    model_version: str
    tokenizer: str
    parameter_count: int
    generation_configuration: Dict
    deployment_metadata: Dict
