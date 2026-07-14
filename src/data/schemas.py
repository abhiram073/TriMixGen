import os
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Unified Dataset Schema
# ---------------------------------------------------------

class GenerationDatasetRow(BaseModel):
    id: str = Field(..., description="Unique identifier for the row")
    language_combination: str = Field(..., description="e.g., HIN_BEN_ENG")
    sentence_1: str = Field(..., description="Monolingual sentence in language 1 (e.g. Hindi)")
    sentence_2: str = Field(..., description="Monolingual sentence in language 2 (e.g. Bengali/Gujarati)")
    sentence_3: str = Field(..., description="Monolingual sentence in language 3 (e.g. English)")
    target: str = Field(..., description="The expected trilingual code-mixed output")
    
class LIDDatasetRow(BaseModel):
    id: str = Field(..., description="Unique identifier for the row")
    language_combination: str = Field(..., description="e.g., HIN_ENG")
    tokens: List[str] = Field(..., description="List of individual tokens")
    labels: List[str] = Field(..., description="Corresponding language labels (HIN, BEN, GUJ, ENG, OTHER)")

# ---------------------------------------------------------
# Registry Schemas
# ---------------------------------------------------------

class DatasetMetadata(BaseModel):
    name: str
    version: str
    source: Literal["huggingface", "local", "url"]
    download_url: str
    license: str
    tasks: List[str]  # e.g., ["generation", "language_identification"]
    supported_languages: List[str]
    supported_combinations: List[str]
    script: str = "Roman"
    local_storage_path: str
    requires_manual_download: bool = False
    checksum: Optional[str] = None

class SplitConfig(BaseModel):
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1
    random_seed: int = 42

class DatasetRegistryConfig(BaseModel):
    datasets: List[DatasetMetadata]
    splits: SplitConfig
