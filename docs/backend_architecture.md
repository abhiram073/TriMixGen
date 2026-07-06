# TriMixGen Backend Architecture

The TriMixGen Backend is a high-performance RESTful API built on **FastAPI**. It serves as the production inference layer, unifying the TriMixGen curriculum pipeline (`mT5-small` + `GEN_003 LoRA`) with Code-Mixing Index metrics and IndicBERT LID predictions.

## System Diagram
```mermaid
graph TD;
    Client[Web Client] --> API[FastAPI Endpoints]
    
    API --> Generate[/generate]
    API --> Tag[/tag]
    API --> Health[/health]
    
    Generate --> GenService[GenerationService]
    GenService --> TriMixGenerator[TriMixGen Generator (mT5+LoRA)]
    
    Generate --> Post[PostprocessService]
    Generate --> LID[LIDService]
    
    LID --> IndicBERT[IndicBERT Token Classifier]
    LID --> Metrics[MetricsService]
    
    Tag --> LID
    
    Metrics --> CMI[Code Mixing Index]
```

## Core Components
1. **app.py**: Defines the FastAPI application, mounts CORS middleware, error handlers, and triggers singleton loading on startup.
2. **api.py**: Routes HTTP requests to the respective services.
3. **schemas.py**: Utilizes Pydantic to validate input data (e.g. bounding prompt length, checking types) and standardizing output JSONs.
4. **Services Layer**: 
    - `GenerationService`: Caches the massive `mt5` model in memory. Translates UI style options to raw instruction strings.
    - `LIDService`: Handles sequence labeling.
    - `MetricsService`: Translates language tag probabilities into the CMI score.
    - `PostprocessService`: Cleans formatting artifacts.

## Design Decisions
- **Lazy Singleton Loading**: To avoid slow application boot loops, model instances are created as Singletons and instantiated in the `@app.on_event("startup")` hook.
- **Dynamic Device Placement**: `config.py` polls `torch.cuda` and `torch.backends.mps` to assign inference to the GPU automatically.
