import time
from fastapi import APIRouter, HTTPException
from backend.schemas import (
    GenerateRequest, GenerateResponse, 
    TagRequest, TagResponse, 
    HealthResponse, ModelInfoResponse
)
from backend.services.generation_service import GenerationService
from backend.services.lid_service import LIDService
from backend.services.metrics_service import MetricsService
from backend.services.postprocess_service import PostprocessService
from backend.config import settings
import psutil
import psutil
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
START_TIME = time.time()

# Singletons (initialized in app.py startup, but we access instances here)
def get_gen_service():
    return GenerationService()

def get_lid_service():
    return LIDService()

@router.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    start = time.time()
    gen_service = get_gen_service()
    lid_service = get_lid_service()
    
    # 1. Format Prompt
    instruction = gen_service.format_style_prompt(
        request.sentence_1, 
        request.sentence_2, 
        request.sentence_3, 
        request.language_pair, 
        request.style, 
        request.english_usage
    )
    logger.info(f"[Step 1] User Sentences: '{request.sentence_1}', '{request.sentence_2}', '{request.sentence_3}' | Pair: {request.language_pair} | Rendered Instruction: '{instruction}'")
    
    # 2. Generate
    try:
        raw_output = gen_service.generate(instruction=instruction, temperature=request.temperature)
        raw_text = raw_output.get("generated_text", "")
        logger.info(f"[Step 2] Generated Output: '{raw_text}'")
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
    # 3. Postprocess
    clean_text = PostprocessService.clean_text(raw_text)
    logger.info(f"[Step 3] Postprocessed Text: '{clean_text}'")
    
    # 4. LID Tagging
    try:
        tag_result = lid_service.tag_text(clean_text, request.language_pair)
        logger.info(f"[Step 4] LID Labels Generated: {tag_result['labels']}")
    except RuntimeError as e:
        logger.error(f"LID Service Unavailable: {str(e)}")
        raise HTTPException(status_code=503, detail=f"LID Service Unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"LID tagging failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LID tagging failed: {str(e)}")
        
    # 5. Calculate CMI
    cmi = MetricsService.calculate_cmi(tag_result["labels"])
    logger.info(f"[Step 5] Calculated CMI: {cmi}")
    
    latency = time.time() - start
    token_count = len(tag_result["tokens"])
    
    return GenerateResponse(
        generated_text=clean_text,
        rendered_prompt=instruction,
        language_tags=tag_result["labels"],
        cmi=cmi,
        latency=latency,
        inference_config={"temperature": request.temperature},
        token_count=token_count
    )

@router.post("/tag", response_model=TagResponse)
def tag_text(request: TagRequest):
    lid_service = get_lid_service()
    try:
        result = lid_service.tag_text(request.text, request.language_pair)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"LID Service Unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LID tagging failed: {str(e)}")
        
    cmi = MetricsService.calculate_cmi(result["labels"])
    
    return TagResponse(
        tokens=result["tokens"],
        labels=result["labels"],
        confidence=1.0,
        cmi=cmi
    )

@router.get("/health", response_model=HealthResponse)
def health_check():
    uptime = time.time() - START_TIME
    gen = get_gen_service()
    lid = get_lid_service()
    
    status = "healthy" if gen.initialized and lid.initialized else "initializing"
    
    return HealthResponse(
        status=status,
        generator_loaded=gen.initialized,
        indicbert_loaded=lid.initialized,
        tokenizer_loaded=gen.initialized, # Tokenizer loads with generator
        lid_mock_mode=lid.is_mock_mode,
        uptime=uptime,
        device=settings.DEVICE,
        model_versions={
            "generator": settings.MODEL_PATH,
            "lora": settings.LORA_ADAPTER_PATH,
            "lid": settings.INDICBERT_MODEL_PATH
        }
    )

@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    # In a real scenario, this would query the loaded model object directly.
    return ModelInfoResponse(
        model_version="TriMixGen-v1.0 (GEN_003)",
        tokenizer="google/mt5-small",
        parameter_count=300000000, # Approx for mt5-small
        generation_configuration={
            "max_length": 256,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        },
        deployment_metadata={
            "device": settings.DEVICE,
            "lora_applied": True
        }
    )
