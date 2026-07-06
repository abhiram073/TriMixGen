from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api import router
from backend.utils import add_exception_handlers, add_logging_middleware, logger
from backend.services.generation_service import GenerationService
from backend.services.lid_service import LIDService

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Backend API for TriMixGen Code-Mixing Generation",
    version="1.0.0"
)

# CORS Middleware
logger.info(f"Adding CORS origins: {settings.BACKEND_CORS_ORIGINS}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers & Logging
add_exception_handlers(app)
add_logging_middleware(app)

# Include Router
app.include_router(router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up TriMixGen Backend...")
    logger.info(f"Target execution device: {settings.DEVICE}")
    
    # Lazy model loading: Initialize singletons here
    logger.info("Triggering singleton model initializations...")
    
    # We initialize the services which triggers the model loading
    # They are wrapped in try-except in their __init__ methods
    GenerationService()
    LIDService()
    
    logger.info("Startup complete. API is ready to receive requests.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down TriMixGen Backend.")
