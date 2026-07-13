import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
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

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

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

# SPA Fallback Route
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Do not intercept API or docs routes
    if full_path.startswith("api/") or full_path in ["docs", "redoc", "openapi.json"]:
        raise HTTPException(status_code=404, detail="Not Found")
        
    dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
    requested_file = os.path.join(dist_dir, full_path)
    
    # Serve requested static file (e.g. assets, favicon)
    if os.path.isfile(requested_file):
        return FileResponse(requested_file)
        
    # Serve index.html for SPA routing
    index_file = os.path.join(dist_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
        
    return {"error": "Frontend not built. Please run 'npm run build' in the frontend directory."}
