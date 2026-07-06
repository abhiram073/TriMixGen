import logging
import sys
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
import uuid

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    
    # Rotating File Handler (10MB max, keep 5 backups)
    fh = RotatingFileHandler(
        "backend.log", maxBytes=10*1024*1024, backupCount=5
    )
    fh.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)
        
    return logger

logger = setup_logging()

def add_exception_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global Exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred.", "error_type": type(exc).__name__}
        )
        
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"Validation Error: {exc}")
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)}
        )

def add_logging_middleware(app: FastAPI):
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        req_id = str(uuid.uuid4())
        logger.info(f"Request started: {req_id} - {request.method} {request.url.path}")
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(f"Request completed: {req_id} - Status: {response.status_code} - Latency: {process_time:.4f}s")
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed: {req_id} - Latency: {process_time:.4f}s - Error: {str(e)}")
            raise e
