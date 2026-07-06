# TriMixGen Backend Deployment Guide

This guide covers running the FastAPI application in a production environment.

## Requirements
* Python 3.9+
* TriMixGen Pipeline Checkpoints (`outputs/experiments/gen_003/best_model`)
* 4GB+ RAM (8GB+ recommended if running CPU inference)

## Installation
1. Install dependencies from the virtual environment.
```bash
pip install -r requirements.txt
```

2. Configure `.env` variables if defaults are insufficient.
```env
# Optional Overrides
BACKEND_CORS_ORIGINS=https://trimixgen.yourdomain.com
MODEL_PATH=google/mt5-small
LORA_ADAPTER_PATH=outputs/experiments/gen_003/best_model
```

## Running the Server

### Development
Use `uvicorn` with reload enabled for testing:
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### Production
Use `gunicorn` with `uvicorn` workers to handle concurrent requests:
```bash
gunicorn backend.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
```

*Note: If running on CPU, `gunicorn` workers will duplicate RAM usage. If memory is tight, reduce `-w 4` to `-w 2`.*

## Logging
The application automatically writes structured logs to `backend.log`. It rotates the file when it hits 10MB (keeping 5 backups). Ensure the application has write permissions to its directory.
