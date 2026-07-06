# TriMixGen Developer Setup

Welcome to TriMixGen! This guide explains how to configure your local machine for reproducible, production-quality ML development.

## 1. Prerequisites
- Python 3.12+
- Git

## 2. Cloning the Repository
```bash
git clone <your-repo-url>
cd TriMixGen
```

## 3. Environment Setup
We enforce the use of virtual environments to avoid global package pollution.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Installing Dependencies
Our `requirements.txt` is modularized into Core, NLP, Training, Visualization, Deployment, and Testing libraries.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Pre-commit Hooks (Code Quality)
We use `black`, `ruff`, and `isort` to automatically format code before every commit.
```bash
# This ensures that code styles are automatically fixed when you run `git commit`
pre-commit install
```

## 6. Data Version Control (DVC) Setup
*(Note: DVC initialization will occur in Phase 3. This is just for awareness.)*
To track massive parquet datasets without bloating Git:
```bash
dvc init
git commit -m "Initialize DVC"
```
When you run the download scripts, DVC will track the `/data` folder while Git ignores the raw CSV/Parquet files.

---

# 🛑 Phase 2.5 Validation Checklist

Before we proceed to **Phase 3 (Exploratory Data Analysis)**, ensure the following checklist is complete on your machine:

- [ ] `.venv` is created and activated.
- [ ] `pip install -r requirements.txt` ran successfully without conflicts.
- [ ] You can run `python -c "import torch; print(torch.cuda.is_available())"` to verify device detection.
- [ ] The `data/` and `outputs/` directories exist (check via `ls`).
- [ ] The `configs/` directory contains `logging.yaml`, `datasets.yaml`, and `model.yaml`.
- [ ] The `src/utils/` directory contains `seed.py` and `logger.py`.
