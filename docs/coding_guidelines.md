# Coding Guidelines

We enforce strict coding standards for TriMixGen to ensure it is GitHub-worthy and maintainable.

### 1. Style & Formatting
- **Black:** We use Black with a line length of 88 characters.
- **Ruff:** All code must pass Ruff linting (a faster alternative to Flake8).
- **isort:** Imports must be alphabetically sorted and grouped (Standard Library -> Third-Party -> First-Party).

### 2. Typing
All Python functions must use type hinting. This improves IDE auto-completion and readability.
```python
# BAD
def process(text):
    return text.lower()

# GOOD
def process(text: str) -> str:
    return text.lower()
```

### 3. Modularity
- No hardcoded paths in scripts. Use `os.path.join()` or `pathlib.Path`.
- All hyperparameters (learning rate, batch size, model name) must reside in `configs/*.yaml`.
- Do not use `print()` for production scripts. Use the custom logger from `src.utils.logger`.

### 4. Reproducibility
- At the start of every training script, call `set_seed(42)` from `src.utils.seed`.
- Save all experiments with timestamps (use `generate_experiment_name` from `src.utils.seed`) into the `outputs/` folder.
