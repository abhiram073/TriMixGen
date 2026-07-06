# Preprocessing Calibration Phase Plan

The preprocessing validation highlighted several systemic issues, primarily the Romanized Confidence Trap and dynamic schema parsing failures. Rather than lowering the global quality threshold, we will implement a calibration phase to intelligently adapt to dataset-specific characteristics and generate visual reports.

## Proposed Changes

### 1. Dataset-Specific Confidence Thresholds
**Component:** `configs/preprocessing.yaml` & `src/features/language_annotator.py`
We will move from a single `sentence_confidence_threshold` to a mapping configuration:
```yaml
language_annotator:
  dataset_thresholds:
    "alpaca": 0.65       # Accommodates 100% Romanized Te (Tier 6 @ 0.70)
    "hold": 0.75         # Stricter threshold for messy natural code-mixed data
    "sentiment": 0.75    # Native Telugu data
    "default": 0.75
```
The annotator will accept a `dataset_name` argument and dynamically apply the correct threshold.

### 2. Robust Schema Resolver & Header Recovery
**Component:** `scripts/run_preprocessing.py`
*   **Column Matching:** A case-insensitive match against a broader alias list (`['text', 'sentence', 'comment', 'comments', 'tweet', 'output', 'input']`).
*   **Header Recovery:** If the resolved column names are abnormally long (e.g. `len > 25` characters) or contain spaces/punctuation, it indicates pandas consumed the first row as the header. The script will automatically convert the headers back into a data row and use string heuristics to identify the text column (e.g., column with highest average string length).

### 3. Low-Confidence Analysis Report
**Component:** `src/features/preprocessing_pipeline.py`
The pipeline will explicitly aggregate rejection statistics:
*   Dataset-wise rejection rates.
*   The primary reason for rejection (e.g., "Too many Ambiguous tokens", "Failed dict lookups").
*   Distribution of confidences for the rejected sentences.

### 4. Visualizations
**Component:** `scripts/run_preprocessing.py`
Using `matplotlib`/`seaborn`, the script will automatically output three PNGs to `data/processed/`:
1.  **Confidence Histogram:** Showing the distribution of sentence average confidences before the threshold cut.
2.  **Heuristic Usage Bar Chart:** Which tiers are doing the most work.
3.  **Rejection Distribution:** Dataset-wise retention vs. rejection ratios.

## Verification Plan
1.  Run the updated `run_preprocessing.py` across all datasets.
2.  Verify `HOLD-Telugu_test.parquet` and `Telugu-Sentiment` splits are parsed successfully.
3.  Verify `Telugu-Alpaca-Romanized` produces a high-confidence dataset without blindly reducing global standards.
4.  Review the generated PNG visualizations and low-confidence analysis report.
