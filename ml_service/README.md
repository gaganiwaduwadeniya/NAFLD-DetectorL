# NAFLD V3 inference service

Phase 1 provides fail-fast loading and readiness checks for the complete V3
ensemble. Prediction is intentionally disabled until Phase 2 adds the exact
training-time preprocessing and cascade inference path.

## Requirements

- 64-bit Python 3.12
- Enough memory to load ten EfficientNetB0 fold models
- The unchanged `nafld_v3` export directory

## Local setup on Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\ml_service\requirements.txt
$env:MODEL_BUNDLE_DIR = 'G:\NAFLD-DetectorL\nafld_v3'
python -m ml_service.app
```

`MODEL_BUNDLE_DIR` is optional when `nafld_v3` is directly under the project
root.

## Readiness check

After all ten models have loaded and completed warm-up inference:

```powershell
Invoke-RestMethod http://127.0.0.1:5001/health
```

The response must report:

```json
{
  "status": "ready",
  "ready": true,
  "schema_version": 3,
  "loaded_models": {
    "binary": 5,
    "grading": 5,
    "total": 10
  }
}
```

The process exits during startup if the contract is invalid, TensorFlow does
not match version 2.20.0, a model is missing or corrupted, a saved shape is
wrong, or warm-up output is invalid.
