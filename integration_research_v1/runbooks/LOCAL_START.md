# Local start: selected PyTorch research cascade

The supported start command works from native Windows PowerShell/Command
Prompt and from WSL/Linux. WSL is not required on Windows:

```text
npm run app
```

The platform-neutral npm entry point selects PowerShell on Windows and Bash on
WSL/Linux. It verifies the restricted model bundle, creates the appropriate
platform-specific Python 3.12 environment, repairs locked Node dependencies,
builds both UI and API, starts both services, and opens the dashboard. The
verified inference code and model bundle are repository-local under
`pytorch_service/research_v1`; no second workspace is required. Full
prerequisites and troubleshooting are in `RUN_NEW_UI.md`.

## Manual WSL/Linux troubleshooting

Run from `/mnt/g/nafld-detectorl` in two terminals only when troubleshooting:

```bash
PYTHONPATH=. \
  .runtime/pytorch-research-v1/bin/python -m uvicorn pytorch_service.app:app \
  --host 127.0.0.1 --port 5002
```

Wait until `GET http://127.0.0.1:5002/ready` reports ten loaded models. Then:

```bash
MODEL_PROVIDER=pytorch_research_v1 \
ML_SERVICE_URL=http://127.0.0.1:5002 \
ML_INFERENCE_TIMEOUT_MS=60000 \
STORE_UPLOAD_IMAGES=false \
USE_LOCAL_DB=true \
npm run dev
```

Open `http://127.0.0.1:3001`. This configuration has no silent fallback. To reproduce historical V3 behavior, stop the processes and explicitly start its old TensorFlow service and Express with `MODEL_PROVIDER=tensorflow_v3` and the old V3 service URL.

## Checks

```bash
npm run lint
npm run test:integration-contract
PYTHONPATH=. .runtime/pytorch-research-v1/bin/python -m pytest -p no:cacheprovider -q integration_research_v1/tests/test_service_contract.py
```

The runtime is local CPU and research-only. Do not expose either service publicly. Uploaded images are not retained in research-v1 history unless `STORE_UPLOAD_IMAGES=true` is deliberately approved. `USE_LOCAL_DB=true` prevents the bundled Firebase configuration from being used during a local demo.
