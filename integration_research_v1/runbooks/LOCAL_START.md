# Local start: selected PyTorch research cascade

Run from `/mnt/g/nafld-detectorl` in two terminals. The model files remain in the restricted research workspace and are not copied into this repository.

```bash
PYTHONPATH=. NAFLD_RESEARCH_ROOT='/mnt/g/NAFLD New' \
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
