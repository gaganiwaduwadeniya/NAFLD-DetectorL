# 02 — Service and API implementation

`pytorch_service` is an isolated FastAPI service. Startup fails closed unless the exact recorded runtime, both manifest hashes, dataset version, five binary models and five staging models verify. It imports the authoritative preprocessing/model/inference implementation from the selected research workspace: EXIF transpose, RGB full field, aspect-preserving black letterbox to 224×224, ImageNet normalization, evaluation mode and mean probability aggregation.

Endpoints are `GET /health`, `GET /ready`, `POST /api/predict`, and `POST /api/explain`. Uploads accept JPEG/PNG only, require MIME/signature agreement, are capped at 20 MB and constrained by dimensions/pixel count. The response contract is schema version 4 and is machine-readable at `contracts/prediction_response.schema.json`.

The binary NAFLD score `q` is the five-member mean. If `q < 0.5`, final stage is 0 and grading is not executed. Otherwise the five EfficientNet-B0 vectors are averaged and the conditional argmax chooses Stage 1/2/3. Joint scores are derived supplementary values and are not calibrated clinical probabilities.

Express authenticates the public request, validates bytes before forwarding, validates all contract identities and probability/gating invariants before storage, and writes history only after success. It stores the final stage, conditional scores when present, hashes, contract/model/dataset identifiers, geometry and latency. Research-v1 inputs are processed in memory and not retained by default. `/api/explain` is authenticated and stateless. `/api/model/health` exposes readiness to authenticated users. Doctor scan detail now enforces record ownership; administrators retain intended access.

Rollback is explicit: restart the old TensorFlow service and Express with `MODEL_PROVIDER=tensorflow_v3`. A PyTorch outage never invokes V3 silently.
