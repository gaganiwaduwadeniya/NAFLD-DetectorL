# Run the New NAFLD Research UI Locally

This guide starts the current React/Express application with the selected
PyTorch research-v1 cascade. It is intended for local research demonstration
only. It is not a clinically validated diagnostic system and neither service
should be exposed to the public internet.

## What starts

The application uses two local processes:

1. **PyTorch inference service** at `http://127.0.0.1:5002`
2. **React UI + Express API** at `http://127.0.0.1:3001`

The selected cascade contains two model architectures, not ten unrelated
models:

- five selected DenseNet121 fold checkpoints are averaged for Normal versus
  NAFLD;
- when NAFLD probability is at least `0.5`, five selected EfficientNet-B0 fold
  checkpoints are averaged for Mild, Moderate, or Severe grading.

These ten selected fold checkpoints are the verified research release. They
are not ten epoch snapshots, and the service fails closed if the manifests,
runtime versions, checkpoint count, or dataset identity do not match.

## Required folder

The application is self-contained inside this repository. The launcher
supports both native Windows and WSL/Linux:

```text
<parent>/nafld-detectorl
```

Examples:

```text
Windows: G:\nafld-detectorl
WSL:     /mnt/g/nafld-detectorl
```

The exact inference source, manifests and selected model bundle live under
`pytorch_service/research_v1/`. No second workspace or environment variable is
required at runtime.

## Prerequisites

### Native Windows (WSL is not required)

Install these once, then reopen PowerShell:

- Node.js 20 or newer, including `npm`: <https://nodejs.org/>
- `uv`: <https://docs.astral.sh/uv/getting-started/installation/>
- a complete checkout of this application repository, including its Git LFS
  model objects

Check the two commands in ordinary Windows PowerShell:

```powershell
node --version
uv --version
```

### WSL/Linux

- Node.js 20 or newer, including `npm`
- `uv`, `curl`, and standard Linux utilities
- a complete checkout of this application repository, including its Git LFS
  model objects

The Python virtual environment and Node packages do not need to exist yet.
The launcher creates or repairs them under this repository.

## One-command start (recommended on Windows and WSL)

Open PowerShell, Command Prompt, or a WSL terminal in the application
repository and run:

```bash
npm run app
```

The npm entry point detects the operating system. It uses native PowerShell on
Windows and Bash on WSL/Linux. It does not invoke or require WSL on Windows.

That single command:

1. locates and verifies the repository-local ten selected fold checkpoints;
2. creates or updates `.runtime/pytorch-research-v1` with Python 3.12 and the
   exact locked inference packages;
3. checks or installs the locked Node dependencies;
4. builds the production UI and Express API;
5. starts the model service and waits for its fail-closed readiness response;
6. starts the application, waits for the dashboard, and opens it in the default
   browser; and
7. keeps both services attached to the terminal so one `Ctrl+C` stops them.

The first run requires internet access to install missing dependencies and can
take several minutes. Later runs reuse the verified environment. The model
service reads only the bundle inside this repository.

Useful launcher modes:

```bash
npm run app -- --check         # verify existing files/runtime without changes
npm run app -- --prepare-only  # update dependencies and build, but do not start
npm run app -- --no-open       # run without opening a browser automatically
```

Runtime logs and the fresh local database are kept under
`.runtime/nafld-local/`, which is ignored by Git.

On native Windows, the managed Python executable is
`.runtime\pytorch-research-v1-windows\Scripts\python.exe`. On WSL/Linux it is
`.runtime/pytorch-research-v1/bin/python`. These environments are intentionally
platform-specific and must not be copied from one operating system to another.

## Manual two-terminal start (WSL/Linux troubleshooting)

Use the commands below only when troubleshooting or developing the two services
independently.

## Start the application

Open two WSL/Linux terminals. Native Windows users should normally use the
one-command launcher above.

### Terminal 1: PyTorch inference service

```bash
cd /mnt/g/nafld-detectorl

PYTHONPATH=. \
.runtime/pytorch-research-v1/bin/python -m uvicorn pytorch_service.app:app \
  --host 127.0.0.1 \
  --port 5002
```

Initial loading can take roughly one to two minutes on CPU and uses about 1 GB
of memory. Keep this terminal open. Do not start the UI backend until the
readiness check below succeeds.

In a third terminal, or from the browser, check:

```bash
curl -s http://127.0.0.1:5002/ready
```

The JSON response must include:

```json
{
  "ready": true,
  "provider": "pytorch_research_v1",
  "loaded_models": {
    "binary": 5,
    "staging": 5,
    "total": 10
  },
  "device": "cpu",
  "research_use_only": true
}
```

If `/ready` returns HTTP 503, inspect the Terminal 1 output and the
troubleshooting section below. Do not switch silently to the old TensorFlow
model.

### Terminal 2: React UI and Express API

```bash
cd /mnt/g/nafld-detectorl

MODEL_PROVIDER=pytorch_research_v1 \
ML_SERVICE_URL=http://127.0.0.1:5002 \
ML_INFERENCE_TIMEOUT_MS=60000 \
STORE_UPLOAD_IMAGES=false \
USE_LOCAL_DB=true \
npm run dev
```

Keep this terminal open. Then browse to:

```text
http://127.0.0.1:3001
```

If Windows cannot resolve that address through WSL, also try:
`http://localhost:3001`.

### Stable production-style local test (recommended for the G: drive)

The development server can rebuild slowly when the repository is mounted from
Windows. For a stable demonstration, build once and run the generated server:

```bash
cd /mnt/g/nafld-detectorl
npm run build

NODE_ENV=production \
MODEL_PROVIDER=pytorch_research_v1 \
ML_SERVICE_URL=http://127.0.0.1:5002 \
ML_INFERENCE_TIMEOUT_MS=60000 \
STORE_UPLOAD_IMAGES=false \
USE_LOCAL_DB=true \
npm run start
```

Use `LOCAL_DATA_DIR=/tmp/nafld-ui-manual-test` before `npm run start` when you
want a disposable, clean demonstration database. A new local database retains
the Doctor and Administrator accounts but starts with zero scan records.

## Sign in

The local fallback database includes two demonstration accounts:

| Role | Email | Password |
|---|---|---|
| Doctor | `doctor@system.com` | `doctor123` |
| Administrator | `admin@system.com` | `admin123` |

The login page has buttons that prefill these credentials. You can also use
the registration page to create a local research-test account. These demo
credentials are for local testing only and must never be used for deployment.

## Test the new model path

1. Sign in as the Doctor account.
2. Confirm the dashboard model card reports the PyTorch research-v1 provider.
3. Upload one permitted, de-identified JPEG or PNG ultrasound image.
4. Enter only fictional/de-identified patient metadata for testing.
5. Submit the scan and wait for the real CPU inference result.
6. Inspect the Normal/NAFLD gate result. Grading scores appear only when the
   gate enters the NAFLD branch.
7. Open the saved detail view and, where available, request the qualitative
   Grad-CAM visualization.

Inference is CPU-bound and can take several seconds. The prediction is genuine
output from the selected checkpoint ensembles; there is no simulated response
and no automatic TensorFlow fallback.

## Privacy and local data behavior

- Accepted files are JPEG/PNG only, with a 20 MB upload limit.
- `STORE_UPLOAD_IMAGES=false` keeps uploaded image bytes out of research-v1
  scan history.
- `USE_LOCAL_DB=true` prevents the repository's bundled Firebase configuration
  from being used during the local demonstration.
- With the one-command launcher, prediction metadata and user-entered patient
  fields are written to `.runtime/nafld-local/data/db.json` unless
  `LOCAL_DATA_DIR` is set.
- For disposable manual testing, isolate all history in a temporary folder:

  ```bash
  LOCAL_DATA_DIR=/tmp/nafld-ui-manual-test \
  MODEL_PROVIDER=pytorch_research_v1 \
  ML_SERVICE_URL=http://127.0.0.1:5002 \
  ML_INFERENCE_TIMEOUT_MS=60000 \
  STORE_UPLOAD_IMAGES=false \
  USE_LOCAL_DB=true \
  npm run dev
  ```

- Do not use identifiable patient names, overlays, headers, or source scans in
  demonstrations or screenshots.
- Grad-CAM is qualitative research evidence, not proof of clinical reasoning.

## Useful checks

Check the model service directly:

```bash
curl -s http://127.0.0.1:5002/health
curl -s http://127.0.0.1:5002/ready
```

Check that the UI server responds:

```bash
curl -I http://127.0.0.1:3001
```

Run the focused repository checks:

```bash
cd /mnt/g/nafld-detectorl
npm run lint
npm run test:integration-contract
npm run test:local-db
PYTHONPATH=. .runtime/pytorch-research-v1/bin/python \
  -m pytest -p no:cacheprovider -q \
  integration_research_v1/tests/test_service_contract.py
```

## Stop the application

When using `npm run app`, press `Ctrl+C` once in its terminal to stop both
managed processes. Manual two-terminal mode requires stopping each terminal.

Confirm both ports are closed:

```bash
curl -sS http://127.0.0.1:3001 || echo "UI stopped"
curl -sS http://127.0.0.1:5002/health || echo "Model service stopped"
```

## Troubleshooting

### `GET /ready` returns 503

Read the JSON error from `/health` and the Terminal 1 log. Common causes are:

- `pytorch_service/research_v1` is incomplete (for example, Git LFS objects
  were not downloaded);
- a checkpoint or manifest hash differs;
- the exact recorded Python package versions are not active;
- fewer than five binary or five staging members were found.

Do not bypass those checks. They protect the verified inference contract.

### The dashboard says the model is unavailable

Confirm Terminal 1 is still running, `/ready` returns HTTP 200, and Terminal 2
was started with both:

```text
MODEL_PROVIDER=pytorch_research_v1
ML_SERVICE_URL=http://127.0.0.1:5002
```

Restart Terminal 2 after correcting environment variables.

### Port already in use

On Windows PowerShell, find the process IDs using the ports:

```powershell
Get-NetTCPConnection -LocalPort 3001,5002 -ErrorAction SilentlyContinue |
  Select-Object LocalPort, State, OwningProcess
```

On WSL/Linux:

```bash
ss -ltnp | rg ':3001|:5002'
```

Stop only the confirmed old NAFLD process. Do not kill unrelated processes.

### Windows says `uv` is unavailable

Install `uv` using its official Windows instructions, close and reopen
PowerShell so `PATH` refreshes, verify `uv --version`, and run `npm run app`
again. Do not copy the WSL `.runtime` environment to Windows; the launcher will
create the correct native Windows environment.

### Windows PowerShell script execution is restricted

Run through `npm run app`. The platform dispatcher invokes this repository's
checked-in PowerShell launcher with a process-scoped execution-policy bypass;
it does not change the computer's persistent PowerShell policy.

### Prediction times out

The default Express timeout is 60 seconds. First confirm that the model service
is ready and the machine is not under heavy CPU or memory pressure. Do not
increase the timeout to conceal a failed or non-ready model service.

### Browser opens but displays an old UI

Hard-refresh the page (`Ctrl+Shift+R`) and verify that Terminal 2 was started
from `/mnt/g/nafld-detectorl` with the PyTorch provider variables above.

## Evidence and implementation references

- `integration_research_v1/runbooks/LOCAL_START.md`
- `integration_research_v1/reports/02_SERVICE_AND_API_IMPLEMENTATION.md`
- `integration_research_v1/reports/04_END_TO_END_TEST_RESULTS.md`
- `integration_research_v1/contracts/prediction_response.schema.json`
- `pytorch_service/runtime.py`
- `pytorch_service/app.py`
- `src/server/pytorchContract.ts`

The old TensorFlow/V3 service and artifacts remain preserved. Switching back is
an explicit separate configuration action; the new path never falls back to it
automatically.
