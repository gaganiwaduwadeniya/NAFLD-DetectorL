#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${NAFLD_RUNTIME_DIR:-${ROOT_DIR}/.runtime}"
PY_RUNTIME_DIR="${RUNTIME_DIR}/pytorch-research-v1"
STATE_DIR="${RUNTIME_DIR}/nafld-local"
LOG_DIR="${STATE_DIR}/logs"
DATA_DIR="${LOCAL_DATA_DIR:-${STATE_DIR}/data}"
PYTHON_BIN="${PY_RUNTIME_DIR}/bin/python"
REQUIREMENTS_FILE="${ROOT_DIR}/pytorch_service/requirements.lock.txt"
REQUIREMENTS_STAMP="${PY_RUNTIME_DIR}/.requirements.sha256"
MODEL_PORT="${ML_SERVICE_PORT:-5002}"
APP_PORT="${PORT:-3001}"
MODEL_URL="http://127.0.0.1:${MODEL_PORT}"
APP_URL="http://127.0.0.1:${APP_PORT}"
CHECK_ONLY=false
PREPARE_ONLY=false
OPEN_BROWSER=true
MODEL_PID=""
APP_PID=""

usage() {
  cat <<'EOF'
Usage: npm run app -- [options]

Prepare and run the local NAFLD research interface from one terminal.

Options:
  --check         Verify existing tools, runtime, models, and application files only.
  --prepare-only  Update dependencies and build the application, then stop.
  --no-open       Start the system without opening a browser.
  --help          Show this help.

Optional environment variables:
  NAFLD_RESEARCH_ROOT  Path to the separate "NAFLD New" research workspace.
  LOCAL_DATA_DIR       Local JSON database directory.
  PORT                 UI/API port (default: 3001).
  ML_SERVICE_PORT      Model-service port (default: 5002).
EOF
}

log() {
  printf '[NAFLD] %s\n' "$*"
}

fail() {
  printf '[NAFLD] ERROR: %s\n' "$*" >&2
  exit 1
}

for argument in "$@"; do
  case "${argument}" in
    --check) CHECK_ONLY=true ;;
    --prepare-only) PREPARE_ONLY=true ;;
    --no-open) OPEN_BROWSER=false ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: ${argument}. Run 'npm run app -- --help'." ;;
  esac
done

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  command_exists "$1" || fail "Required command '$1' is unavailable. See RUN_NEW_UI.md."
}

resolve_research_root() {
  if [[ -n "${NAFLD_RESEARCH_ROOT:-}" ]]; then
    printf '%s\n' "${NAFLD_RESEARCH_ROOT}"
    return
  fi

  local candidate
  for candidate in \
    "${ROOT_DIR}/../NAFLD New" \
    "/mnt/g/NAFLD New"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  fail "The separate 'NAFLD New' workspace was not found. Set NAFLD_RESEARCH_ROOT to its path."
}

verify_model_bundle() {
  local research_root="$1"
  [[ -d "${research_root}/src/nafld" ]] || fail "Research inference source is missing under ${research_root}/src."
  [[ -f "${research_root}/checkpoints/research-v1/binary/manifest.json" ]] || fail "Binary model manifest is missing."
  [[ -f "${research_root}/checkpoints/research-v1/staging/manifest.json" ]] || fail "Staging model manifest is missing."

  local binary_count staging_count
  binary_count="$(find "${research_root}/checkpoints/research-v1/binary" -maxdepth 1 -type f -name 'fold-*.pt' | wc -l | tr -d ' ')"
  staging_count="$(find "${research_root}/checkpoints/research-v1/staging" -maxdepth 1 -type f -name 'fold-*.pt' | wc -l | tr -d ' ')"
  [[ "${binary_count}" == "5" ]] || fail "Expected 5 selected DenseNet121 checkpoints; found ${binary_count}."
  [[ "${staging_count}" == "5" ]] || fail "Expected 5 selected EfficientNet-B0 checkpoints; found ${staging_count}."
}

requirements_hash() {
  sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}'
}

python_runtime_valid() {
  [[ -x "${PYTHON_BIN}" ]] || return 1
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as metadata
import sys

expected = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "timm": "1.0.29",
    "numpy": "2.1.3",
    "Pillow": "13.0.0",
    "fastapi": "0.121.3",
    "uvicorn": "0.38.0",
    "python-multipart": "0.0.20",
    "jsonschema": "4.25.1",
    "psutil": "7.1.3",
}

if sys.version_info[:2] != (3, 12):
    raise SystemExit(1)

for package, version in expected.items():
    if metadata.version(package) != version:
        raise SystemExit(1)
PY
}

prepare_python_runtime() {
  local expected_hash actual_hash=""
  expected_hash="$(requirements_hash)"
  if [[ -f "${REQUIREMENTS_STAMP}" ]]; then
    actual_hash="$(tr -d '[:space:]' < "${REQUIREMENTS_STAMP}")"
  fi

  if [[ "${actual_hash}" == "${expected_hash}" ]] && python_runtime_valid; then
    log "Python runtime is current."
    return
  fi

  if [[ "${CHECK_ONLY}" == "true" ]]; then
    fail "The Python runtime is missing or out of date. Run 'npm run app' to repair it."
  fi

  require_command uv
  mkdir -p "${RUNTIME_DIR}"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "Creating the managed Python 3.12 environment..."
    uv venv --python 3.12 "${PY_RUNTIME_DIR}"
  fi

  log "Synchronizing the exact verified PyTorch runtime..."
  uv pip install \
    --python "${PYTHON_BIN}" \
    --requirements "${REQUIREMENTS_FILE}" \
    --torch-backend cu128 \
    --exact \
    --strict

  python_runtime_valid || fail "Python packages do not match the verified inference contract after synchronization."
  printf '%s\n' "${expected_hash}" > "${REQUIREMENTS_STAMP}"
  log "Python runtime verified."
}

node_runtime_valid() {
  node -e "const major=Number(process.versions.node.split('.')[0]); process.exit(major >= 20 ? 0 : 1)" >/dev/null 2>&1
}

prepare_node_runtime() {
  require_command node
  require_command npm
  node_runtime_valid || fail "Node.js 20 or newer is required."

  if npm ls --depth=0 --silent >/dev/null 2>&1; then
    log "Node dependencies are current."
    return
  fi

  if [[ "${CHECK_ONLY}" == "true" ]]; then
    fail "Node dependencies are missing or out of date. Run 'npm run app' to repair them."
  fi

  log "Installing the locked Node dependencies..."
  npm ci
  npm ls --depth=0 --silent >/dev/null 2>&1 || fail "Node dependency validation failed after npm ci."
}

build_application() {
  if [[ "${CHECK_ONLY}" == "true" ]]; then
    [[ -f "${ROOT_DIR}/dist/server.cjs" ]] || fail "The production build is missing. Run 'npm run app' to build it."
    log "Existing production build found."
    return
  fi

  log "Building the production UI and API..."
  npm run build
  [[ -f "${ROOT_DIR}/dist/server.cjs" ]] || fail "Production server bundle was not created."
}

wait_for_model() {
  local attempts=0
  while (( attempts < 180 )); do
    if curl --silent --fail "${MODEL_URL}/ready" | grep -q '"ready":true'; then
      return 0
    fi
    if [[ -n "${MODEL_PID}" ]] && ! kill -0 "${MODEL_PID}" 2>/dev/null; then
      return 1
    fi
    attempts=$((attempts + 1))
    sleep 1
  done
  return 1
}

wait_for_application() {
  local attempts=0
  while (( attempts < 60 )); do
    if curl --silent --fail "${APP_URL}/login" >/dev/null; then
      return 0
    fi
    if [[ -n "${APP_PID}" ]] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      return 1
    fi
    attempts=$((attempts + 1))
    sleep 1
  done
  return 1
}

cleanup() {
  trap - EXIT INT TERM
  log "Stopping local services..."
  [[ -z "${APP_PID}" ]] || kill "${APP_PID}" 2>/dev/null || true
  [[ -z "${MODEL_PID}" ]] || kill "${MODEL_PID}" 2>/dev/null || true
  [[ -z "${APP_PID}" ]] || wait "${APP_PID}" 2>/dev/null || true
  [[ -z "${MODEL_PID}" ]] || wait "${MODEL_PID}" 2>/dev/null || true
}

open_dashboard() {
  if [[ "${OPEN_BROWSER}" != "true" ]]; then
    return
  fi
  if command_exists powershell.exe; then
    powershell.exe -NoProfile -Command "Start-Process '${APP_URL}'" >/dev/null 2>&1 || true
  elif command_exists xdg-open; then
    xdg-open "${APP_URL}" >/dev/null 2>&1 || true
  elif command_exists open; then
    open "${APP_URL}" >/dev/null 2>&1 || true
  else
    log "No supported browser opener was found. Open ${APP_URL} manually."
  fi
}

main() {
  cd "${ROOT_DIR}"
  require_command curl
  require_command sha256sum
  require_command find

  local research_root
  research_root="$(resolve_research_root)"
  verify_model_bundle "${research_root}"
  log "Verified the selected 5 + 5 checkpoint bundle."

  prepare_python_runtime
  prepare_node_runtime
  build_application

  if [[ "${CHECK_ONLY}" == "true" ]]; then
    log "Readiness check passed. No services were started."
    return
  fi
  if [[ "${PREPARE_ONLY}" == "true" ]]; then
    log "Preparation completed. No services were started."
    return
  fi

  mkdir -p "${LOG_DIR}" "${DATA_DIR}"
  umask 077
  if [[ ! -s "${STATE_DIR}/jwt.secret" ]]; then
    "${PYTHON_BIN}" -c "import secrets; print(secrets.token_hex(32))" > "${STATE_DIR}/jwt.secret"
  fi
  local jwt_secret
  jwt_secret="$(<"${STATE_DIR}/jwt.secret")"

  if curl --silent --fail "${MODEL_URL}/ready" >/dev/null 2>&1; then
    fail "Port ${MODEL_PORT} already has a ready model service. Stop it before using this managed launcher."
  fi
  if curl --silent --fail "${APP_URL}/login" >/dev/null 2>&1; then
    fail "Port ${APP_PORT} is already serving an application. Stop it before using this managed launcher."
  fi

  trap cleanup EXIT INT TERM

  log "Starting the verified model service; first load can take about one minute..."
  PYTHONPATH="${ROOT_DIR}" \
  NAFLD_RESEARCH_ROOT="${research_root}" \
  "${PYTHON_BIN}" -m uvicorn pytorch_service.app:app \
    --host 127.0.0.1 \
    --port "${MODEL_PORT}" \
    > "${LOG_DIR}/model-service.log" 2>&1 &
  MODEL_PID=$!

  if ! wait_for_model; then
    tail -n 80 "${LOG_DIR}/model-service.log" >&2 || true
    fail "Model service did not become ready."
  fi
  log "Model service is ready."

  log "Starting the local research dashboard..."
  NODE_ENV=production \
  PORT="${APP_PORT}" \
  LOCAL_DATA_DIR="${DATA_DIR}" \
  USE_LOCAL_DB=true \
  MODEL_PROVIDER=pytorch_research_v1 \
  ML_SERVICE_URL="${MODEL_URL}" \
  ML_INFERENCE_TIMEOUT_MS=60000 \
  STORE_UPLOAD_IMAGES=false \
  JWT_SECRET="${jwt_secret}" \
  node dist/server.cjs > "${LOG_DIR}/application.log" 2>&1 &
  APP_PID=$!

  if ! wait_for_application; then
    tail -n 80 "${LOG_DIR}/application.log" >&2 || true
    fail "The dashboard did not become ready."
  fi

  log "System ready: ${APP_URL}"
  log "Logs: ${LOG_DIR}"
  log "Press Ctrl+C once to stop both services."
  open_dashboard

  wait -n "${MODEL_PID}" "${APP_PID}"
  fail "A managed service stopped unexpectedly. Review ${LOG_DIR}."
}

main
