"""FastAPI boundary for the exact selected PyTorch research-v1 cascade."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from .runtime import ResearchRuntime


runtime = ResearchRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(runtime.load_fail_closed)
    yield


app = FastAPI(title="NAFLD PyTorch Research V1", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return runtime.status()


@app.get("/ready")
def ready():
    status = runtime.status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail="Model bundle is not ready")
    return status


async def _read_upload(image: UploadFile) -> tuple[bytes, str]:
    content_type = image.content_type or ""
    payload = await image.read(20 * 1024 * 1024 + 1)
    return payload, content_type


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    payload, content_type = await _read_upload(image)
    try:
        return await asyncio.to_thread(runtime.predict, payload, content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/explain")
async def explain(image: UploadFile = File(...)):
    payload, content_type = await _read_upload(image)
    try:
        png, metadata = await asyncio.to_thread(runtime.explain, payload, content_type)
        return Response(
            content=png,
            media_type="image/png",
            headers={"X-NAFLD-Explanation": json.dumps(metadata, separators=(",", ":"))},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
