"""Versioned API contract for the selected PyTorch research cascade."""

from __future__ import annotations

from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError


CONTRACT_ID = "nafld-pytorch-research-v1-single-frame-v1"
DATASET_VERSION = "48b42abbee97b892591054019cdb4f84a65af9e0cd88d09fcbf8b3ae2a8b9856"
BINARY_MANIFEST_SHA256 = "daa75075d2de25823a2648dcb061b273e46459513657014cb239b5b8d2975cd9"
STAGING_MANIFEST_SHA256 = "12a0894534756dd380d08a4bfe9a6fe8086abde1420fd068140e709d3d83a52a"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 40_000_000

STAGE_LABELS = {
    0: "Stage 0 — Non-NAFLD / Normal",
    1: "Stage 1 — NAFLD, Mild steatosis",
    2: "Stage 2 — NAFLD, Moderate steatosis",
    3: "Stage 3 — NAFLD, Severe steatosis",
}


def _probabilities(values, width: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (result.shape != (width,) or not np.isfinite(result).all()
            or np.any(result < 0) or np.any(result > 1)
            or not np.isclose(result.sum(), 1.0, atol=1e-5)):
        raise ValueError("Model returned an invalid probability vector")
    return result


def validate_image_bytes(payload: bytes, content_type: str) -> Image.Image:
    if not payload:
        raise ValueError("Image payload is empty")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MB upload limit")
    signatures = {
        "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": payload.startswith(b"\xff\xd8"),
    }
    if content_type not in signatures:
        raise ValueError("Only JPEG and PNG images are accepted")
    if not signatures[content_type]:
        raise ValueError("Image MIME type and byte signature do not agree")
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Image decode failed: {type(exc).__name__}") from exc
    if image.mode not in {"RGB", "L"}:
        raise ValueError(f"Unsupported image mode {image.mode}; expected 8-bit RGB or L")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Multi-frame images are not accepted by the single-frame contract")
    width, height = image.size
    if width < 1 or height < 1 or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError("Image dimensions are outside the supported range")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Image pixel count exceeds the safety limit")
    return image


def build_response(binary_predictor, staging_predictor, image: Image.Image, latency_ms: float) -> dict:
    binary = _probabilities(binary_predictor.predict_image(image), 2)
    q = float(binary[1])
    stage = 0
    conditional = None
    joint = None
    grading_performed = False
    if q >= 0.5:
        grading = _probabilities(staging_predictor.predict_image(image), 3)
        stage = int(np.argmax(grading)) + 1
        grading_performed = True
        conditional = {str(i + 1): float(value) for i, value in enumerate(grading)}
        joint = {"0": float(binary[0]), **{
            str(i + 1): float(q * value) for i, value in enumerate(grading)
        }}
        if not math.isclose(sum(joint.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Derived joint stage scores do not sum to one")
    return {
        "success": True,
        "schema_version": 4,
        "contract_id": CONTRACT_ID,
        "provider": "pytorch_research_v1",
        "model_version": {"binary": "nafld-research-v1", "staging": "nafld-research-v1"},
        "dataset_version": DATASET_VERSION,
        "research_use_only": True,
        "clinically_validated": False,
        "probability_calibration_verified": False,
        "input_mode": "single_frame",
        "final_stage": stage,
        "final_stage_label": STAGE_LABELS[stage],
        "binary_probabilities": {"Non-NAFLD": float(binary[0]), "NAFLD": q},
        "binary_threshold": 0.5,
        "grading_performed": grading_performed,
        "conditional_stage_probabilities": conditional,
        "joint_stage_probabilities": joint,
        "model_bundle": {
            "binary": {"architecture": "densenet121", "members": 5, "manifest_sha256": BINARY_MANIFEST_SHA256},
            "staging": {"architecture": "efficientnet_b0", "members": 5, "manifest_sha256": STAGING_MANIFEST_SHA256},
        },
        "service_inference_ms": round(float(latency_ms), 3),
        "explanation_available": True,
    }
