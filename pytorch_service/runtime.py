"""Fail-closed loader around the authoritative research inference modules."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import psutil
import torch

from .contract import (
    BINARY_MANIFEST_SHA256,
    DATASET_VERSION,
    STAGING_MANIFEST_SHA256,
    build_response,
    validate_image_bytes,
)


EXPECTED_VERSIONS = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "timm": "1.0.29",
    "numpy": "2.1.3",
    "Pillow": "13.0.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResearchRuntime:
    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.binary = None
        self.staging = None
        self.loaded_at = None
        self.load_seconds = None
        self.rss_bytes = None

    def _root(self) -> Path:
        return (Path(__file__).resolve().parent / "research_v1").resolve()

    def _verify_environment(self) -> dict:
        import timm
        import torchvision
        import PIL
        versions = {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "timm": str(timm.__version__),
            "numpy": str(np.__version__),
            "Pillow": str(PIL.__version__),
        }
        if versions != EXPECTED_VERSIONS:
            raise RuntimeError(f"Exact recorded inference runtime required; got {versions}")
        return versions

    def load(self) -> None:
        started = time.perf_counter()
        root = self._root()
        source_root = root / "src"
        if not source_root.is_dir():
            raise RuntimeError("Repository-local research inference source is unavailable")
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        versions = self._verify_environment()
        binary_root = root / "checkpoints/research-v1/binary"
        staging_root = root / "checkpoints/research-v1/staging"
        if sha256_file(binary_root / "manifest.json") != BINARY_MANIFEST_SHA256:
            raise RuntimeError("Binary export manifest hash mismatch")
        if sha256_file(staging_root / "manifest.json") != STAGING_MANIFEST_SHA256:
            raise RuntimeError("Staging export manifest hash mismatch")
        from nafld.inference.pipeline import TaskPredictor
        self.binary = TaskPredictor(binary_root, device="cpu")
        self.staging = TaskPredictor(staging_root, device="cpu")
        if len(self.binary.models) != 5 or len(self.staging.models) != 5:
            raise RuntimeError("Exactly five binary and five staging members are required")
        if self.binary.dataset_version != DATASET_VERSION or self.staging.dataset_version != DATASET_VERSION:
            raise RuntimeError("Model dataset version mismatch")

        safe_fixture = Image.new("RGB", (32, 24), (0, 0, 0))
        first_binary = self.binary.predict_image(safe_fixture)
        second_binary = self.binary.predict_image(safe_fixture)
        first_staging = self.staging.predict_image(safe_fixture)
        second_staging = self.staging.predict_image(safe_fixture)
        if not np.array_equal(first_binary, second_binary) or not np.array_equal(first_staging, second_staging):
            raise RuntimeError("Same-runtime deterministic warm-up failed")

        self.ready = True
        self.error = None
        self.loaded_at = time.time()
        self.load_seconds = time.perf_counter() - started
        self.rss_bytes = psutil.Process().memory_info().rss
        self.versions = versions

    def load_fail_closed(self) -> None:
        try:
            self.load()
        except Exception as exc:
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict:
        return {
            "status": "ready" if self.ready else "not_ready",
            "ready": self.ready,
            "contract_id": "nafld-pytorch-research-v1-single-frame-v1",
            "provider": "pytorch_research_v1",
            "loaded_models": {"binary": 5 if self.ready else 0, "staging": 5 if self.ready else 0, "total": 10 if self.ready else 0},
            "device": "cpu",
            "versions": getattr(self, "versions", None),
            "load_seconds": self.load_seconds,
            "rss_bytes": self.rss_bytes,
            "error": self.error,
            "research_use_only": True,
        }

    def predict(self, payload: bytes, content_type: str) -> dict:
        if not self.ready or self.binary is None or self.staging is None:
            raise RuntimeError("Model service is not ready")
        image = validate_image_bytes(payload, content_type)
        started = time.perf_counter()
        response = build_response(self.binary, self.staging, image, (time.perf_counter() - started) * 1000)
        # build_response performs inference, so replace the initial timing placeholder.
        response["service_inference_ms"] = round((time.perf_counter() - started) * 1000, 3)
        response["input_sha256"] = hashlib.sha256(payload).hexdigest()
        response["input_geometry"] = {"width": image.width, "height": image.height, "mode": image.mode}
        return response

    def explain(self, payload: bytes, content_type: str) -> tuple[bytes, dict]:
        if not self.ready or self.binary is None or self.staging is None:
            raise RuntimeError("Model service is not ready")
        image = validate_image_bytes(payload, content_type)
        result = self.predict(payload, content_type)
        if result["final_stage"] == 0:
            predictor, target_index, label = self.binary, 1, "NAFLD detector target"
        else:
            predictor = self.staging
            target_index = int(result["final_stage"]) - 1
            label = f"Stage {result['final_stage']} staging target"
        from nafld.inference.explain import grad_cam
        prepared = predictor.preprocessors[0].process(image)
        pixels = torch.from_numpy(prepared.pixels).unsqueeze(0).to(predictor.device)
        heatmap = grad_cam(predictor.models[0], pixels, target_index=target_index)
        left, top, right, bottom = prepared.geometry["padding_ltrb"]
        height, width = heatmap.shape
        unpadded = heatmap[top:height-bottom if bottom else height, left:width-right if right else width]
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        heat = Image.fromarray(np.uint8(np.clip(unpadded, 0, 1) * 255), mode="L").resize(oriented.size, Image.Resampling.BILINEAR)
        heat_array = np.asarray(heat, dtype=np.uint8)
        colored = np.zeros((*heat_array.shape, 3), dtype=np.uint8)
        colored[..., 0] = heat_array
        colored[..., 1] = heat_array // 3
        overlay = Image.blend(oriented, Image.fromarray(colored, mode="RGB"), 0.35)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG", optimize=True)
        metadata = {
            "model": "DenseNet121 binary" if result["final_stage"] == 0 else "EfficientNet-B0 staging",
            "member": 0,
            "experiment_id": predictor.member_ids[0],
            "target_index": target_index,
            "target_label": label,
            "qualitative_only": True,
            "input_sha256": hashlib.sha256(payload).hexdigest(),
        }
        return buffer.getvalue(), metadata
