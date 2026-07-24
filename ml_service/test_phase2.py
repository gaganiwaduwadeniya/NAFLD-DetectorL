r"""Phase 2 verification script.

Run from the project root with the MODEL_BUNDLE_DIR env var set:

    $env:MODEL_BUNDLE_DIR = 'G:\NAFLD-DetectorL\nafld_v3'
    & '.\.venv\Scripts\python.exe' ml_service/test_phase2.py

All tests use the Flask test client (no live server required). A synthetic
224x224 grayscale JPEG is used as the test image.
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import traceback


# ---------------------------------------------------------------------------
# Minimal synthetic image builders (no external image library needed).
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 224, height: int = 224, value: int = 128) -> bytes:
    """Return a valid minimal JPEG for a solid-grey image using Pillow if
    available, otherwise fall back to TensorFlow encoding."""
    try:
        from PIL import Image as PILImage
        img = PILImage.new("L", (width, height), color=value)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: use TensorFlow to encode a tensor as JPEG.
    import numpy as np
    import tensorflow as tf
    arr = np.full((height, width, 3), value, dtype=np.uint8)
    encoded = tf.image.encode_jpeg(arr, quality=85)
    return encoded.numpy()


def _make_png_bytes(width: int = 32, height: int = 32, value: int = 200) -> bytes:
    """Return a valid PNG using Pillow or TensorFlow."""
    try:
        from PIL import Image as PILImage
        img = PILImage.new("L", (width, height), color=value)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass

    import numpy as np
    import tensorflow as tf
    arr = np.full((height, width, 3), value, dtype=np.uint8)
    encoded = tf.image.encode_png(arr)
    return encoded.numpy()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}", flush=True)
    if not condition and detail:
        print(f"         -> {detail}", flush=True)
    _results.append((name, condition, detail))


def section(title: str) -> None:
    print(f"\n{'-' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'-' * 60}", flush=True)


# ---------------------------------------------------------------------------
# Main test suite
# ---------------------------------------------------------------------------

def run_tests() -> int:
    os.environ.setdefault("MODEL_BUNDLE_DIR", str(
        ((__import__("pathlib").Path(__file__).resolve().parents[1]) / "nafld_v3")
    ))

    print("\n==================================================================")
    print("         NAFLD Detector -- Phase 2 Verification Script")
    print("==================================================================")
    print(f"\nMODEL_BUNDLE_DIR = {os.environ.get('MODEL_BUNDLE_DIR')}")

    # ------------------------------------------------------------------
    # Import and build Flask app (loads all 10 models — may take ~30s).
    # ------------------------------------------------------------------
    section("0 . App startup (loads all 10 fold models)")
    try:
        from ml_service.app import create_app
        flask_app = create_app()
        client = flask_app.test_client()
        check("Flask app created without error", True)
    except Exception as exc:
        check("Flask app created without error", False, traceback.format_exc())
        print("\n  App failed to start.", flush=True)
        return 1

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    section("1 . Health endpoint")
    resp = client.get("/health")
    data = resp.get_json()
    check("GET /health returns 200", resp.status_code == 200,
          f"status={resp.status_code}")
    check("ready=True in health payload", data.get("ready") is True,
          json.dumps(data, indent=2))
    check("10 models loaded",
          data.get("loaded_models", {}).get("total") == 10,
          str(data.get("loaded_models")))

    # ------------------------------------------------------------------
    # Build test images
    # ------------------------------------------------------------------
    section("2 . Building synthetic test images")
    try:
        jpeg_mid = _make_jpeg_bytes(224, 224, value=128)
        jpeg_white = _make_jpeg_bytes(224, 224, value=255)
        jpeg_black = _make_jpeg_bytes(224, 224, value=0)
        png_img = _make_png_bytes(64, 64, value=180)
        check("Synthetic images built", True)
    except Exception as exc:
        check("Synthetic images built", False, traceback.format_exc())
        print("\n  Image generation failed.", flush=True)
        return 1

    def post_image_typed(img_bytes: bytes, mime: str, filename: str = "test.img"):
        from werkzeug.datastructures import FileStorage
        fs = FileStorage(stream=io.BytesIO(img_bytes), filename=filename,
                         content_type=mime)
        return client.post(
            "/api/predict",
            data={"image": fs},
            content_type="multipart/form-data",
        )

    # ------------------------------------------------------------------
    # Valid JPEG upload — mid-grey image
    # ------------------------------------------------------------------
    section("3 . Valid JPEG upload (mid-grey 224x224)")
    resp = post_image_typed(jpeg_mid, "image/jpeg", "grey.jpg")
    check("/api/predict no longer returns 501",
          resp.status_code != 501,
          f"status={resp.status_code}")
    check("/api/predict returns 200",
          resp.status_code == 200,
          f"status={resp.status_code}")

    if resp.status_code == 200:
        d = resp.get_json()
        check("success=True", d.get("success") is True, str(d))
        check("binary_result present", "binary_result" in d, str(list(d.keys())))
        check("final_label present", "final_label" in d, str(list(d.keys())))
        check("research_use_only=True", d.get("research_use_only") is True)
        check("input_mode=single_frame", d.get("input_mode") == "single_frame")
        check("model_version present", bool(d.get("model_version")))
        check("contract_sha256 present (64 hex chars)",
              len(d.get("contract_sha256", "")) == 64)

        prob_nafld = d.get("binary_prob_nafld", -1)
        prob_non = d.get("binary_prob_non_nafld", -1)
        check("binary_prob_nafld in [0,1]", 0.0 <= prob_nafld <= 1.0,
              f"got {prob_nafld}")
        check("binary_prob_non_nafld in [0,1]", 0.0 <= prob_non <= 1.0,
              f"got {prob_non}")
        check("binary probs sum ~1.0",
              abs(prob_nafld + prob_non - 1.0) < 1e-4,
              f"sum={prob_nafld + prob_non}")
        check("binary_fold_probs has 5 entries",
              len(d.get("binary_fold_probs", [])) == 5,
              str(d.get("binary_fold_probs")))
        check("binary_threshold matches contract",
              abs(d.get("binary_threshold", 0) - 0.5536812544) < 1e-6,
              f"got {d.get('binary_threshold')}")

        final_label = d.get("final_label", "")
        check("final_label is one of three valid values",
              final_label in ("Non-NAFLD",
                              "NAFLD-Grade1_Mild",
                              "NAFLD-Grade2_Moderate_Severe"),
              f"got {final_label!r}")

        grading_performed = d.get("grading_performed")
        if grading_performed:
            check("grading_result present when grading_performed",
                  "grading_result" in d)
            g_prob_mod = d.get("grading_prob_moderate_severe", -1)
            g_prob_mild = d.get("grading_prob_mild", -1)
            check("grading_prob_moderate_severe in [0,1]",
                  0.0 <= g_prob_mod <= 1.0, f"got {g_prob_mod}")
            check("grading probs sum ~1.0",
                  abs(g_prob_mod + g_prob_mild - 1.0) < 1e-4,
                  f"sum={g_prob_mod + g_prob_mild}")
            check("grading_fold_probs has 5 entries",
                  len(d.get("grading_fold_probs", [])) == 5)
            check("grading_threshold matches contract",
                  abs(d.get("grading_threshold", 0) - 0.6078062057) < 1e-6,
                  f"got {d.get('grading_threshold')}")
        else:
            check("grading keys absent for Non-NAFLD result",
                  "grading_result" not in d and "grading_prob_moderate_severe" not in d,
                  str(list(d.keys())))

    # ------------------------------------------------------------------
    # White image (boundary)
    # ------------------------------------------------------------------
    section("4 . Boundary: pure-white image")
    resp_w = post_image_typed(jpeg_white, "image/jpeg", "white.jpg")
    check("White image returns 200", resp_w.status_code == 200,
          f"status={resp_w.status_code}")
    if resp_w.status_code == 200:
        dw = resp_w.get_json()
        check("White image has valid final_label",
              dw.get("final_label") in ("Non-NAFLD",
                                        "NAFLD-Grade1_Mild",
                                        "NAFLD-Grade2_Moderate_Severe"))

    # ------------------------------------------------------------------
    # Black image (boundary)
    # ------------------------------------------------------------------
    section("5 . Boundary: pure-black image")
    resp_b = post_image_typed(jpeg_black, "image/jpeg", "black.jpg")
    check("Black image returns 200", resp_b.status_code == 200,
          f"status={resp_b.status_code}")

    # ------------------------------------------------------------------
    # PNG upload
    # ------------------------------------------------------------------
    section("6 . Valid PNG upload")
    resp_png = post_image_typed(png_img, "image/png", "scan.png")
    check("PNG returns 200", resp_png.status_code == 200,
          f"status={resp_png.status_code}")

    # ------------------------------------------------------------------
    # Error: missing image field
    # ------------------------------------------------------------------
    section("7 . Error: missing 'image' field")
    resp_no_field = client.post("/api/predict",
                                data={"file": (io.BytesIO(jpeg_mid), "img.jpg")},
                                content_type="multipart/form-data")
    check("Missing field returns 400", resp_no_field.status_code == 400,
          f"status={resp_no_field.status_code}")
    d_err = resp_no_field.get_json()
    check("Error response has 'error' key", "error" in d_err)

    # ------------------------------------------------------------------
    # Error: corrupt / random bytes with valid JPEG magic
    # ------------------------------------------------------------------
    section("8 . Error: corrupt bytes declared as JPEG")
    garbage = b"\xff\xd8\xff" + os.urandom(50)  # valid magic, random rest
    resp_garbage = post_image_typed(garbage, "image/jpeg", "bad.jpg")
    check("Corrupt image returns 400 or 500",
          resp_garbage.status_code in (400, 500),
          f"status={resp_garbage.status_code}")

    # ------------------------------------------------------------------
    # Error: wrong MIME type (PNG bytes declared as JPEG)
    # ------------------------------------------------------------------
    section("9 . Error: MIME/magic mismatch (PNG bytes as image/jpeg)")
    resp_mismatch = post_image_typed(png_img, "image/jpeg", "mismatch.jpg")
    check("MIME mismatch returns 400", resp_mismatch.status_code == 400,
          f"status={resp_mismatch.status_code}")

    # ------------------------------------------------------------------
    # Error: unsupported MIME type
    # ------------------------------------------------------------------
    section("10 . Error: unsupported MIME type (image/tiff)")
    resp_tiff = post_image_typed(jpeg_mid, "image/tiff", "scan.tiff")
    check("Unsupported MIME returns 400", resp_tiff.status_code == 400,
          f"status={resp_tiff.status_code}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for name, ok, detail in _results:
            if not ok:
                print(f"    X {name}")
    else:
        print("  -- ALL PASSED")
    print(f"{'=' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
