from __future__ import annotations

import logging
import os
from http import HTTPStatus
from pathlib import Path

from flask import Flask, jsonify, request

from .inference import ImageValidationError, ModelBundle, ModelBundleError


def create_app(bundle_dir: str | Path | None = None) -> Flask:
    app = Flask(__name__)

    # Loading is deliberately synchronous. The process must not accept traffic
    # unless the contract, hashes, shapes, and warm-up checks all succeed.
    model_bundle = ModelBundle(bundle_dir=bundle_dir)
    app.extensions["nafld_model_bundle"] = model_bundle

    @app.get("/health")
    @app.get("/healthz")
    @app.get("/api/health")
    def health():
        payload = model_bundle.health_payload()
        status = HTTPStatus.OK if model_bundle.ready else HTTPStatus.SERVICE_UNAVAILABLE
        return jsonify(payload), status

    @app.post("/api/predict")
    def predict():
        # --- Service readiness guard ---
        if not model_bundle.ready:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Model service is not ready. Please retry.",
                        "model_version": model_bundle.model_version,
                    }
                ),
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

        # --- Require the 'image' multipart field ---
        image_file = request.files.get("image")
        if image_file is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Missing multipart field 'image'. Send the image as a form-data file under the key 'image'.",
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )

        image_bytes = image_file.read()
        # Use the upload's declared content type; fall back to mimetype from Flask.
        content_type = (
            image_file.content_type
            or image_file.mimetype
            or "application/octet-stream"
        )

        try:
            result = model_bundle.predict_single_frame(image_bytes, content_type)
        except ImageValidationError as exc:
            # Bad image from the client — 400.
            return (
                jsonify({"success": False, "error": str(exc)}),
                HTTPStatus.BAD_REQUEST,
            )
        except ModelBundleError as exc:
            # Internal model failure — 500; log detail server-side only.
            app.logger.error("ModelBundleError during inference: %s", exc, exc_info=True)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Model inference failed. See server logs.",
                        "model_version": model_bundle.model_version,
                    }
                ),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except Exception as exc:  # noqa: BLE001
            # Unexpected failure — 500.
            app.logger.error("Unexpected error during inference: %s", exc, exc_info=True)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "An unexpected error occurred. See server logs.",
                    }
                ),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        return jsonify(result), HTTPStatus.OK

    return app


logging.basicConfig(
    level=os.getenv("ML_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("ML_SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("ML_SERVICE_PORT", "5001")),
        debug=False,
    )

