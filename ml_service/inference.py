from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


LOGGER = logging.getLogger(__name__)

EXPECTED_SCHEMA_VERSION = 3
EXPECTED_TASKS = ("binary", "grading")
EXPECTED_AGGREGATION = "arithmetic_mean_of_fold_softmax_probabilities"
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "nafld_v3"


class ModelBundleError(RuntimeError):
    """Raised when the exported V3 bundle is incomplete or incompatible."""


class ImageValidationError(ValueError):
    """Raised when the uploaded image is invalid or unsupported."""


# ---------------------------------------------------------------------------
# Supported MIME types and their file-signature (magic bytes) prefixes.
# ---------------------------------------------------------------------------
_SUPPORTED_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/bmp", "image/x-bmp", "image/x-ms-bmp"}
)
# (offset, bytes) pairs that must match for each MIME family.
_MAGIC: dict[str, tuple[int, bytes]] = {
    "jpeg": (0, b"\xff\xd8\xff"),
    "png": (0, b"\x89PNG\r\n\x1a\n"),
    "bmp": (0, b"BM"),
}
_MIME_TO_MAGIC_KEY: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/bmp": "bmp",
    "image/x-bmp": "bmp",
    "image/x-ms-bmp": "bmp",
}
# Maximum accepted upload size (20 MB — single ultrasound frame).
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bundle_dir(bundle_dir: str | Path | None = None) -> Path:
    configured = bundle_dir or os.getenv("MODEL_BUNDLE_DIR")
    path = Path(configured).expanduser() if configured else DEFAULT_BUNDLE_DIR
    return path.resolve()


class ModelBundle:
    """Loads and validates the complete binary and grading V3 ensembles."""

    def __init__(self, bundle_dir: str | Path | None = None) -> None:
        self.bundle_dir = resolve_bundle_dir(bundle_dir)
        self.contract_path = self.bundle_dir / "inference_contract_v3.json"
        self.models_dir = self.bundle_dir / "models"
        self.ready = False
        self.contract: dict[str, Any] = {}
        self.contract_sha256 = ""
        self.binary_models: tuple[keras.Model, ...] = ()
        self.grading_models: tuple[keras.Model, ...] = ()
        self.runtime_devices: tuple[str, ...] = ()

        self._load()

    def _load(self) -> None:
        LOGGER.info("Loading NAFLD V3 model bundle from %s", self.bundle_dir)

        if not self.contract_path.is_file():
            raise ModelBundleError(
                f"Inference contract was not found: {self.contract_path}"
            )
        if not self.models_dir.is_dir():
            raise ModelBundleError(f"Model directory was not found: {self.models_dir}")

        try:
            with self.contract_path.open("r", encoding="utf-8") as handle:
                contract = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelBundleError(
                f"Unable to read inference contract: {self.contract_path}"
            ) from exc

        if not isinstance(contract, dict):
            raise ModelBundleError("Inference contract must contain a JSON object.")

        self.contract = contract
        self.contract_sha256 = sha256_file(self.contract_path)
        self._validate_contract()
        self._configure_tensorflow_runtime()

        # Verify every artifact before allocating memory for any model.
        verified_paths = {
            task: self._verify_task_artifacts(task) for task in EXPECTED_TASKS
        }

        self.binary_models = self._load_task_models(
            "binary", verified_paths["binary"]
        )
        self.grading_models = self._load_task_models(
            "grading", verified_paths["grading"]
        )

        expected_total = self.fold_count * len(EXPECTED_TASKS)
        loaded_total = len(self.binary_models) + len(self.grading_models)
        if loaded_total != expected_total:
            raise ModelBundleError(
                f"Loaded {loaded_total} models; expected {expected_total}."
            )

        self.ready = True
        LOGGER.info(
            "NAFLD V3 bundle ready: %d binary folds and %d grading folds",
            len(self.binary_models),
            len(self.grading_models),
        )

    @property
    def fold_count(self) -> int:
        return int(self.contract["ensemble"]["folds"])

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.contract["input"]["shape"])

    @property
    def model_version(self) -> str:
        return str(self.contract.get("notebook_version", "3"))

    def _validate_contract(self) -> None:
        contract = self.contract

        if contract.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise ModelBundleError(
                "Unsupported inference contract schema: "
                f"{contract.get('schema_version')!r}; expected {EXPECTED_SCHEMA_VERSION}."
            )
        if contract.get("research_use_only") is not True:
            raise ModelBundleError(
                "The V3 service requires research_use_only=true in the contract."
            )

        framework = contract.get("framework")
        if not isinstance(framework, dict):
            raise ModelBundleError("Contract framework section is missing.")
        trained_tf_version = str(framework.get("tensorflow", ""))
        if trained_tf_version != tf.__version__:
            raise ModelBundleError(
                "TensorFlow version mismatch: bundle was exported with "
                f"{trained_tf_version}, runtime is {tf.__version__}."
            )

        input_contract = contract.get("input")
        if not isinstance(input_contract, dict):
            raise ModelBundleError("Contract input section is missing.")
        if input_contract.get("shape") != [224, 224, 3]:
            raise ModelBundleError(
                f"Unexpected contract input shape: {input_contract.get('shape')!r}."
            )
        if input_contract.get("dtype") != "float32":
            raise ModelBundleError("V3 input dtype must be float32.")
        if input_contract.get("pixel_value_range") != [0.0, 255.0]:
            raise ModelBundleError("V3 input pixel range must be [0.0, 255.0].")
        if input_contract.get("external_rescaling") is not False:
            raise ModelBundleError("V3 must not use external input rescaling.")

        ensemble = contract.get("ensemble")
        if not isinstance(ensemble, dict):
            raise ModelBundleError("Contract ensemble section is missing.")
        if ensemble.get("aggregation") != EXPECTED_AGGREGATION:
            raise ModelBundleError(
                f"Unsupported ensemble aggregation: {ensemble.get('aggregation')!r}."
            )
        folds = ensemble.get("folds")
        if not isinstance(folds, int) or folds <= 0:
            raise ModelBundleError(f"Invalid ensemble fold count: {folds!r}.")

        for task in EXPECTED_TASKS:
            task_contract = contract.get(task)
            if not isinstance(task_contract, dict):
                raise ModelBundleError(f"Contract task section is missing: {task}.")

            entries = task_contract.get("models")
            if not isinstance(entries, list) or len(entries) != folds:
                raise ModelBundleError(
                    f"Task {task} must define exactly {folds} model entries."
                )

            fold_numbers = sorted(entry.get("fold") for entry in entries)
            if fold_numbers != list(range(folds)):
                raise ModelBundleError(
                    f"Task {task} has invalid fold identifiers: {fold_numbers!r}."
                )

            class_names = task_contract.get("class_names")
            if not isinstance(class_names, list) or len(class_names) != 2:
                raise ModelBundleError(
                    f"Task {task} must define exactly two class names."
                )
            if task_contract.get("positive_class_index") != 1:
                raise ModelBundleError(
                    f"Task {task} positive_class_index must be 1."
                )

            threshold = task_contract.get("single_frame_threshold")
            if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                raise ModelBundleError(
                    f"Task {task} has an invalid single-frame threshold: {threshold!r}."
                )

    def _configure_tensorflow_runtime(self) -> None:
        devices = tf.config.list_physical_devices()
        self.runtime_devices = tuple(
            f"{device.device_type}:{device.name}" for device in devices
        )

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                raise ModelBundleError(
                    f"Unable to configure TensorFlow GPU memory growth for {gpu.name}."
                ) from exc

    def _verify_task_artifacts(self, task: str) -> tuple[Path, ...]:
        entries = sorted(
            self.contract[task]["models"], key=lambda entry: int(entry["fold"])
        )
        verified: list[Path] = []

        for entry in entries:
            model_file = entry.get("model_file")
            expected_hash = entry.get("sha256")

            if not isinstance(model_file, str) or Path(model_file).name != model_file:
                raise ModelBundleError(
                    f"Task {task} contains an unsafe model filename: {model_file!r}."
                )
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ModelBundleError(
                    f"Task {task} has an invalid SHA-256 for {model_file!r}."
                )

            path = self.models_dir / model_file
            if not path.is_file():
                raise ModelBundleError(f"Model file was not found: {path}")

            actual_hash = sha256_file(path)
            if actual_hash.lower() != expected_hash.lower():
                raise ModelBundleError(
                    f"SHA-256 mismatch for {model_file}: "
                    f"expected {expected_hash}, received {actual_hash}."
                )

            verified.append(path)

        LOGGER.info("Verified %d %s model artifacts", len(verified), task)
        return tuple(verified)

    def _load_task_models(
        self, task: str, paths: tuple[Path, ...]
    ) -> tuple[keras.Model, ...]:
        models: list[keras.Model] = []
        warmup_batch = tf.zeros((1, *self.input_shape), dtype=tf.float32)

        for path in paths:
            LOGGER.info("Loading %s model %s", task, path.name)
            try:
                model = keras.models.load_model(path, compile=False)
            except Exception as exc:
                raise ModelBundleError(f"Unable to load model: {path.name}") from exc

            model_input_shape = tuple(model.input_shape[1:])
            model_output_shape = tuple(model.output_shape[1:])
            if model_input_shape != self.input_shape:
                raise ModelBundleError(
                    f"{path.name} input shape is {model.input_shape}; "
                    f"expected (None, {', '.join(map(str, self.input_shape))})."
                )
            if model_output_shape != (2,):
                raise ModelBundleError(
                    f"{path.name} output shape is {model.output_shape}; expected (None, 2)."
                )

            try:
                probabilities = np.asarray(
                    model(warmup_batch, training=False), dtype=np.float32
                )
            except Exception as exc:
                raise ModelBundleError(
                    f"Warm-up inference failed for {path.name}."
                ) from exc

            if probabilities.shape != (1, 2):
                raise ModelBundleError(
                    f"{path.name} warm-up returned {probabilities.shape}; expected (1, 2)."
                )
            if not np.isfinite(probabilities).all():
                raise ModelBundleError(
                    f"{path.name} warm-up returned a non-finite probability."
                )
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
                raise ModelBundleError(
                    f"{path.name} warm-up probabilities do not sum to one."
                )

            models.append(model)

        return tuple(models)

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "starting",
            "ready": self.ready,
            "service": "nafld-v3-inference",
            "schema_version": self.contract.get("schema_version"),
            "model_version": self.model_version,
            "research_use_only": self.contract.get("research_use_only"),
            "tensorflow_version": tf.__version__,
            "contract_file": self.contract_path.name,
            "contract_sha256": self.contract_sha256,
            "folds": self.fold_count,
            "loaded_models": {
                "binary": len(self.binary_models),
                "grading": len(self.grading_models),
                "total": len(self.binary_models) + len(self.grading_models),
            },
            "input_shape": list(self.input_shape),
            "runtime_devices": list(self.runtime_devices),
        }

    # ------------------------------------------------------------------
    # Phase 2 — image preprocessing and cascade inference
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image_bytes(image_bytes: bytes, content_type: str) -> None:
        """Raise ImageValidationError for obviously bad inputs."""
        if not image_bytes:
            raise ImageValidationError("No image data received.")
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ImageValidationError(
                f"Image exceeds maximum allowed size of {_MAX_IMAGE_BYTES // (1024 * 1024)} MB."
            )

        # Normalise content type (strip parameters like "; charset=…").
        mime = content_type.split(";")[0].strip().lower() if content_type else ""
        if mime not in _SUPPORTED_MIME_TYPES:
            raise ImageValidationError(
                f"Unsupported image type {mime!r}. Accepted: JPEG, PNG, BMP."
            )

        # Magic-byte check — catches renamed or corrupted files.
        magic_key = _MIME_TO_MAGIC_KEY[mime]
        offset, expected = _MAGIC[magic_key]
        actual = image_bytes[offset : offset + len(expected)]
        if actual != expected:
            raise ImageValidationError(
                f"File signature does not match declared type {mime!r}."
            )

    @staticmethod
    def _preprocess(image_bytes: bytes) -> tf.Tensor:
        """Decode and preprocess one image to match training exactly.

        Returns a float32 tensor of shape (1, 224, 224, 3) with pixel
        values in [0, 255]. EfficientNetB0 performs rescaling internally;
        this function must NOT divide by 255.
        """
        # Decode: supports JPEG, PNG, BMP; output always 3 channels.
        try:
            raw = tf.image.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
                dtype=tf.uint8,
            )
        except (tf.errors.InvalidArgumentError, Exception) as exc:
            raise ImageValidationError(f"Unable to decode image: {exc}") from exc

        # Validate decoded dimensions.
        shape = raw.shape
        if len(shape) != 3:
            raise ImageValidationError(
                f"Decoded image has unexpected rank {len(shape)}; expected 3."
            )
        h, w, c = shape
        if h < 1 or w < 1 or c not in (1, 3, 4):
            raise ImageValidationError(
                f"Decoded image has invalid dimensions {h}x{w}x{c}."
            )

        # --- Match notebook preprocessing exactly ---

        # 1. Convert to grayscale (handles all channel counts via decode_image
        #    channels=3 above; rgb_to_grayscale reduces 3→1).
        gray = tf.image.rgb_to_grayscale(raw)  # (H, W, 1), uint8

        # 2. Cast to float32 (keep 0–255 range).
        gray_f = tf.cast(gray, tf.float32)  # (H, W, 1)

        # 3. Replicate single channel to three identical channels.
        rgb_f = tf.repeat(gray_f, repeats=3, axis=-1)  # (H, W, 3)

        # 4. TensorFlow bilinear resize with antialiasing to 224×224.
        resized = tf.image.resize(
            rgb_f,
            size=(224, 224),
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=True,
        )  # (224, 224, 3), float32

        # 5. Add batch dimension.
        batch = tf.expand_dims(resized, axis=0)  # (1, 224, 224, 3)

        return batch

    @staticmethod
    def _run_ensemble(
        models: tuple[keras.Model, ...], batch: tf.Tensor
    ) -> tuple[np.ndarray, list[float], float]:
        """Run all fold models and return (mean_probs, fold_pos_probs, std_pos).

        mean_probs — shape (1, 2), arithmetic mean of fold softmax vectors.
        fold_pos_probs — list of per-fold positive-class (index 1) probabilities.
        std_pos — standard deviation of fold positive-class probabilities.
        """
        fold_probs: list[np.ndarray] = []
        for model in models:
            probs = np.asarray(model(batch, training=False), dtype=np.float32)
            if probs.shape != (1, 2):
                raise ModelBundleError(
                    f"Model returned unexpected output shape {probs.shape}; expected (1, 2)."
                )
            if not np.isfinite(probs).all():
                raise ModelBundleError("Model returned non-finite probability.")
            fold_probs.append(probs)

        mean_probs = np.mean(fold_probs, axis=0)  # (1, 2)
        fold_pos_list = [float(p[0, 1]) for p in fold_probs]
        std_pos = float(np.std(fold_pos_list))
        return mean_probs, fold_pos_list, std_pos

    def predict_single_frame(
        self, image_bytes: bytes, content_type: str
    ) -> dict[str, Any]:
        """Run the full V3 cascade on a single uploaded image frame.

        Parameters
        ----------
        image_bytes:
            Raw bytes of the uploaded file.
        content_type:
            MIME type declared by the client (e.g. ``'image/jpeg'``).

        Returns
        -------
        dict
            Structured prediction result as specified in the Phase 2
            inference contract. Always contains binary result fields.
            Grading fields are only present when ``grading_performed`` is
            True (i.e. binary predicted NAFLD).

        Raises
        ------
        ImageValidationError
            When the input bytes are empty, too large, wrong MIME type, or
            have a mismatched file signature or decoding failure.
        ModelBundleError
            When a fold model returns an unexpected shape or non-finite
            values (internal server error).
        """
        if not self.ready:
            raise ModelBundleError("Model bundle is not ready.")

        # --- Input validation ---
        self._validate_image_bytes(image_bytes, content_type)

        # --- Preprocessing ---
        batch = self._preprocess(image_bytes)  # (1, 224, 224, 3), float32, 0–255

        # --- Binary ensemble ---
        binary_task = self.contract["binary"]
        binary_threshold = float(binary_task["single_frame_threshold"])
        binary_class_names: list[str] = binary_task["class_names"]

        bin_mean, bin_fold_probs, bin_fold_std = self._run_ensemble(
            self.binary_models, batch
        )
        prob_nafld = float(bin_mean[0, 1])
        prob_non_nafld = float(bin_mean[0, 0])
        binary_positive = prob_nafld >= binary_threshold
        binary_result = binary_class_names[1] if binary_positive else binary_class_names[0]

        LOGGER.info(
            "Binary prediction: %s (prob_nafld=%.4f, threshold=%.4f)",
            binary_result,
            prob_nafld,
            binary_threshold,
        )

        result: dict[str, Any] = {
            "success": True,
            "input_mode": "single_frame",
            "model_version": self.model_version,
            "contract_sha256": self.contract_sha256,
            "research_use_only": True,
            # Binary
            "binary_result": binary_result,
            "binary_prob_nafld": round(prob_nafld, 8),
            "binary_prob_non_nafld": round(prob_non_nafld, 8),
            "binary_threshold": binary_threshold,
            "binary_fold_probs": [round(p, 8) for p in bin_fold_probs],
            "binary_fold_std": round(bin_fold_std, 8),
        }

        if not binary_positive:
            # Non-NAFLD: grading is NOT performed.
            result["grading_performed"] = False
            result["final_label"] = "Non-NAFLD"
            LOGGER.info("Cascade stopped: Non-NAFLD.")
            return result

        # --- Grading ensemble (only when binary predicts NAFLD) ---
        grading_task = self.contract["grading"]
        grading_threshold = float(grading_task["single_frame_threshold"])
        grading_class_names: list[str] = grading_task["class_names"]

        gr_mean, gr_fold_probs, gr_fold_std = self._run_ensemble(
            self.grading_models, batch
        )
        prob_moderate_severe = float(gr_mean[0, 1])
        prob_mild = float(gr_mean[0, 0])
        grading_positive = prob_moderate_severe >= grading_threshold
        grading_result = (
            grading_class_names[1] if grading_positive else grading_class_names[0]
        )

        LOGGER.info(
            "Grading prediction: %s (prob_moderate_severe=%.4f, threshold=%.4f)",
            grading_result,
            prob_moderate_severe,
            grading_threshold,
        )

        result["grading_performed"] = True
        result["grading_result"] = grading_result
        result["grading_prob_moderate_severe"] = round(prob_moderate_severe, 8)
        result["grading_prob_mild"] = round(prob_mild, 8)
        result["grading_threshold"] = grading_threshold
        result["grading_fold_probs"] = [round(p, 8) for p in gr_fold_probs]
        result["grading_fold_std"] = round(gr_fold_std, 8)
        result["final_label"] = f"NAFLD-{grading_result}"

        return result
