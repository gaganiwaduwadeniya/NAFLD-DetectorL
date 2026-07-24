"""Evaluate the V3 binary ensemble on labeled external image folders.

Expected layout (folder names are intentionally matched case-insensitively):

    Test_Data/
      normal/   -> ground truth Non-NAFLD
      NFLD/     -> ground truth NAFLD

This is a research evaluation utility. It uses the preprocessing, model bundle,
class order, and single-frame threshold from the production V3 inference code.
It never trains or modifies a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_service.inference import ModelBundle  # noqa: E402


CLASS_FOLDERS = {
    "normal": (0, "Non-NAFLD"),
    "nfld": (1, "NAFLD"),
}
MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    true_index: int
    true_label: str
    content_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V3 binary Non-NAFLD/NAFLD single-frame performance."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing normal/ and NFLD/ (default: this script's folder).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=PROJECT_ROOT / "nafld_v3",
        help="V3 model bundle directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: DATA_DIR/binary_v3_results).",
    )
    return parser.parse_args()


def discover_images(data_dir: Path) -> list[ImageItem]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Test-data directory does not exist: {data_dir}")

    discovered: list[ImageItem] = []
    seen_classes: set[str] = set()
    for folder in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        key = folder.name.casefold()
        if key not in CLASS_FOLDERS:
            continue
        seen_classes.add(key)
        true_index, true_label = CLASS_FOLDERS[key]
        for path in sorted(folder.rglob("*"), key=lambda p: str(p).casefold()):
            if not path.is_file():
                continue
            content_type = MIME_BY_SUFFIX.get(path.suffix.casefold())
            if content_type is None:
                continue
            discovered.append(ImageItem(path, true_index, true_label, content_type))

    missing = sorted(set(CLASS_FOLDERS) - seen_classes)
    if missing:
        raise FileNotFoundError(
            f"Required class folder(s) missing under {data_dir}: {', '.join(missing)}"
        )
    if not discovered:
        raise FileNotFoundError(f"No supported images found under {data_dir}")
    return discovered


def safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["true_index"] for row in rows], dtype=np.int32)
    predictions = np.asarray([row["predicted_index"] for row in rows], dtype=np.int32)
    scores = np.asarray([row["prob_nafld"] for row in rows], dtype=np.float64)

    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "evaluated_images": len(rows),
        "class_counts": {"Non-NAFLD": tn + fp, "NAFLD": tp + fn},
        "confusion_matrix": {
            "order": ["Non-NAFLD", "NAFLD"],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "accuracy": rounded(safe_div(tp + tn, len(rows))),
        "balanced_accuracy": rounded(
            None
            if sensitivity is None or specificity is None
            else (sensitivity + specificity) / 2.0
        ),
        "sensitivity_nafld_recall": rounded(sensitivity),
        "specificity_normal_recall": rounded(specificity),
        "precision_nafld": rounded(safe_div(tp, tp + fp)),
        "negative_predictive_value": rounded(safe_div(tn, tn + fn)),
        "f1_nafld": rounded(safe_div(2 * tp, 2 * tp + fp + fn)),
        "roc_auc": rounded(roc_auc_binary(labels, scores)),
        "mean_prob_nafld_by_true_class": {
            "Non-NAFLD": rounded(float(scores[labels == 0].mean())),
            "NAFLD": rounded(float(scores[labels == 1].mean())),
        },
    }


def predict_binary_image(
    bundle: ModelBundle, path: Path, content_type: str
) -> dict[str, Any]:
    """Predict one image without receiving or using its ground-truth label."""
    threshold = float(bundle.contract["binary"]["single_frame_threshold"])
    class_names = list(bundle.contract["binary"]["class_names"])
    image_bytes = path.read_bytes()
    bundle._validate_image_bytes(image_bytes, content_type)
    batch = bundle._preprocess(image_bytes)
    mean_probs, fold_probs, fold_std = bundle._run_ensemble(
        bundle.binary_models, batch
    )
    prob_nafld = float(mean_probs[0, 1])
    predicted_index = int(prob_nafld >= threshold)
    return {
        "predicted_index": predicted_index,
        "predicted_label": class_names[predicted_index],
        "prob_non_nafld": round(float(mean_probs[0, 0]), 8),
        "prob_nafld": round(prob_nafld, 8),
        "threshold": threshold,
        "margin_from_threshold": round(prob_nafld - threshold, 8),
        "fold_std": round(fold_std, 8),
        "fold_prob_nafld": [round(value, 8) for value in fold_probs],
    }


def run_evaluation(
    bundle: ModelBundle, items: list[ImageItem]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Infer each image independently, then compare with its folder label."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, item in enumerate(items, start=1):
        try:
            # Ground truth is deliberately not an argument to this function.
            prediction = predict_binary_image(bundle, item.path, item.content_type)
            correct = prediction["predicted_index"] == item.true_index
            row = {
                "file": str(item.path.relative_to(Path(__file__).resolve().parent)),
                "true_index": item.true_index,
                "true_label": item.true_label,
                **prediction,
                "correct": correct,
            }
            rows.append(row)
            status = "CORRECT" if correct else "WRONG"
            print(
                f"[{index:03d}/{len(items):03d}] {item.path.name:<20} "
                f"predicted={prediction['predicted_label']:<10} "
                f"P(NAFLD)={prediction['prob_nafld']:.6f} | "
                f"actual={item.true_label:<10} -> {status}",
                flush=True,
            )
        except Exception as exc:  # retain the failure and continue the audit
            failure = {"file": str(item.path), "error": str(exc)}
            failures.append(failure)
            print(
                f"[{index:03d}/{len(items):03d}] {item.path.name:<20} ERROR: {exc}",
                flush=True,
            )
    return rows, failures


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "file",
        "true_label",
        "predicted_label",
        "correct",
        "prob_non_nafld",
        "prob_nafld",
        "threshold",
        "margin_from_threshold",
        "fold_std",
        "fold_prob_nafld",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fields}
            output["fold_prob_nafld"] = json.dumps(output["fold_prob_nafld"])
            writer.writerow(output)


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    metrics = summary["metrics"]
    cm = metrics["confusion_matrix"]
    mistakes = sorted(
        (row for row in rows if not row["correct"]),
        key=lambda row: abs(row["margin_from_threshold"]),
        reverse=True,
    )

    lines = [
        "# V3 Binary External Image Evaluation",
        "",
        f"Generated: {summary['created_at_utc']}",
        "",
        "> Research evaluation only. These folder labels are treated as ground truth,",
        "> but this report does not establish their provenance, patient independence,",
        "> or clinical reference standard.",
        "",
        "## Setup",
        "",
        f"- Images evaluated: **{metrics['evaluated_images']}**",
        f"- Decode failures: **{len(failures)}**",
        f"- True Non-NAFLD: **{metrics['class_counts']['Non-NAFLD']}**",
        f"- True NAFLD: **{metrics['class_counts']['NAFLD']}**",
        f"- Single-frame NAFLD threshold: **{summary['binary_threshold']:.10f}**",
        f"- Model version: **{summary['model_version']}**",
        f"- Contract SHA-256: `{summary['contract_sha256']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Accuracy | {pct(metrics['accuracy'])} |",
        f"| Balanced accuracy | {pct(metrics['balanced_accuracy'])} |",
        f"| NAFLD sensitivity/recall | {pct(metrics['sensitivity_nafld_recall'])} |",
        f"| Normal specificity/recall | {pct(metrics['specificity_normal_recall'])} |",
        f"| NAFLD precision | {pct(metrics['precision_nafld'])} |",
        f"| Negative predictive value | {pct(metrics['negative_predictive_value'])} |",
        f"| NAFLD F1 | {pct(metrics['f1_nafld'])} |",
        f"| ROC-AUC | {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'} |",
        "",
        "## Confusion matrix",
        "",
        "| True / predicted | Non-NAFLD | NAFLD |",
        "|---|---:|---:|",
        f"| Non-NAFLD | {cm['tn']} | {cm['fp']} |",
        f"| NAFLD | {cm['fn']} | {cm['tp']} |",
        "",
        f"Correct: **{cm['tn'] + cm['tp']}**; incorrect: **{cm['fp'] + cm['fn']}**.",
        "",
        "## Incorrect predictions",
        "",
    ]
    if mistakes:
        lines.extend(
            [
                "| File | True | Predicted | P(NAFLD) | Threshold | Fold std |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in mistakes:
            lines.append(
                f"| `{row['file']}` | {row['true_label']} | {row['predicted_label']} | "
                f"{row['prob_nafld']:.6f} | {row['threshold']:.6f} | {row['fold_std']:.6f} |"
            )
    else:
        lines.append("No incorrect predictions.")

    if failures:
        lines.extend(["", "## Decode/inference failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['file']}`: {failure['error']}")

    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- The classes are strongly imbalanced, so balanced accuracy, sensitivity,",
            "  and specificity are more informative than accuracy alone.",
            "- Each file is evaluated as a single frame with the frame threshold.",
            "- If multiple files belong to one patient, image-level metrics overstate the",
            "  effective independent sample size; patient IDs are required for a valid",
            "  patient-level analysis.",
            "- External scanner, acquisition, crop, annotation, or compression differences",
            "  can cause domain shift.",
            "- Review `binary_predictions.csv` for every score and fold disagreement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "binary_v3_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = discover_images(data_dir)
    counts = {
        label: sum(item.true_label == label for item in items)
        for label in ("Non-NAFLD", "NAFLD")
    }
    print(
        f"Found {len(items)} images: {counts['Non-NAFLD']} Non-NAFLD, "
        f"{counts['NAFLD']} NAFLD"
    )
    print(f"Loading V3 bundle from {args.bundle_dir.resolve()} ...")
    started = time.perf_counter()
    bundle = ModelBundle(args.bundle_dir)
    load_seconds = time.perf_counter() - started

    inference_started = time.perf_counter()
    rows, failures = run_evaluation(bundle, items)
    inference_seconds = time.perf_counter() - inference_started
    if not rows:
        raise RuntimeError("No images were evaluated successfully.")

    metrics = calculate_metrics(rows)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_use_only": True,
        "evaluation": "V3 binary fold ensemble on labeled external single frames",
        "data_dir": str(data_dir),
        "bundle_dir": str(bundle.bundle_dir),
        "model_version": bundle.model_version,
        "contract_sha256": bundle.contract_sha256,
        "binary_threshold": float(bundle.contract["binary"]["single_frame_threshold"]),
        "folds": len(bundle.binary_models),
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "failed_images": failures,
        "metrics": metrics,
    }

    write_csv(output_dir / "binary_predictions.csv", rows)
    (output_dir / "binary_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown(output_dir / "BINARY_EVALUATION_REPORT.md", summary, rows, failures)

    cm = metrics["confusion_matrix"]
    print("\nBinary evaluation complete")
    print(f"  Evaluated:          {metrics['evaluated_images']}")
    print(f"  Failed:             {len(failures)}")
    print(f"  Accuracy:           {pct(metrics['accuracy'])}")
    print(f"  Balanced accuracy:  {pct(metrics['balanced_accuracy'])}")
    print(f"  NAFLD sensitivity:  {pct(metrics['sensitivity_nafld_recall'])}")
    print(f"  Normal specificity: {pct(metrics['specificity_normal_recall'])}")
    print(f"  ROC-AUC:            {metrics['roc_auc']}")
    print(f"  Confusion matrix:   [[{cm['tn']}, {cm['fp']}], [{cm['fn']}, {cm['tp']}]]")
    print(f"  Report:             {output_dir / 'BINARY_EVALUATION_REPORT.md'}")
    print(f"  Per-image CSV:      {output_dir / 'binary_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
