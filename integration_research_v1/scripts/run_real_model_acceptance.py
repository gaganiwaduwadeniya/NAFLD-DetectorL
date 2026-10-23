#!/usr/bin/env python3
"""Run bounded, local acceptance checks against the selected ten-model release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from pytorch_service.runtime import ResearchRuntime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explanations", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.explanations.mkdir(parents=True, exist_ok=True)

    runtime = ResearchRuntime()
    load_started = time.perf_counter()
    runtime.load()
    records = []
    for index, path in enumerate(args.inputs, start=1):
        payload = path.read_bytes()
        content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        first = runtime.predict(payload, content_type)
        second = runtime.predict(payload, content_type)
        stable = {
            "same_final_stage": first["final_stage"] == second["final_stage"],
            "same_binary_scores": first["binary_probabilities"] == second["binary_probabilities"],
            "same_conditional_scores": first["conditional_stage_probabilities"] == second["conditional_stage_probabilities"],
        }
        explanation, metadata = runtime.explain(payload, content_type)
        explanation_path = args.explanations / f"case_{index:02d}_gradcam.png"
        explanation_path.write_bytes(explanation)
        records.append({
            "case_id": f"case_{index:02d}",
            "privacy_source": "approved_anonymized_consolidation_example",
            "input_filename": path.name,
            "input_sha256": sha256(path),
            "input_bytes": len(payload),
            "first_prediction": first,
            "repeat_check": stable,
            "explanation": {
                "path": str(explanation_path),
                "sha256": sha256(explanation_path),
                "bytes": len(explanation),
                "metadata": metadata,
            },
        })
    result = {
        "test": "selected ten-model local CPU acceptance",
        "load_elapsed_seconds": round(time.perf_counter() - load_started, 3),
        "runtime_status": runtime.status(),
        "cases": records,
        "all_repeat_checks_passed": all(all(case["repeat_check"].values()) for case in records),
        "training_performed": False,
        "external_uploads": False,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "cases": [{"case_id": row["case_id"], "final_stage": row["first_prediction"]["final_stage"], "label": row["first_prediction"]["final_stage_label"]} for row in records],
        "all_repeat_checks_passed": result["all_repeat_checks_passed"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
