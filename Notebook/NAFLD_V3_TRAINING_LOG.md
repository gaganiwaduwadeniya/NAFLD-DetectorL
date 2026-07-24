# NAFLD Evidence Pipeline V3 — Training and Scaling Log

## Purpose

V3 is a new experiment line. It does not replace or modify the completed V2 notebook or its archived artifacts. V2 remains the historical single-split baseline; V3 produces more reliable development evidence from patient-stratified cross-validation and exports fold ensembles for research integration.

This project is for research use only. It is not a clinically validated diagnostic device.

## Versioned artifacts

- Notebook: `Notebook/notebook602ead6518_v3.ipynb`
- Kaggle output root: `/kaggle/working/nafld_v3`
- Local fallback output root: `nafld_v3_outputs`
- Notebook version: `3.0.0`
- V2 evidence remains in `Notebook/notebook602ead6518_v2.ipynb` and the downloaded V2 archive.

Never overwrite V2 outputs with V3 files. After a successful Kaggle run, preserve the entire `nafld_v3` output directory and the executed V3 notebook with cell outputs.

## Why V3 exists

V2 demonstrated that both Keras models could be trained, saved, reloaded, hashed, and used by an end-to-end cascade. Its predictive estimates were unstable because the dataset contains only 55 independent patients and the single validation/test partitions contained very few patients.

V3 addresses the development-evidence problem before changing architecture or combining datasets:

- All 55 patients receive an out-of-fold prediction.
- Five folds are stratified by three composite patient strata.
- Frames from one patient can never appear in more than one fold.
- Binary and grading tasks use the same fold assignment.
- Model selection prioritizes validation patient AUC.
- Validation loss resolves equal patient-AUC results.
- Frame-level and patient-level thresholds are stored separately.
- Binary thresholds target sensitivity before specificity.
- Patient-level bootstrap confidence intervals are reported.
- The complete three-class cascade is evaluated out of fold.
- Every exported fold model is reloaded and hash-verified.

## Dataset and label contract

Expected MATLAB file:

`dataset_liver_bmodes_steatosis_assessment_IJCARS.mat`

Required patient fields:

- `id`
- `class`
- `fat`
- `images`

V3 label mapping:

| Task | Class 0 | Class 1 |
|---|---|---|
| Binary | Non-NAFLD | NAFLD |
| Grading | Mild, fat below 33% | Moderate/severe, fat at or above 33% |

Cascade classes:

1. Non-NAFLD
2. Grade 1 mild
3. Grade 2 moderate/severe

If another dataset uses 30/70 rather than 33/66 boundaries, do not silently concatenate labels. Preserve the original label, reference method, and source-specific grade, then define an explicit harmonization rule. Ambiguous boundary cases should be reviewed or excluded.

## Cross-validation contract

- Split unit: patient
- Folds: 5
- Shuffle: enabled with fixed seed 42
- Stratification: Non-NAFLD / NAFLD mild / NAFLD moderate-severe
- OOF definition: the prediction for a patient must come from the model whose training folds excluded that patient
- Training weights: equalize class contribution at patient level and equalize each patient's total contribution even when frame counts differ

OOF AUC is development evidence. Thresholded OOF metrics are also development estimates because deployment thresholds are chosen from the complete OOF predictions. A separate external cohort must be evaluated after thresholds are locked.

## Training contract

- Backbone: ImageNet EfficientNetB0
- Input: grayscale converted to RGB
- Resize: 224 × 224 bilinear with antialiasing
- Input range: 0–255; no external division by 255
- Phase 1: frozen backbone
- Phase 2: last 20 backbone layers considered for fine-tuning; Batch Normalization remains frozen
- Output: two float32 softmax probabilities
- Loss: sparse categorical cross-entropy
- Optimizer: Adam
- Patient-balanced per-frame sample weights
- Early stopping and checkpointing monitor validation loss
- Phase selection: patient AUC, then lower validation loss, then frame AUC

Training augmentation is inference-safe and included in each saved model. It contains limited horizontal flipping, rotation, translation, zoom, contrast variation, and Gaussian noise. Augmentation layers are inactive during inference.

## Multi-GPU contract

V3 creates `tf.distribute.MirroredStrategy` when more than one GPU is detected. The configured batch size is per replica, and the notebook prints the replica count and computed global batch size before training. Mixed precision is enabled when a GPU is available. All model construction and compilation occurs within the distribution strategy scope.

The run should show two synchronous replicas on a Kaggle T4 ×2 session. One replica means the notebook can still run, but the expected multi-GPU acceleration was not active.

## Evidence generated

Dataset evidence:

- `reports/dataset_audit.png`
- `reports/patient_frame_montage.png`
- `tables/patient_audit.csv`
- `tables/frame_audit.csv`
- `tables/cross_patient_duplicates.csv`
- `tables/patient_fold_manifest.csv`

Per-task evidence:

- Fold training-history plots
- Fold summary CSV files
- OOF frame predictions
- OOF patient predictions
- Frame ROC, precision-recall, reliability, confusion, score, and fold-AUC plots
- Patient ROC, precision-recall, reliability, confusion, score, and fold-AUC plots
- Patient-bootstrap 95% confidence intervals

Cascade evidence:

- `reports/cascade_oof_evidence.png`
- `tables/cascade_oof_frame_predictions.csv`
- `tables/cascade_oof_patient_predictions.csv`
- Frame and patient three-class confusion matrices
- Accuracy, balanced accuracy, and macro F1

## Exported inference artifacts

- Five binary fold models
- Five grading fold models
- Phase checkpoints for auditability
- `inference_contract_v3.json`
- `experiment_report_v3.json`
- `MODEL_CARD_V3.md`
- `ensemble_smoke_test_v3.json`

Inference averages the five fold softmax vectors for a task. A single uploaded frame uses the frame threshold. An examination containing multiple frames should average probabilities across frames first and use the patient threshold.

The grading ensemble runs only when the binary ensemble score reaches the corresponding binary threshold.

## Completion gates

A run is complete only when all of the following hold:

- Dataset audit finishes without conflicting cross-patient duplicate labels.
- Every patient has exactly one fold assignment.
- Every fold contains all three composite strata.
- Both classes are present in every task training and validation fold.
- Binary OOF coverage includes all 55 patients.
- Grading OOF coverage includes every NAFLD patient.
- No patient/frame has duplicate OOF predictions.
- All probabilities have shape `(N, 2)` and sum to one.
- Contract and experiment JSON files reload successfully.
- Every recorded model SHA-256 matches its downloaded file.
- Every fold model reloads with input shape `(224, 224, 3)` and output shape `(2,)`.
- The last cell prints `All fold models reloaded; end-to-end ensemble smoke test: PASSED`.

## External-dataset scaling rules

Do not add an external dataset until the current-data V3 run is complete. For every future source, record:

- Stable patient identifier
- Dataset and institution identifier
- Scanner/device identifier when available
- Original image or study identifier
- Original label and grading system
- Label reference: biopsy, MRI-PDFF, radiologist, ultrasound diagnosis, or other
- Harmonized binary and grade labels
- License and permitted use

If reliable patient identifiers are unavailable, use the images only for representation pretraining. Never use an image-level random split to claim patient-level generalization.

Recommended experiment sequence:

1. V3 current-data cross-validation baseline.
2. Audited cropping/preprocessing experiment.
3. Controlled augmentation experiment.
4. Shared multi-task or ordinal-head experiment.
5. External pretraining with source tracking.
6. Domain-balanced pooled training.
7. Locked-threshold external validation on a completely untouched patient cohort.

Change one major factor at a time and retain all fold manifests and OOF predictions.

## Run record template

### Identity

- Date:
- Owner:
- Kaggle notebook version:
- Git commit:
- Dataset SHA-256:
- TensorFlow version:
- GPUs and replica count:

### Controlled change

- Hypothesis:
- Baseline being compared:
- Preprocessing change:
- Augmentation change:
- Architecture change:
- Loss/weighting change:
- Dataset change:

### Binary OOF result

- Frame AUC:
- Patient AUC and 95% CI:
- Patient sensitivity and 95% CI:
- Patient specificity and 95% CI:
- Frame threshold:
- Patient threshold:
- Fold AUC mean ± standard deviation:

### Grading OOF result

- Frame AUC:
- Patient AUC and 95% CI:
- Patient sensitivity and 95% CI:
- Patient specificity and 95% CI:
- Frame threshold:
- Patient threshold:
- Fold AUC mean ± standard deviation:

### Cascade result

- Frame balanced accuracy:
- Frame macro F1:
- Patient balanced accuracy:
- Patient macro F1:
- Dominant false-negative pattern:
- Dominant false-positive pattern:

### Integrity

- Fold coverage checks passed:
- JSON validation passed:
- All hashes passed:
- All models reloaded:
- Smoke test passed:

### Decision

- Accept or reject:
- Reason:
- Next experiment:
- External validation status:

## Decision history

| Date | Decision | Reason |
|---|---|---|
| 2026-07-23 | Freeze V2 as historical evidence | Its artifacts, contract, plots, and hashes were archived successfully |
| 2026-07-23 | Create V3 as a new notebook | Preserve evidence and avoid retroactively changing the completed baseline |
| 2026-07-23 | Use five shared composite patient folds | Increase patient-level development evidence without frame leakage |
| 2026-07-23 | Export fold ensembles | Reduce dependence on one unstable small split and retain auditable fold models |
| 2026-07-23 | Delay external-data merging | Establish a trustworthy current-data baseline and source-aware ingestion contract first |
