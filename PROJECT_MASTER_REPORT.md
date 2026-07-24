# NAFLD Detector — Master Project Report and Agent Handoff

Last updated: 2026-07-24

Current model line: V3 (`3.0.0`)

Current integration status: Phase 1 complete; Phase 2 is next

Project status: research prototype only; not clinically validated

## 1. Why this document exists

This is the main technical handoff for the entire repository. A developer or
chat agent should read this file before changing model inference, the Express
API, the React UI, the database schema, or the training pipeline.

The source-of-truth files are:

- `PROJECT_MASTER_REPORT.md` — overall architecture, history, current status,
  and implementation phases.
- `nafld_v3/inference_contract_v3.json` — machine-readable V3 preprocessing,
  model, class, threshold, hash, and evidence contract.
- `nafld_v3/MODEL_CARD_V3.md` — concise model limitations and evidence.
- `Notebook/NAFLD_V3_TRAINING_LOG.md` — detailed V3 training methodology.
- `Notebook/notebook602ead6518_v3.ipynb` — executed V3 training notebook.
- `ml_service/README.md` — Python inference-service setup.

V1 and V2 are historical and have been moved into `archive/`. They must not be
used by the active inference service.

## 2. Project introduction

NAFLD Detector is a full-stack research decision-support application for
analysing liver B-mode ultrasound frames. Its intended workflow is:

1. A clinician enters basic patient demographics.
2. The clinician uploads a liver ultrasound image frame.
3. A binary model estimates Non-NAFLD versus NAFLD.
4. If the binary result reaches the NAFLD threshold, a grading model estimates
   mild versus moderate/severe steatosis.
5. The application displays and stores the result for later review.

The final intended inference cascade is:

```text
Uploaded ultrasound frame
        |
        v
Five-fold binary ensemble
        |
        +-- score below threshold --> Non-NAFLD; stop
        |
        +-- score at/above threshold
                    |
                    v
          Five-fold grading ensemble
                    |
                    +-- below threshold --> Grade 1 mild
                    +-- at/above threshold --> Grade 2 moderate/severe
```

The system is not a medical device and has not received external clinical
validation. It must be presented as a research prototype, not as an autonomous
diagnostic system.

## 3. Technology and runtime architecture

### Frontend

- React 19
- TypeScript
- Vite
- Tailwind CSS 4
- Recharts
- React Router

### Application backend

- Express 4
- JWT authentication
- Multer in-memory uploads
- Local JSON database fallback
- Optional Firebase/Firestore/Storage integration

### Machine-learning service

- Python 3.12
- Flask 3.1.3
- TensorFlow 2.20.0
- Keras V3 `.keras` model files
- Five binary fold models and five grading fold models

### Intended request path

```text
React browser
  -> authenticated Express POST /api/predict
  -> internal Flask POST /api/predict
  -> TensorFlow V3 ensemble
  -> validated result returned to Express
  -> Express stores image/result metadata
  -> React displays result and history
```

The `.keras` files must remain server-side. They must not be bundled into the
React frontend.

## 4. Active repository map

```text
G:\NAFLD-DetectorL
|-- PROJECT_MASTER_REPORT.md       This document
|-- README.md                      Application overview
|-- server.ts                      Express API and current mock prediction path
|-- src/                           React UI, types, auth, and database helpers
|-- ml_service/                    Python V3 inference service
|-- nafld_v3/
|   |-- models/                    Ten canonical fold models
|   |-- reports/                   Training/evaluation evidence plots
|   |-- tables/                    Fold, OOF, audit, and cascade tables
|   |-- inference_contract_v3.json
|   |-- experiment_report_v3.json
|   |-- MODEL_CARD_V3.md
|   `-- ensemble_smoke_test_v3.json
|-- Notebook/
|   |-- notebook602ead6518_v3.ipynb
|   `-- NAFLD_V3_TRAINING_LOG.md
|-- archive/                       Historical V1/V2 material; Git-ignored
|-- data/db.json                   Local fallback users and scan records
`-- .venv/                         Local Python environment; Git-ignored
```

The V3 phase-training checkpoints were removed during cleanup. Only the ten
canonical hash-verified models required by the inference contract remain.

## 5. Dataset and label definition

### Source dataset

- File: `dataset_liver_bmodes_steatosis_assessment_IJCARS.mat`
- Dataset SHA-256:
  `912dc5367e06ccf24d8ba33e503d7b755b92ba84a6840cb232c92096dbe936f7`
- Independent patients: 55
- Frames: 550
- Frames per patient: 10
- Source modality: grayscale B-mode liver ultrasound

Required MATLAB patient fields:

- `id`
- `class`
- `fat`
- `images`

### Active V3 task definitions

| Task | Output 0 | Output 1 | Positive output |
|---|---|---|---|
| Binary | `Non-NAFLD` | `NAFLD` | Index 1, `NAFLD` |
| Grading | `Grade1_Mild` | `Grade2_Moderate_Severe` | Index 1, moderate/severe |

Grading includes only patients whose source binary `class` is NAFLD. The grade
rule is:

- Mild: fat below 33 percent
- Moderate/severe: fat at or above 33 percent

Do not derive the binary target from the fat value. The original dataset
`class` field is the binary source of truth.

## 6. Model development history: V1 to V3

### 6.1 V1 — initial two-model prototype

Historical notebook:
`archive/Notebook/notebook602ead6518.ipynb`

V1 established the basic idea:

- Train two EfficientNetB0 classifiers.
- Stage 1: binary NAFLD detection.
- Stage 2: mild versus moderate/severe grading.
- Use a frozen-backbone phase followed by fine-tuning of the final 20 layers.
- Save `.h5` models.

V1 used directory generators with an 80/20 frame split, heavy image
augmentation, class weights, and ImageNet EfficientNetB0.

#### V1 problems

1. **Double rescaling**: `ImageDataGenerator(rescale=1./255)` divided images by
   255 even though EfficientNetB0 already performs input rescaling internally.
2. **Frame-level leakage risk**: ten correlated frames from one patient could
   appear in both training and validation.
3. **Class-order ambiguity**: binary directory ordering was
   `{'NAFLD': 0, 'Non-NAFLD': 1}`, while later pipeline expectations use
   Non-NAFLD at index 0 and NAFLD at index 1.
4. **Misleading AUC**: the reported binary AUC used output index 1, which was
   Non-NAFLD in V1, not NAFLD.
5. **Class collapse**: argmax predictions selected only one class for each
   task.
6. **No machine-readable inference contract or saved-model hash manifest**.
7. **No patient-level test evidence**.

#### V1 recorded numbers

| Task | Validation frames | ROC-AUC | Accuracy | Per-class behaviour |
|---|---:|---:|---:|---|
| Binary | 110 | 0.8518 | 0.6909 | NAFLD recall 1.0000; Non-NAFLD recall 0.0000 |
| Grading | 76 | 0.8778 | 0.5263 | Mild recall 1.0000; moderate/severe recall 0.0000 |

The AUC values showed ranking information, but the decision rule was unusable
because both classifiers collapsed to a single predicted class. The results
were also based on a frame split and must not be treated as patient-level
generalization.

### 6.2 V2 — corrected training and evidence baseline

Historical assets:

- `archive/Notebook/notebook602ead6518_v2.ipynb`
- `archive/Notebook/NAFLD_V2_TRAINING_LOG.md`
- `archive/nafld_v2/`

V2 corrected the main V1 engineering failures:

- Loaded the original patient-based MATLAB structure directly.
- Used patient-level train/validation/test partitions: 38/8/9 patients.
- Stratified patients by Non-NAFLD, mild NAFLD, and moderate/severe NAFLD.
- Removed external division by 255.
- Standardized input to grayscale -> RGB, 224 x 224, float32, range 0–255.
- Added separate binary and grading thresholds selected on validation data.
- Added frame-level and patient-level evidence.
- Added a two-stage cascade contract.
- Saved native `.keras` models rather than only legacy `.h5` files.
- Reloaded models after saving and recorded SHA-256 hashes.
- Added evidence plots and a machine-readable V2 inference contract.
- Supported Kaggle T4 x2 through `MirroredStrategy`.

V2 retained an EfficientNetB0 backbone but used a more regularized head:

```text
EfficientNetB0
  -> global average pooling
  -> Dense 128 + ReLU + L2
  -> batch normalization
  -> dropout 0.40
  -> Dense 64 + ReLU + L2
  -> dropout 0.30
  -> Dense 2 softmax
```

#### V2 patient-level test numbers

| Task | Patients | Threshold | ROC-AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary | 9 | 0.826679 | 0.8333 | 0.6667 | 0.7500 | 0.5000 | 1.0000 |
| Grading | 6 NAFLD | 0.568374 | 0.6667 | 0.5000 | 0.5000 | 0.6667 | 0.3333 |

V2 demonstrated a loadable and auditable pipeline, but the tiny single split
was unstable. Binary validation patient AUC was 0.5000 with zero sensitivity,
while grading validation was perfectly separated and then fell to 0.6667 AUC
on test. These swings showed that one validation/test split was too small for
reliable development decisions.

### 6.3 V3 — patient-level cross-validation and fold ensembles

Active assets:

- `Notebook/notebook602ead6518_v3.ipynb`
- `Notebook/NAFLD_V3_TRAINING_LOG.md`
- `nafld_v3/`

V3 did not primarily chase a larger architecture. It improved the reliability
and coverage of development evidence:

- Five patient-stratified folds with seed 42.
- Every one of the 55 patients receives an out-of-fold binary prediction.
- Every NAFLD patient receives an out-of-fold grading prediction.
- Frames from a patient never cross fold boundaries.
- Binary and grading tasks share the same fold assignment.
- Patient-balanced per-frame sample weights.
- Phase selection by validation patient AUC, then validation loss and frame
  AUC as tie-breakers.
- Five binary and five grading canonical models.
- Inference averages fold softmax probabilities arithmetically.
- Separate frame and patient thresholds.
- Binary thresholds selected to maximize specificity while maintaining at
  least 0.90 development sensitivity.
- Grading thresholds selected using Youden J / balanced accuracy.
- Patient bootstrap 95 percent confidence intervals.
- Complete three-class cascade evaluation.
- Dataset, duplicate, frame, fold, OOF, and calibration evidence.
- Reload and hash verification for every fold model.

Training still uses EfficientNetB0, two-phase transfer learning, inference-safe
augmentation, frozen BatchNormalization during fine-tuning, and no external
pixel rescaling.

## 7. V3 quantitative evidence

### 7.1 Binary model

| Level | Threshold | ROC-AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frame OOF | 0.553681 | 0.8702 | 0.7964 | 0.7324 | 0.9000 | 0.5647 | 0.8593 |
| Patient OOF | 0.503990 | 0.8824 | 0.8182 | 0.7546 | 0.9211 | 0.5882 | 0.8750 |

Patient bootstrap 95 percent intervals:

- ROC-AUC: 0.7817–0.9644
- Balanced accuracy: 0.6238–0.8824
- Sensitivity: 0.8158–1.0000
- Specificity: 0.3529–0.8235

Patient confusion matrix:

```text
                 Pred Non-NAFLD   Pred NAFLD
True Non-NAFLD         10              7
True NAFLD              3             35
```

### 7.2 Grading model

| Level | Threshold | ROC-AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frame OOF | 0.607806 | 0.7327 | 0.6895 | 0.6881 | 0.6611 | 0.7150 | 0.6685 |
| Patient OOF | 0.593562 | 0.7583 | 0.7105 | 0.7139 | 0.7778 | 0.6500 | 0.7179 |

Patient bootstrap 95 percent intervals:

- ROC-AUC: 0.6000–0.8972
- Balanced accuracy: 0.5638–0.8612
- Sensitivity: 0.5556–0.9444
- Specificity: 0.4500–0.8500

Patient confusion matrix:

```text
                         Pred Mild   Pred Moderate/Severe
True Mild                    13               7
True Moderate/Severe          4              14
```

### 7.3 Full cascade

| Level | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Frame OOF | 0.5836 | 0.5853 | 0.5899 |
| Patient OOF | 0.6182 | 0.6220 | 0.6208 |

Patient cascade confusion matrix, ordered as Non-NAFLD, mild, and
moderate/severe:

```text
[[10, 5, 2],
 [ 3,10, 7],
 [ 0, 4,14]]
```

### 7.4 What improved numerically

The evaluation regimes changed between versions, so the following differences
are engineering indicators rather than controlled clinical performance deltas.

- V1 binary balanced accuracy was effectively 0.5000 because Non-NAFLD recall
  was zero. V3 patient OOF balanced accuracy is 0.7546.
- V1 grading balanced accuracy was effectively 0.5000 because
  moderate/severe recall was zero. V3 patient OOF balanced accuracy is 0.7139.
- Compared with the V2 nine-patient binary test, V3 OOF patient AUC increased
  from 0.8333 to 0.8824 and sensitivity from 0.5000 to 0.9211. Specificity
  decreased from 1.0000 to 0.5882 because V3 intentionally selected a
  sensitivity-oriented threshold and evaluated all 55 patients.
- Compared with the V2 six-patient grading test, V3 OOF patient AUC increased
  from 0.6667 to 0.7583, balanced accuracy from 0.5000 to 0.7139, sensitivity
  from 0.6667 to 0.7778, and specificity from 0.3333 to 0.6500.
- V3 provides predictions for the full development cohort rather than making
  conclusions from only nine binary test patients and six grading patients.

V3 is a stronger research baseline, but the confidence intervals remain wide
and the complete cascade balanced accuracy is only 0.6220. This is not evidence
of clinical readiness.

### 7.5 External folder-based binary test — 2026-07-24

This test was added after Phases 1–6 to measure how the unchanged V3 binary
ensemble behaves on images outside the original 55-patient development data.
It is the first local evidence that directly exposes the model to a different
image collection.

Test assets:

- Evaluation script: `Test_Data/evaluate_binary_v3.py`
- Human-readable report:
  `Test_Data/binary_v3_results/BINARY_EVALUATION_REPORT.md`
- Machine-readable summary:
  `Test_Data/binary_v3_results/binary_summary.json`
- Per-image predictions:
  `Test_Data/binary_v3_results/binary_predictions.csv`
- Copies of correctly classified images: `Test_Data/passed/`

The evaluator loaded and hash-verified the full V3 bundle. Each image was
decoded and processed independently using the production preprocessing path.
The model received only image pixels; the containing folder was consulted
after inference to score the result. The five binary fold probabilities were
averaged, and the locked single-frame threshold `0.5536812544` was applied.

Folder labels were mapped as follows:

| Folder | Evaluation label |
|---|---|
| `Test_Data/normal/` | `Non-NAFLD` |
| `Test_Data/NFLD/` | `NAFLD` |

#### External-test results

| Measure | Result |
|---|---:|
| Images | 87 |
| Decode/inference failures | 0 |
| Non-NAFLD images | 11 |
| NAFLD images | 76 |
| Correct images | 66 |
| Incorrect images | 21 |
| Accuracy | 0.7586 |
| Balanced accuracy | 0.5120 |
| NAFLD sensitivity/recall | 0.8421 |
| Normal specificity/recall | 0.1818 |
| NAFLD precision | 0.8767 |
| Negative predictive value | 0.1429 |
| NAFLD F1 | 0.8591 |
| ROC-AUC | 0.6017 |

Confusion matrix:

```text
                 Pred Non-NAFLD   Pred NAFLD
True Non-NAFLD          2              9
True NAFLD             12             64
```

The raw accuracy of 75.86 percent is inflated by the class distribution: 76 of
87 images are labelled NAFLD. Balanced accuracy is only 51.20 percent, close to
chance, because the model incorrectly calls 9 of the 11 normal images NAFLD.
The mean predicted NAFLD probability is `0.6491` for the normal folder and
`0.6863` for the NAFLD folder, so the scores have weak separation on this
collection. This is consistent with a substantial distribution, acquisition,
cropping, or label-definition mismatch rather than merely a slightly imperfect
threshold.

Compared with V3 patient-level OOF development evidence:

| Measure | V3 development OOF | External folder test |
|---|---:|---:|
| ROC-AUC | 0.8824 | 0.6017 |
| Balanced accuracy | 0.7546 | 0.5120 |
| Sensitivity | 0.9211 | 0.8421 |
| Specificity | 0.5882 | 0.1818 |

These values are not a controlled head-to-head comparison. The development
figures are patient-aggregated OOF results, whereas the external figures are
single-image results with unknown patient grouping. Nevertheless, the drop is
large enough that deployment claims must remain blocked.

#### What this test does not prove

- Folder names are treated as labels, but label provenance and clinical
  reference standards have not yet been audited.
- It is unknown whether several files belong to the same patient. Therefore,
  87 images must not be described as 87 independent patients.
- Scanner, site, acquisition view, compression, crop, overlay, demographics,
  and disease-severity metadata are not yet recorded.
- There are only 11 normal images, so specificity has high sampling
  uncertainty.
- The threshold must not be retuned on this set and then reported on the same
  set as external performance.

#### Rules for `Test_Data/passed/`

`Test_Data/passed/` contains the 66 correctly classified copies: 2 normal and
64 NAFLD. It exists for visual error analysis and demonstration only.

Do **not** train on only the `passed` folder. That would select data using the
current model's answer, remove hard cases and nearly all normal examples, and
produce a biased model that appears better without learning the actual target.
Do not treat the passed copies as new independent data; they duplicate files
already present under `Test_Data/`.

Before any retraining decision, manually compare the 21 failures, 66 passes,
and original training frames for ultrasound sector shape, liver visibility,
gain, depth, overlays, crop, resolution, and diagnostic-label compatibility.

## 8. Active V3 inference contract

### Input preprocessing

| Property | Required value |
|---|---|
| Source | One grayscale or colour ultrasound frame |
| Output shape | `(1, 224, 224, 3)` |
| Tensor dtype | `float32` |
| Pixel range | 0–255 |
| Colour handling | Convert to grayscale, then replicate to RGB |
| Resize | TensorFlow bilinear with antialiasing |
| External division by 255 | No |
| Crop fractions | All zero in V3 |

EfficientNetB0 performs rescaling inside every saved model. An inference service
that divides by 255 externally will violate the training contract.

### Ensemble rules

- Load five binary fold models and five grading fold models.
- Call every model with `training=False`.
- Average the five softmax vectors for a task.
- Validate output shape `(N, 2)`, finite values, and row sums near 1.
- A single uploaded image must use the frame thresholds.
- A multi-frame examination must first average frame probabilities and then use
  the patient thresholds.
- Do not run grading when the binary score is below its corresponding
  threshold.

### Thresholds

| Task | Single-frame threshold | Patient/multi-frame threshold |
|---|---:|---:|
| Binary NAFLD | 0.5536812544 | 0.5039900541 |
| Moderate/severe grading | 0.6078062057 | 0.5935616493 |

## 9. Model upgrade plan

Integration of V3 should be completed before training another architecture.
After integration, model improvements should be controlled experiments against
the unchanged V3 folds and OOF baseline.

### Upgrade M1 — image-region and shortcut audit

Goal: determine whether the model is learning liver tissue or scanner/display
artifacts.

- Detect and remove black borders.
- Mask text, measurements, patient details, and scanner overlays.
- Crop to the valid ultrasound sector or a clinically meaningful liver ROI.
- Keep the V3 folds, architecture, weighting, and metrics unchanged.
- Compare fold-by-fold OOF performance, not only one aggregate score.
- Retain the change only if patient-level improvement is consistent.

Primary target: improve binary specificity without materially reducing the
current 0.9211 sensitivity.

### Upgrade M2 — patient-level multi-frame prediction

The training dataset contains ten frames per patient, but the current UI accepts
one frame. A later UI should accept an examination or a selected frame set.

- Predict every frame with the fold ensemble.
- Aggregate frame probabilities by arithmetic mean to match V3 evidence.
- Also report median, interquartile range, and fold/frame disagreement as
  uncertainty indicators.
- Use patient thresholds only after aggregation.
- Reject or flag studies with too few valid frames or severe disagreement.

### Upgrade M3 — calibration and locked thresholds

- Measure calibration on a genuinely independent cohort.
- If sufficient data exists, compare temperature scaling or another simple
  calibration method fitted without touching the external test set.
- Lock thresholds before external evaluation.
- Record calibration version and threshold origin in the inference contract.

### Upgrade M4 — external data or prospective collection

The main limitation is 55 independent patients, not model size.

- Prefer new patients with stable patient IDs, source/site/device metadata, and
  a documented diagnostic reference.
- Use a new cohort as external validation before adding it to training.
- Never perform random image-level splitting when several images belong to one
  patient.
- Do not merge datasets with missing patient grouping, uncertain labels, or
  incompatible grading definitions.

Two explored public-data routes were rejected:

- Processed BeHSoF images lacked a reliable accessible patient/image mapping
  for safe patient-level splitting.
- A later `normal/benign/malignant` dataset visually represented focal liver
  lesions and contained image/spreadsheet mapping errors; it was not suitable
  for NAFLD grading and was removed from the workspace.

### Upgrade M5 — grading architecture

With enough additional patients, compare:

- Shared multi-task binary and grading features.
- A three-class mild/moderate/severe softmax baseline.
- An ordinal model because steatosis grades are ordered.

Do not split moderate and severe with the current 55-patient dataset unless a
patient-count audit shows adequate representation in every evaluation fold.

### Upgrade M6 — external clinical validation

Before any clinical claim:

- Evaluate locked models and thresholds on untouched patients.
- Report source/site-specific results.
- Report sensitivity, specificity, calibration, failure rate, and confidence
  intervals.
- Review false positives and false negatives with a domain expert.
- Document inclusion/exclusion criteria and reference-standard quality.

### Dataset upgrade plan after the external binary test

The external result changes the immediate priority. The next gain is more
likely to come from reliable, diverse patient data and acquisition controls
than from replacing EfficientNetB0 with a larger network.

#### D1 — audit and freeze the current external set

Create one metadata row per original file, not per copy, with:

- stable image ID and cryptographic hash;
- patient/study ID where recoverable;
- binary label and exact label source;
- reference standard: biopsy, CAP/elastography, radiologist assessment,
  clinical report, or unknown;
- acquisition site, scanner vendor/model, probe, frequency, depth, gain, view,
  date, and image dimensions where available;
- crop/overlay/annotation status;
- inclusion/exclusion decision and reason;
- license and permitted research use.

Deduplicate by file hash and perceptual similarity across the original MATLAB
data, `Test_Data`, and any future source. The same patient or near-duplicate
frame must never cross training, validation, and test partitions.

Until that audit is complete, freeze `Test_Data` as an exploratory external
test. Do not tune the model, threshold, augmentation, or ROI rules against its
labels. If it is intentionally converted into training/validation data, record
that change and acquire a new untouched external test cohort.

#### D2 — collect more independent normal patients first

The external set has only 11 normal images and V3 misclassifies 9 of them. The
highest-priority acquisition is therefore diverse, confidently labelled normal
patients—not more frames from the same NAFLD cases.

Collection targets should be defined in patients and studies, not images:

- multiple sites and scanner vendors;
- balanced sex and broad age/BMI ranges;
- relevant confounders and common non-steatotic liver appearances;
- consistent required views plus several frames per view;
- a documented clinical reference standard;
- enough normal, mild, moderate, and severe patients to populate every split.

Do not fabricate a universal minimum sample size. Before collection, perform a
power/precision analysis for the desired confidence-interval width around
sensitivity and specificity. A useful practical milestone is an untouched
external cohort with enough independent positive and negative patients that
both class-specific confidence intervals are clinically interpretable.

#### D3 — store examinations, not isolated screenshots

The next dataset format should be hierarchical:

```text
patient
  -> examination/date/site/scanner
       -> standard acquisition view
            -> frame or short cine sequence
```

Keep all frames, including difficult but valid frames. Add frame-quality labels
such as liver visible, kidney visible, valid acoustic window, overlay present,
and severe artifact. This supports quality control and multi-frame aggregation
without pretending correlated frames are independent patients.

Large liver-steatosis work has reported repeatable assessment using multiple
images per viewpoint and validation across scanners, which supports moving this
project from one screenshot toward standardized multi-view examinations.

#### D4 — improve the reference target

Whenever possible, retain continuous or ordinal reference information rather
than only folder labels:

- histologic steatosis percentage and biopsy timing;
- CAP/attenuation measurement and timing;
- expert grade with reader identity and adjudication;
- clinical laboratory/demographic variables with consent and governance.

Store uncertainty and missingness in the label rather than forcing uncertain
cases into Normal or NAFLD. Maintain separate targets for imaging steatosis,
NAFLD/MASLD clinical diagnosis, and severity; these are related but not
interchangeable labels.

#### D5 — use unlabeled ultrasound data safely

If many lawful ultrasound frames can be obtained without reliable steatosis
labels, use them only for ultrasound-domain self-supervised pretraining or
quality/ROI learning. Patient grouping and source metadata are still required.
Unlabeled images do not solve external validation and must not enter the locked
test set.

### Advanced V4 model upgrade program

Each candidate below must be evaluated against the unchanged V3 patient folds
and the same primary patient-level metrics. Change one major factor at a time,
save OOF predictions, and report fold-level deltas and confidence intervals.

#### M7 — quality gate, ultrasound-sector extraction, and liver ROI

Build an explicit preprocessing experiment before a larger classifier:

1. Detect whether the upload is a valid liver B-mode image.
2. Mask text, measurements, logos, borders, and identifying overlays.
3. Extract the ultrasound sector.
4. Localize a liver parenchyma ROI; when reliable, include a hepatorenal ROI or
   view indicator.
5. Reject or abstain when no valid liver ROI is available.

Compare whole-frame, masked-sector, ROI-only, and whole-frame-plus-ROI inputs.
Use saliency/occlusion checks only as supporting audits, not proof of clinical
reasoning. The external normal false positives should be reviewed first for
scanner or overlay shortcuts.

#### M8 — ultrasound-domain self-supervised pretraining

ImageNet pretraining is convenient but not ultrasound-specific. Pretrain an
encoder on a larger patient-separated collection of lawful unlabeled B-mode
frames using a contrastive or masked-image objective, then fine-tune on the
steatosis labels.

Required controls:

- ImageNet-initialized V3-equivalent baseline;
- identical downstream folds and classifier head;
- no test-patient images during self-supervised pretraining;
- patient-aware sampling so ten nearby frames are not treated as ten unrelated
  examples;
- robustness tests for gain, depth, scanner, compression, and view.

#### M9 — examination-level multi-instance learning

Once several frames/views per examination are available, compare:

- mean probability aggregation, which is the current evidence-backed baseline;
- median/trimmed aggregation;
- learned attention pooling over frame embeddings;
- view-aware pooling with a missing-view mask.

Train and score at the patient/examination level. The aggregator must not learn
from frame labels copied from the patient as though every frame visibly proves
the disease. Report per-frame disagreement and permit abstention when the study
is internally inconsistent.

#### M10 — ordinal and multi-task learning

With adequate patient counts and label quality, replace the disconnected
binary/grading pipeline experiment with a shared encoder and related heads:

- steatosis presence;
- ordered grade or continuous fat percentage;
- image/view quality;
- optional scanner/site adversarial head for domain robustness.

Compare ordinary softmax, ordinal thresholds, and continuous regression. A
multi-task model is retained only if it improves locked patient-level binary
specificity and grading performance without degrading sensitivity. Do not add
age, sex, BMI, laboratory values, or scanner metadata unless they are available
consistently at inference and their missingness is handled explicitly.

#### M11 — domain robustness, OOD detection, and abstention

The external score shift means the system needs a safe failure mode, not just a
forced label. Evaluate:

- source-balanced sampling and scanner/style augmentation;
- leave-one-site or leave-one-scanner-out validation;
- feature-distance or density-based out-of-distribution scoring;
- ensemble disagreement plus input-quality signals;
- a predeclared abstention rule returning `Unable to assess`.

Do not call fold standard deviation calibrated uncertainty. Select any
abstention threshold on development/validation data, lock it, then report
coverage and performance on untouched external patients.

#### M12 — calibration and threshold policy

ROC-AUC `0.6017` shows that threshold adjustment alone cannot repair the weak
external ranking. First improve data compatibility and representation. Then:

- fit temperature scaling or another simple calibrator on an independent
  calibration partition;
- report reliability diagrams, Brier score, log loss, and class-specific
  calibration alongside discrimination;
- select thresholds from an explicit clinical operating objective;
- lock calibration and thresholds before external evaluation;
- version them in the inference contract.

Calibration estimates are themselves unreliable with tiny samples, so avoid
complex calibrators until substantially more independent patients are
available.

### Recommended experiment order

| Priority | Experiment | Reason |
|---:|---|---|
| 1 | Audit labels, patient IDs, duplicates, licenses, and acquisition metadata | Determines whether the external result is valid and whether data can be reused |
| 2 | Blinded review of all 21 errors plus matched correct images | Identifies label problems, invalid views, overlays, and domain shortcuts |
| 3 | Collect diverse independent normal patients and a new locked external cohort | Directly targets the observed 18.18% specificity and class imbalance |
| 4 | Sector/overlay masking and liver ROI ablation on unchanged V3 folds | Lowest-complexity model-side response to likely shortcut/domain shift |
| 5 | Multi-frame mean aggregation on standardized examinations | Strong baseline before learned aggregation |
| 6 | Ultrasound self-supervised encoder | Uses additional unlabeled data without inventing labels |
| 7 | Multi-instance/view-aware model | Uses examination structure after sufficient data exists |
| 8 | Ordinal/multi-task architecture | Requires stronger grading labels and larger patient counts |
| 9 | Calibration, abstention, and locked external validation | Must use independent data after the representation is stable |

The acceptance gate for a V4 candidate is not higher training accuracy. It must
show repeatable patient-level improvement across folds, especially external
normal specificity, without an unacceptable sensitivity loss, and must pass an
untouched external evaluation with documented provenance.

### Evidence informing the upgrade direction

- Byra et al. used biopsy-referenced liver ultrasound and transfer-learned deep
  features, illustrating the value of a strong reference standard:
  https://pubmed.ncbi.nlm.nih.gov/30094778/
- Li et al. evaluated multi-view liver ultrasound across scanners and reported
  repeatability using multiple images per viewpoint:
  https://pubmed.ncbi.nlm.nih.gov/35979264/
- Ghesu et al. studied large-scale self-supervised medical-image pretraining,
  including ultrasonography:
  https://arxiv.org/abs/2201.01283
- Zhang et al. showed why calibration evaluation can be misleading in
  small-data settings and studied ensemble/post-hoc calibration methods:
  https://proceedings.mlr.press/v119/zhang20k.html

## 10. Current UI and backend behaviour

### Authentication and roles

The application supports:

- Doctor login and registration.
- Admin login.
- JWT sessions issued by Express.
- Doctor-specific scan history.
- Admin access to all scans, user directory, and aggregate statistics.

The local fallback database is `data/db.json`. Firebase can optionally provide
Firestore and image storage when configured.

### Doctor dashboard

`src/pages/DoctorDashboard.tsx` currently:

- Collects patient name, age, and assigned sex.
- Accepts one uploaded image.
- Sends multipart form data to Express `/api/predict`.
- Displays the returned `Scan` through `ResultCard`.
- Shows the uploaded/stored image after a result.
- Saves successful results into history through the Express backend.

### Image uploader

`src/components/ImageUploader.tsx` currently:

- Supports drag-and-drop or file browsing.
- Previews the selected image.
- Accepts JPEG, PNG, and BMP MIME types.
- Enforces a 50 MB client limit.

The UI text currently mentions DICOM, but DICOM is not accepted by the input
element or Express MIME filter. That claim must be removed unless DICOM support
is deliberately implemented.

### Result card

`src/components/ResultCard.tsx` currently expects only:

- `Normal` or `Abnormal`
- One `confidence` percentage
- `Normal` and `Abnormal` percentages

It does not yet support the V3 class names, thresholds, grading result, model
version, fold disagreement, or research-only metadata.

The term `confidence` is misleading. The new UI should say `model probability`
or `ensemble score` unless calibration has been independently established.

### History and scan details

- `PatientHistory.tsx` lists and filters previous doctor scans.
- `ScanDetail.tsx` displays the stored image and reuses `ResultCard`.
- The displayed image resolution is currently hardcoded as 1024 x 768 rather
  than derived from the upload.
- Existing mock records use the legacy schema and should remain readable after
  the schema upgrade.

### Admin dashboard

`AdminDashboard.tsx` and `dbGetAdminStats()` currently summarize:

- Total scans
- Normal count
- Abnormal count
- Recent scans
- Scan counts by day

These calculations must be updated to use the new binary V3 label while
remaining compatible with legacy records.

### Critical current limitation: predictions are still mocked

`server.ts` currently forwards to `ML_SERVICE_URL` when configured, but if the
service is missing, fails, or returns an unexpected response, it silently calls
`generateSimulatedPrediction()`. That function uses the filename and randomness.

The Python service currently returns HTTP 501 from `/api/predict` because real
preprocessing and inference belong to Phase 2.

Therefore, at the time of this report, the frontend does **not** produce real
V3 predictions. Phase 1 proves that the models load; it does not connect them to
uploads yet.

## 11. Frontend/model integration phases

### Phase 1 — model-loading service

Status: **completed and runtime-verified**

Included:

- Created `ml_service/` Flask package.
- Pinned Flask 3.1.3 and TensorFlow 2.20.0.
- Read `inference_contract_v3.json` at startup.
- Required `research_use_only=true` and schema version 3.
- Required the TensorFlow runtime to match the training contract.
- Verified all ten model files and SHA-256 hashes before loading.
- Loaded models with `compile=False`.
- Validated every input shape as `(None, 224, 224, 3)`.
- Validated every output shape as `(None, 2)`.
- Ran warm-up inference and validated finite softmax output.
- Exposed `/health`, `/healthz`, and `/api/health`.
- Failed closed when contract/model validation fails.
- Left `/api/predict` disabled with HTTP 501 until Phase 2.

Completion evidence recorded on 2026-07-23:

```text
schema_version: 3
tensorflow_version: 2.20.0
ready: True
runtime_devices: CPU:/physical_device:CPU:0
binary models: 5
grading models: 5
total models: 10
```

### Phase 2 — exact preprocessing and cascade inference

Status: **completed and runtime-verified**

Primary files:

- `ml_service/inference.py`
- `ml_service/app.py`
- `ml_service/requirements.txt`
- New inference-service tests

Required changes:

1. Accept one multipart field named `image`.
2. Validate MIME type, file signature, non-empty bytes, dimensions, channel
   count, and a safe size limit.
3. Decode JPEG/PNG/BMP.
4. Match notebook preprocessing exactly:
   - colour to grayscale;
   - float32;
   - grayscale to three identical channels;
   - TensorFlow bilinear resize with antialiasing to 224 x 224;
   - preserve 0–255 values;
   - add batch dimension.
5. Run all five binary models with `training=False`.
6. Average fold probabilities.
7. Apply binary frame threshold `0.5536812544`.
8. Run all five grading models only when binary predicts NAFLD.
9. Apply grading frame threshold `0.6078062057`.
10. Return exact labels, probabilities, thresholds, fold scores, fold standard
    deviation, model version, contract hash, and `research_use_only`.
11. Return explicit 4xx errors for invalid images and 5xx/503 errors for model
    failures. Never return fabricated values.

Phase 2 completion criteria:

- Known notebook and service predictions match within an agreed numerical
  tolerance, normally `1e-4` on CPU.
- Every probability vector is finite, shape `(1, 2)`, and sums to one.
- A binary negative response contains no grading prediction.
- Boundary tests prove the documented threshold behaviour.
- `/api/predict` no longer returns HTTP 501.

### Phase 3 — strict Express integration

Status: **completed and runtime-verified**

Primary file: `server.ts`

Required changes:

- Require `ML_SERVICE_URL`; fail clearly when missing.
- Add an inference timeout and abort signal.
- Forward the uploaded bytes and content type to Flask.
- Validate the complete V3 response schema.
- Remove `generateSimulatedPrediction()` and every fallback to randomness.
- Preserve meaningful Flask validation errors.
- Return 503 when the model service is unavailable.
- Do not upload/store an image or create a scan record when inference fails.
- Optionally protect the internal Flask endpoint with a service token.
- Reduce the current 50 MB upload limit to a justified image limit.

Phase 3 completion criteria:

- Stopping Flask makes Express fail closed.
- Invalid Flask output is rejected and not saved.
- No code path can create a simulated clinical result.
- Successful responses are stored exactly once.

### Phase 4 — versioned scan and database schema

Status: **completed and runtime-verified**

Primary files:

- `src/types.ts`
- `src/lib/db.ts`
- `data/db.json` compatibility handling
- Firestore documents/rules as required

New scan fields should include:

- `schemaVersion`
- `modelVersion`
- `inputMode`
- `binaryResult`
- `gradingResult`
- `finalLabel`
- `researchUseOnly`
- probabilities in a documented 0–1 unit
- thresholds
- fold scores or fold standard deviation
- contract/model hashes
- actual image width and height
- inference timestamp and latency

Legacy `Normal`/`Abnormal` records must remain readable. Do not rewrite old
mock records as if they came from V3.

### Phase 5 — V3 result UI

Status: **completed and runtime-verified**

Primary files:

- `src/pages/DoctorDashboard.tsx`
- `src/components/ImageUploader.tsx`
- `src/components/ResultCard.tsx`
- `src/components/ProbabilityChart.tsx`
- `src/pages/ScanDetail.tsx`
- `src/components/ScanTable.tsx`
- `src/pages/AdminDashboard.tsx`

Required changes:

- Display Non-NAFLD or NAFLD using the exact contract labels.
- Show binary probabilities and decision threshold.
- Show grading only when performed.
- Show Grade 1 mild or Grade 2 moderate/severe probabilities and threshold.
- Replace `confidence` wording with ensemble probability/score.
- Show fold disagreement without presenting it as calibrated uncertainty.
- Show model version, input mode, and research-only notice.
- Remove `medical-grade`, `clinical-grade`, unsupported DICOM, and other claims
  that exceed current evidence.
- Display actual image metadata instead of hardcoded resolution.
- Provide clear invalid-image and model-unavailable states.
- Preserve a legacy display path for existing mock records.

### Phase 6 — automated verification

Status: **completed and runtime-verified**

Required coverage:

- Contract validation failures.
- Missing/corrupt model hashes.
- Model input/output shapes.
- Grayscale and colour preprocessing.
- Pixel-range and no-double-rescaling checks.
- Ensemble arithmetic.
- Cascade gating and threshold boundaries.
- Invalid upload and malformed-image errors.
- Express timeout/fail-closed behaviour.
- No database write on inference failure.
- React rendering for binary negative, mild, moderate/severe, legacy, loading,
  and error states.
- End-to-end upload through Express to Flask and back to history.

Record startup time, per-request CPU latency, memory use, and model-service
health after repeated requests.

### Phase 7 — packaging, deployment, and security hardening

Status: pending

Required changes:

- Document or containerize the Node and Python processes.
- Use one TensorFlow worker initially; multiple workers duplicate all ten
  models in memory.
- Configure health/readiness checks and startup ordering.
- Set production secrets; never use the default JWT secret.
- Replace unsalted SHA-256 password handling with a password-hashing scheme
  designed for authentication.
- Define image-retention, deletion, and access rules.
- Avoid storing large base64 images in `data/db.json` for production.
- Configure Firebase/Firestore permissions when Firebase is used.
- Add structured logs without patient-identifying image contents.
- Add request IDs, latency, model version, and failure reason to audit logs.
- Ensure the service is presented as research-only.

Phase 7 completion criteria:

- Clean startup from documented commands or deployment configuration.
- No development secrets or mock users in production.
- Health probes cover both Express and TensorFlow readiness.
- A deployment rollback retains the matching model contract and hashes.

## 12. Completed phase instructions and commands

### Phase 1 commands used on this Windows machine

The installed interpreter is:

```text
G:\Python 3.12\python.exe
```

PowerShell script execution is restricted by Device Guard, and the standalone
`pip.exe` launcher is blocked. Activation is unnecessary. Invoke Python and pip
through the interpreter directly.

From `G:\NAFLD-DetectorL`:

```powershell
& 'G:\Python 3.12\python.exe' -m venv .venv

& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip

& '.\.venv\Scripts\python.exe' -m pip install -r '.\ml_service\requirements.txt'
```

Start the model service:

```powershell
$env:MODEL_BUNDLE_DIR = 'G:\NAFLD-DetectorL\nafld_v3'

& '.\.venv\Scripts\python.exe' -m ml_service.app
```

Keep that terminal open. From a second PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:5001/health
```

The required result is:

```text
ready: True
status: ready
schema_version: 3
tensorflow_version: 2.20.0
loaded_models.binary: 5
loaded_models.grading: 5
loaded_models.total: 10
```

Stop the development service with `Ctrl+C` in its terminal.

### Phase 2 commands used on this Windows machine

With the Phase 1 virtual environment already in place, no additional
packages were required. Phase 2 added preprocessing and cascade inference
to `ml_service/inference.py` and replaced the HTTP 501 stub in
`ml_service/app.py` with a working `POST /api/predict` route.

Verification script (uses Flask test client; no live server needed):

```powershell
$env:MODEL_BUNDLE_DIR = 'G:\NAFLD-DetectorL\nafld_v3'
$env:PYTHONUTF8 = '1'

& '.\.venv\Scripts\python.exe' -m ml_service.test_phase2
```

Completion evidence recorded on 2026-07-23:

```text
Results: 30/30 passed -- ALL PASSED

Section 0  Flask app created without error
Section 1  GET /health 200, ready=True, 10 models loaded
Section 2  Synthetic images built
Section 3  /api/predict returns 200 (not 501)
           success=True, binary_result present, final_label present
           research_use_only=True, input_mode=single_frame
           binary probs sum ~1.0, 5 fold scores, threshold matches contract
           Non-NAFLD result: grading keys absent (cascade gating confirmed)
Section 4  Pure-white image: 200, valid final_label
Section 5  Pure-black image: 200
Section 6  PNG upload: 200
Section 7  Missing 'image' field: 400
Section 8  Corrupt JPEG bytes: 400
Section 9  MIME/magic mismatch (PNG bytes as image/jpeg): 400
Section 10 Unsupported MIME (image/tiff): 400
```

Sample binary prediction observed during verification:

```text
Binary prediction: Non-NAFLD (prob_nafld=0.5283, threshold=0.5537)
Cascade stopped: Non-NAFLD.
```

Sample cascade prediction observed during verification:

```text
Binary prediction: NAFLD (prob_nafld=0.5878, threshold=0.5537)
Grading prediction: Grade1_Mild (prob_moderate_severe=0.5486, threshold=0.6078)
```

### Phase 3 commands used on this Windows machine

Phase 3 hardened `server.ts` by enforcing `ML_SERVICE_URL`, adding a 15-second
`AbortController` timeout, validating the complete V3 response schema, deleting
`generateSimulatedPrediction()` and all fallback-to-randomness paths, returning 503
when Flask is unavailable, forwarding 4xx error messages, and capping uploads at 20 MB.

Verification script (`server_phase3_test.mjs`):

```powershell
$env:PATH = "G:\Nodejs;$env:PATH"
& 'G:\Nodejs\node.exe' server_phase3_test.mjs
```

Completion evidence recorded on 2026-07-23:

```text
Results: 20/20 passed -- ALL PASSED

T1 - No ML_SERVICE_URL -> 503 (fail-closed, no simulation)
     [PASS] Returns 503 when ML_SERVICE_URL is absent
     [PASS] Error message mentions ML_SERVICE_URL or configuration
     [PASS] No scan written for 503 response

T2 - Flask unreachable -> 503
     [PASS] Returns 503 when Flask is unreachable
     [PASS] No DB write on unreachable Flask

T3 - Flask returns 400 -> Express returns 400 with forwarded error
     [PASS] Returns 400 when Flask returns 400
     [PASS] Flask error message forwarded to client
     [PASS] No DB write on Flask 400

T4 - Flask returns 500 -> Express returns 502
     [PASS] Returns 502 when Flask returns 500
     [PASS] No DB write on Flask 500

T5 - Flask returns bad schema -> Express returns 502, no DB write
     [PASS] Returns 502 on bad schema
     [PASS] No DB write on schema validation failure

T6 - Valid V3 Flask response -> 200, scan saved once
     [PASS] Returns 200 on valid V3 response
     [PASS] Scan has id
     [PASS] prediction is Normal (Non-NAFLD mapped)
     [PASS] confidence is a number in (0,100]
     [PASS] probabilities.Normal and probabilities.Abnormal present
     [PASS] probabilities sum ~100
     [PASS] Scan written and readable back (GET /api/scans/:id returns 200)

T7 - No image field -> 400
     [PASS] Returns 400 when no image field
```

### Phase 4 commands used on this Windows machine

Phase 4 expanded the `Scan` schema in `src/types.ts` to include a new
`V3InferenceMetadata` interface (`schemaVersion: 3`, `modelVersion`,
`contractSha256`, `inputMode`, `researchUseOnly`, binary cascade fields,
optional grading cascade fields, `finalLabel`, `imageWidth`, `imageHeight`,
`inferenceLatencyMs`). The `Scan` interface now extends
`Partial<V3InferenceMetadata>` while retaining legacy `prediction`,
`confidence`, and `probabilities` fields for backwards compatibility.

`server.ts` was updated to:
- Parse image dimensions from raw buffer bytes (PNG `0x89PNG`, JPEG `0xFFD8`,
  BMP `BM` magic bytes).
- Measure inference latency around the Flask `fetch()` call.
- Populate all V3 fields in the scan record before writing to Firestore/JSON.
- Keep legacy fields derived from `binary_result` and `binary_prob_*` so the
  existing UI continues to function without modification.

Verification script (`server_phase4_test.mjs`):

```powershell
$env:PATH = "G:\Nodejs;$env:PATH"
& 'G:\Nodejs\node.exe' server_phase4_test.mjs
```

Completion evidence recorded on 2026-07-24:

```text
Results: 40/40 passed -- ALL PASSED

T1 - Non-NAFLD scan stores schemaVersion: 3 & full V3 fields
     [PASS] Returns 200 OK
     [PASS] schemaVersion is 3
     [PASS] modelVersion is 3.0.0
     [PASS] contractSha256 present
     [PASS] inputMode is single_frame
     [PASS] researchUseOnly is true
     [PASS] binaryResult is Non-NAFLD
     [PASS] binaryProbNafld is 0.35
     [PASS] binaryProbNonNafld is 0.65
     [PASS] binaryThreshold present
     [PASS] binaryFoldProbs has 5 entries
     [PASS] binaryFoldStd present
     [PASS] gradingPerformed is false
     [PASS] gradingResult absent
     [PASS] finalLabel is Non-NAFLD
     [PASS] imageWidth parsed (512)
     [PASS] imageHeight parsed (256)
     [PASS] inferenceLatencyMs is a positive number
     [PASS] Legacy prediction is Normal
     [PASS] Legacy confidence is 65
     [PASS] Legacy probabilities present

T2 - NAFLD + Mild grading scan stores full grading V3 fields
     [PASS] Returns 200 OK
     [PASS] schemaVersion is 3
     [PASS] binaryResult is NAFLD
     [PASS] gradingPerformed is true
     [PASS] gradingResult is Grade1_Mild
     [PASS] gradingProbMild is 0.58
     [PASS] gradingProbModerateSevere is 0.42
     [PASS] gradingThreshold present
     [PASS] gradingFoldProbs has 5 entries
     [PASS] finalLabel is NAFLD-Grade1_Mild
     [PASS] Legacy prediction is Abnormal

T3 - GET /api/scans/:id retrieves full versioned Scan object
     [PASS] GET /api/scans/:id returns 200
     [PASS] Retrieved scan matches ID
     [PASS] Retrieved scan has schemaVersion=3
     [PASS] Retrieved scan has finalLabel NAFLD-Grade1_Mild
     [PASS] Retrieved scan preserves contractSha256

T4 - Legacy scans remain readable via GET /api/scans
     [PASS] GET /api/scans returns 200
     [PASS] Returns an array of scans
     [PASS] Contains scans
```

### Phase 5 changes verified on this machine

Completion criteria verified via `tsc --noEmit` (zero errors) on 2026-07-24.

| File | Change |
|---|---|
| `src/components/ResultCard.tsx` | V3 path: exact contract `finalLabel`, ensemble probability rows (0–1) + threshold, grading section shown only when `gradingPerformed=true`, fold std-dev note, model-metadata block, research-only notice. Legacy path unchanged. |
| `src/components/ProbabilityChart.tsx` | Dual-mode: V3 uses 0–1 axis with `ReferenceLine` at threshold; legacy keeps 0–100% axis. |
| `src/components/ImageUploader.tsx` | Removed DICOM and "Medical-Grade" references; replaced with "Research Use Only" badge; size limit corrected to 20 MB. |
| `src/pages/ScanDetail.tsx` | Hardcoded "1024 × 768 DICOM Frame" replaced with `scan.imageWidth × scan.imageHeight` (or "Unknown"). Removed "DICOM Diagnostic Viewer" and "SECURE DISK" labels. Added V3 inference-metadata footer strip. |
| `src/components/ScanTable.tsx` | Shows `finalLabel` for V3 scans and legacy `prediction` for old records. Confidence column suffix is "prob" for V3 and "conf" for legacy. Footer updated to "NAFLD-Detector V3". |
| `src/pages/DoctorDashboard.tsx` | Removed DICOM and marketing copy. 503 responses now render a dedicated "Inference Service Unavailable" amber banner instead of generic error. |

Verification command:
```powershell
$env:PATH = "G:\Nodejs;$env:PATH"
cmd /c "node_modules\.bin\tsc --noEmit"
# Output: (empty — zero errors)
```

### Phase 6 changes verified on this machine

Completion criteria verified via `server_phase6_test.mjs` (111/111 passed) on 2026-07-24.

Verification command:
```powershell
$env:PATH = "G:\Nodejs;$env:PATH"
& 'G:\Nodejs\node.exe' server_phase6_test.mjs
```

Performance snapshot recorded during verification:
- Express startup: **2 274 ms**
- Steady-state per-request latency (5 requests, after warmup): min=548 ms, max=3 483 ms, mean=1 296 ms
- Memory RSS (test process): 42 MB → 47 MB across full suite
- Model-service health: 5/5 repeated requests succeeded

| Section | Coverage | Checks |
|---|---|---|
| S1 | Contract validation failures — inline validator unit tests (null/bad types/out-of-range/wrong enums) | 26 |
| S2 | Missing/corrupt model hashes — metadata & sha256 storage | 7 |
| S3 | Model input/output shapes — prob range enforcement via HTTP (7 bad-schema → 502) | 7 |
| S4 | Grayscale/colour preprocessing — raw bytes forwarded, imageUrl & dimensions preserved | 5 |
| S5 | Ensemble arithmetic — fold probs and std stored verbatim, fold mean ≈ binaryProbNafld | 8 |
| S6 | Cascade gating & threshold boundaries (grading absent when not performed, invalid grading_result → 502, boundary probs 0 and 1 accepted) | 10 |
| S7 | Invalid upload errors — no image, empty name, invalid JWT, no auth, missing ML_SERVICE_URL | 5 |
| S8 | Express timeout / fail-closed — 16 s slow Flask → 503 within 15 s | 2 |
| S9 | No DB write on failure — scan count unchanged after schema failure, Flask 500, and Flask 4xx | 7 |
| S10 | React rendering states — loading, V3 Non-NAFLD, V3 NAFLD-Mild, V3 NAFLD-Mod/Severe, legacy Normal, legacy Abnormal, error | 20 |
| S11 | End-to-end: POST predict → GET by ID → list history → 404 for missing | 9 |
| S12 | Performance: startup time, 5-request latency, health rate, memory | 5 |

**Total: 111/111 — ALL PASSED**

## 13. Known limitations and risks

- Only 55 independent patients are available.
- Ten frames per patient are correlated observations, not independent samples.
- Data comes from a small single-source cohort.
- There is no locked-threshold external validation.
- V3 thresholds were selected using complete OOF development predictions.
- Binary specificity is 0.5882 at the sensitivity-oriented patient threshold.
- Complete cascade patient balanced accuracy is 0.6220.
- The grading model cannot separate moderate from severe disease.
- The single-frame UI does not exploit patient-level frame aggregation.
- Softmax probabilities are not proven calibrated clinical confidence.
- Scanner/site/domain effects have not been externally tested.
- Phases 2 and 3 integration completed: Express now strictly forwards to Flask, validates V3 response schema, fails closed, and contains zero simulated prediction code paths.
- Current local authentication and storage are prototypes, not production
  clinical-security controls.

## 14. Guardrails for future agents

1. Read this file and `nafld_v3/inference_contract_v3.json` before editing.
2. Use only the ten canonical `*_model_v3.keras` files in `nafld_v3/models/`.
3. Never use the archived V1/V2 models for active predictions.
4. Never divide pixels by 255 outside the saved EfficientNet models.
5. Never use patient thresholds for a single uploaded frame.
6. Never run grading below the binary threshold.
7. Never restore random, filename-driven, or simulated predictions as a
   fallback.
8. Never save a scan record after failed inference.
9. Preserve exact class order, thresholds, contract hash, and model version.
10. Preserve legacy database records without relabelling their origin.
11. Do not claim medical-grade or clinical validation.
12. Do not merge external images without stable patient IDs, label provenance,
    source tracking, and license review.
13. Do not change the training architecture and evaluation protocol at the same
    time; make controlled comparisons against V3 OOF results.
14. Do not delete V3 evidence tables, reports, contract, model card, notebook,
    or training log.

## 15. Exact next task

Phases 1–6 are **complete and runtime-verified**. The next agent should implement
**Phase 7 — packaging, deployment, and security hardening** only:

> - Document or containerize the Node and Python processes.
> - Use one TensorFlow worker initially; multiple workers duplicate all ten models in memory.
> - Configure health/readiness checks and startup ordering.
> - Set production secrets; never use the default JWT secret.
> - Replace unsalted SHA-256 password handling with a proper password-hashing scheme (e.g. bcrypt).
> - Define image-retention, deletion, and access rules.
> - Avoid storing large base64 images in `data/db.json` for production.
> - Configure Firebase/Firestore permissions when Firebase is used.
> - Add structured logs without patient-identifying image contents.
> - Add request IDs, latency, model version, and failure reason to audit logs.
> - Ensure the service is presented as research-only throughout.

Phase 7 completion criteria (from master report):
- Clean startup from documented commands or deployment configuration.
- No secrets in source control.
- Structured logs produced without image bytes.
- `data/db.json` not used in production path.
