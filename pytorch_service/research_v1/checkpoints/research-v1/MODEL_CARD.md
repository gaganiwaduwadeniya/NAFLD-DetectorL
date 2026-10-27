# NAFLD research v1

Selected research architectures: DenseNet121 for Non-NAFLD/NAFLD detection,
and EfficientNet-B0 for mild/moderate/severe ultrasound steatosis classification.
Each task export contains its five independently trained patient-fold models.
The inference implementation averages class probabilities across those members.
It applies staging only when the binary NAFLD probability is at least 0.5.

This is a research prototype, not a clinically validated diagnostic system.

## Data and labels

109 retained patients, 1,646 images; grades 0/1/2/3 contain 10/35/46/18 patients.
Grade 0 is normal/Non-NAFLD; grades 1, 2 and 3 are mild, moderate and severe
steatosis, respectively, according to the user's qualified-doctor attestation.
Detailed grading and inter-reader criteria were not independently verified.
All original source images and annotations were preserved. Source-bound P006
exclusion, P015 retention and exact duplicate removal are recorded in the audit.
The images contain embedded identifiers and remain within authorized private use.

Dataset version: `48b42abbee97b892591054019cdb4f84a65af9e0cd88d09fcbf8b3ae2a8b9856`.

## Selection

The explicit decision is in `configs/selection.yaml` and
`outputs/selection/research-v1/decision.json`. Both tasks were selected separately
after four complete frozen-backbone comparisons and two shortlisted fine-tuning
comparisons per task. ResNet18 continuation was set aside at the user's request;
its incomplete result is not treated as a full benchmark. The optional direct
four-class baseline was not executed.

The fine-tuned binary candidates tie on mean fold AUROC (0.9445). DenseNet121
has better pooled AUROC, sensitivity, specificity and balanced accuracy than
fine-tuned EfficientNet-B0. Frozen EfficientNet-B0 has slightly higher mean fold
AUROC (0.9495), but lower pooled AUROC and sensitivity with equal specificity.
This release chooses DenseNet121 descriptively; no superiority test is claimed.

EfficientNet-B0 has the highest mean fold staging macro F1 (0.5540) and pooled
macro F1 (0.5510). EfficientNetV2-S has fewer two-grade errors and higher moderate
recall. B0 is retained for its macro F1 and severe recall, with the tradeoff below.
Selection preceded calculation of the exact pair's combined OOF results.

## Development evaluation

| Component | Metric | Point estimate | Conditional 95% interval |
| --- | --- | ---: | --- |
| Binary, 109 patients | AUROC | 0.9273 | 0.8687-0.9707 |
| Binary | Sensitivity | 0.9495 | 0.9091-0.9798 |
| Binary | Specificity | 0.7000 | 0.5000-0.9000 |
| Staging, 99 NAFLD patients | Macro F1 | 0.5510 | 0.4601-0.6331 |
| Selected cascade, 109 patients | Macro F1 | 0.5682 | 0.4716-0.6503 |

Binary confusion matrix: true Non-NAFLD [7, 3]; true NAFLD [5, 94].
Staging recalls for grades 1/2/3: 0.6857/0.3913/0.7222. Six of 99 NAFLD
patients have a two-grade staging error. For the cascade, nine of 109 patients
have an error of two or more grades; ordinal mean absolute error is 0.5413.
Full precision/recall, PR-AUC, balanced accuracy, fold metrics, CSV predictions,
confusion matrices and plots are in `outputs/evaluation/`.

These estimates use one held-out member per patient and average that patient's
image probabilities before the cascade decision. They do not estimate the
five-member inference ensemble's performance: four members were trained on
each development patient. Four image inference examples are integration checks
only, not a new test set or performance evaluation.

Five-fold patient splits are shared by both tasks. Staging training excludes
grade 0. Validation folds were used for epoch and architecture selection.
The 1,000 bootstrap replicates resample patients within fold and actual grade,
condition on the fitted models, and exclude retraining, selection uncertainty,
new split variability and population shift. Only ten controls are available,
two per validation fold. No independent test set or external validation exists.
Threshold 0.5 was fixed; no threshold was optimized on scored patients.
Probabilities are uncalibrated. The severity results, especially moderate-grade
recall, are insufficient to claim dependable clinical grading.

## Processing and training

8-bit RGB/grayscale images are EXIF-oriented, converted to RGB, resized with
bilinear interpolation and letterboxed to 224 pixels while preserving the full
field. Black padding and fixed ImageNet mean/std are used. No data-fitted
normalization, cropping, contrast change or unapproved augmentation is applied.
Full preprocessing and label mappings are saved inside every checkpoint.

Both selected candidates used native timm pretrained weights, conservative
fine-tuning with a common linear head, AdamW, differential learning rates
1e-5/1e-3, weight decay 0.01, cosine schedule, head dropout 0.1, class-weighted
loss, maximum 20 epochs, patience 5, gradient clipping and mixed precision.
The actual global batch size was 16 across two T4 GPUs. The registry records
seed, native pretraining source, data/code hashes, best epochs and Kaggle runs.
All 20 shortlisted fits completed; successful runs were not repeated.
Losses with and without weighting are recorded during fitting; no controlled
unweighted training ablation was run, so an effect attributable to weighting
cannot be claimed.

## Reproduction and runtime

Run from the project root:

```bash
.venv/bin/python scripts/finalize_models.py --selection-config configs/selection.yaml
.venv/bin/python scripts/prepare_inference_check.py
.venv/bin/python scripts/run_kaggle.py start --config configs/kaggle-inference.yaml --bundle kaggle/inference-check-v1 --state-root outputs/kaggle-inference
```

The submission registry skips an identical already-submitted job. Export
directories and the selection decision are immutable; existing matching
exports are verified and reused. No GPU training occurs in these commands.

Inference enforces the recorded Torch and timm versions. Trained runtime:
Torch 2.11.0+cu128 and timm 1.0.29. The exact private Kaggle container is pinned
in `kaggle/inference-check-v1/kernel-metadata.json`; internet and GPU are disabled.
Local development dependencies differ and are not a verified inference runtime.
The CPU integration result and full runtime versions are saved separately in
`outputs/inference/research-v1/verification.json` after successful retrieval.

In the matching runtime, a new supplied image can be processed with:

```bash
python scripts/predict.py /path/to/image.jpg --binary-export checkpoints/research-v1/binary --staging-export checkpoints/research-v1/staging --output outputs/inference/new-image.json --grad-cam outputs/inference/new-image-grad-cam.png
```

JSON output includes grade, binary probabilities, conditional severity and joint
probabilities when staging is applied, threshold, model version, dataset version
and image hash. Grad-CAM explains one named ensemble member and is qualitative;
it does not establish clinical localization or justify a diagnosis.
