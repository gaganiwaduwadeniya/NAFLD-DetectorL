# V3 Binary External Image Evaluation

Generated: 2026-07-24T13:12:48.073078+00:00

> Research evaluation only. These folder labels are treated as ground truth,
> but this report does not establish their provenance, patient independence,
> or clinical reference standard.

## Setup

- Images evaluated: **87**
- Decode failures: **0**
- True Non-NAFLD: **11**
- True NAFLD: **76**
- Single-frame NAFLD threshold: **0.5536812544**
- Model version: **3.0.0**
- Contract SHA-256: `b3cae3967199f27570889d8392d4ce31d3ce4977471aab34277b48993c3abd6f`

## Metrics

| Metric | Result |
|---|---:|
| Accuracy | 75.86% |
| Balanced accuracy | 51.20% |
| NAFLD sensitivity/recall | 84.21% |
| Normal specificity/recall | 18.18% |
| NAFLD precision | 87.67% |
| Negative predictive value | 14.29% |
| NAFLD F1 | 85.91% |
| ROC-AUC | 0.601675 |

## Confusion matrix

| True / predicted | Non-NAFLD | NAFLD |
|---|---:|---:|
| Non-NAFLD | 2 | 9 |
| NAFLD | 12 | 64 |

Correct: **66**; incorrect: **21**.

## Incorrect predictions

| File | True | Predicted | P(NAFLD) | Threshold | Fold std |
|---|---|---|---:|---:|---:|
| `NFLD\id52.jpg` | NAFLD | Non-NAFLD | 0.225105 | 0.553681 | 0.103168 |
| `normal\id3.jpg` | Non-NAFLD | NAFLD | 0.766827 | 0.553681 | 0.091540 |
| `normal\id51.jpg` | Non-NAFLD | NAFLD | 0.755029 | 0.553681 | 0.079334 |
| `normal\id61.jpg` | Non-NAFLD | NAFLD | 0.715457 | 0.553681 | 0.106766 |
| `normal\id28.jpg` | Non-NAFLD | NAFLD | 0.711508 | 0.553681 | 0.128400 |
| `normal\id1.jpg` | Non-NAFLD | NAFLD | 0.696107 | 0.553681 | 0.104964 |
| `NFLD\id54.jpg` | NAFLD | Non-NAFLD | 0.430478 | 0.553681 | 0.057684 |
| `NFLD\id33.jpg` | NAFLD | Non-NAFLD | 0.432877 | 0.553681 | 0.095815 |
| `NFLD\id85.jpg` | NAFLD | Non-NAFLD | 0.446155 | 0.553681 | 0.092966 |
| `normal\id2.jpg` | Non-NAFLD | NAFLD | 0.660834 | 0.553681 | 0.048122 |
| `normal\id50.jpg` | Non-NAFLD | NAFLD | 0.655267 | 0.553681 | 0.071625 |
| `normal\id84.jpg` | Non-NAFLD | NAFLD | 0.629466 | 0.553681 | 0.103314 |
| `NFLD\id49.jpg` | NAFLD | Non-NAFLD | 0.480882 | 0.553681 | 0.117545 |
| `NFLD\id12.jpg` | NAFLD | Non-NAFLD | 0.485300 | 0.553681 | 0.063490 |
| `NFLD\id41.jpg` | NAFLD | Non-NAFLD | 0.486535 | 0.553681 | 0.088240 |
| `NFLD\id10.jpg` | NAFLD | Non-NAFLD | 0.500204 | 0.553681 | 0.094470 |
| `NFLD\id7.jpg` | NAFLD | Non-NAFLD | 0.500489 | 0.553681 | 0.086784 |
| `NFLD\id83.jpg` | NAFLD | Non-NAFLD | 0.509419 | 0.553681 | 0.131476 |
| `NFLD\id84.jpg` | NAFLD | Non-NAFLD | 0.542748 | 0.553681 | 0.120872 |
| `NFLD\id72.jpg` | NAFLD | Non-NAFLD | 0.543868 | 0.553681 | 0.132209 |
| `normal\id20.jpg` | Non-NAFLD | NAFLD | 0.561333 | 0.553681 | 0.111598 |

## Interpretation cautions

- The classes are strongly imbalanced, so balanced accuracy, sensitivity,
  and specificity are more informative than accuracy alone.
- Each file is evaluated as a single frame with the frame threshold.
- If multiple files belong to one patient, image-level metrics overstate the
  effective independent sample size; patient IDs are required for a valid
  patient-level analysis.
- External scanner, acquisition, crop, annotation, or compression differences
  can cause domain shift.
- Review `binary_predictions.csv` for every score and fold disagreement.
