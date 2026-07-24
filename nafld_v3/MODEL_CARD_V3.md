# NAFLD Ensemble V3 Model Card

Generated: 2026-07-22T19:38:43.575604+00:00

Research use only. Not clinically validated.

## Data and evaluation

- Independent patients: 55
- Frames: 550
- Patient-stratified folds: 5
- Binary patient OOF AUC: 0.8824
- Grading patient OOF AUC: 0.7583
- Cascade patient balanced accuracy: 0.6220

## Limitations

Thresholds were selected from OOF development predictions. The small single-site cohort and
correlated frames limit precision. A locked-threshold, patient-grouped external evaluation is
required before any clinical claim.
