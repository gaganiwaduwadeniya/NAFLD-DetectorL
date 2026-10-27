# NAFLD research-v1 application integration evidence

This package documents the local integration of the selected five-member DenseNet121 binary ensemble and five-member EfficientNet-B0 conditional staging ensemble into the existing React/Express application. It is research software, not a clinically validated diagnostic system.

Start with `reports/04_END_TO_END_TEST_RESULTS.md`, then `reports/06_LIMITATIONS_AND_RESEARCH_INTERPRETATION.md`. Runtime commands are in `runbooks/LOCAL_START.md`. The exact restricted inference source and selected model bundle are included under `pytorch_service/research_v1`; the running application has no dependency on another workspace.

No training, Kaggle job, model selection, dataset mutation, deployment, or thesis edit occurred in this task.
