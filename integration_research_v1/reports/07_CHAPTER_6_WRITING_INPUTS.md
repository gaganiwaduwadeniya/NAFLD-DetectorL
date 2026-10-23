# 07 — Chapter 6 writing inputs

The implemented system uses a React/TypeScript research dashboard, an authenticated Express orchestration API and an isolated FastAPI PyTorch inference service. The process boundary preserves the historical TensorFlow V3 service while allowing explicit selection of the research-v1 provider. At service startup, the implementation verifies the recorded runtime, model manifests, dataset version and all ten ensemble members before readiness.

For an accepted single ultrasound frame, the DenseNet121 ensemble produces the mean NAFLD score. A locked threshold of 0.5 controls whether conditional grading is performed. Below the threshold the output is Stage 0; at or above it the EfficientNet-B0 ensemble mean determines Mild, Moderate or Severe by conditional argmax. The web API validates this relationship and saves a versioned result only after successful inference.

System verification used approved de-identified images and synthetic identities. One authenticated E2E case produced Stage 2 with a 0.7890 NAFLD score and conditional Mild/Moderate/Severe scores of 0.1215/0.5155/0.3629. A service-outage test returned 503 and created no history record. A model-derived Grad-CAM was returned with member and target provenance. These observations establish implementation behavior, not external clinical validity.

Recommended Chapter 6 figures: process topology diagram; schema-v4 gated decision sequence; privacy/data-retention boundary; verified response table; service-outage sequence; the included qualitative Grad-CAM with an explicit non-causal disclaimer. UI screenshots should be recaptured manually or in a normal browser environment before thesis use because automated UI capture remained blocked here.

Primary source references: selected release model card/system/manifests in `G:\NAFLD New\checkpoints\research-v1`; authoritative inference/preprocessing modules in `G:\NAFLD New\src\nafld`; this integration's contract, service source, TypeScript validator and executed JSON test evidence.
