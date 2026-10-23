# Chapter 6 source map

| Topic | Verified source |
|---|---|
| Architecture and selected weights | `G:\NAFLD New\checkpoints\research-v1\MODEL_CARD.md`, `system.json`, binary/staging manifests |
| Preprocessing and inference semantics | `G:\NAFLD New\src\nafld\preprocessing\images.py`, `inference\pipeline.py`, `scripts\predict.py` |
| Service topology and validation | `pytorch_service/`, `src/server/pytorchContract.ts`, `server.ts` |
| UI behavior | `src/components/ResearchV1Card.tsx`, `src/pages/DoctorDashboard.tsx`, history/detail components |
| Exact contract | `integration_research_v1/contracts/prediction_response.schema.json` |
| Numerical/runtime evidence | `integration_research_v1/test-results/real_model_acceptance.json` |
| Authenticated E2E and outage evidence | `integration_research_v1/test-results/authenticated_e2e.json` |
| Valid visual | `integration_research_v1/evidence/screenshots/06_real_gradcam_e2e.png` |
| Limitations | `integration_research_v1/reports/06_LIMITATIONS_AND_RESEARCH_INTERPRETATION.md` |

Use these as factual inputs; do not infer clinical validation or external generalization.
