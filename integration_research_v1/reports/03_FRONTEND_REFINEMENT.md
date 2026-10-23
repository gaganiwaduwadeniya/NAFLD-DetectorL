# 03 — Frontend refinement

The React UI now has a separate schema-v4 result path while preserving historical V3 rendering. It displays the exact four labels, DenseNet binary model scores and locked 0.5 gate. EfficientNet conditional stage scores appear only when grading ran. Derived joint scores are hidden in an expandable provenance area and explicitly labelled supplementary/un-calibrated.

The result card shows architecture/member counts, contract/model/dataset provenance, manifest hashes, latency, input mode, and whether the image was retained. Research-only and single-frame limitations are prominent. “Clinical grade”, “HIPAA compliant”, autonomous diagnosis and calibrated-confidence language was removed from the global UI. History and detail pages support schema v4 and show an explicit “image not retained” state.

The dashboard can request a real Grad-CAM. Stage 0 uses the DenseNet detector NAFLD target; positive results use EfficientNet-B0 staging member 0 and the predicted conditional target. The UI identifies it as qualitative and not a segmentation or proof of medical reasoning. JPEG/PNG acceptance is consistent across client, Express and Python.

Browser automation was attempted with Windows Chrome and local Playwright. The headless browser could not complete loading the Vite development client in the isolated execution environment, despite the API path operating. Blank probe captures were removed and are not evidence. Therefore responsive/keyboard visual QA and full UI screenshots remain unverified, while TypeScript compilation and the authenticated API/history contract are verified.
