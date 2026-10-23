# 04 — End-to-end test results

## Passed checks

- TypeScript `tsc --noEmit`: passed.
- Express response/image contract tests: 5/5 passed.
- Python gating/image/service contract tests: 4/4 passed.
- Exact-runtime model readiness: 10/10 selected checkpoints loaded; 5 DenseNet121 plus 5 EfficientNet-B0; hashes and dataset version verified.
- Direct CPU acceptance on three approved anonymized frames: genuine stages 2, 0 and 1; repeated predictions were exactly stable within the same runtime. Stage 0 omitted conditional/joint grading values as required.
- Authenticated HTTP E2E: register/authenticate → Express upload → PyTorch prediction → Express schema validation → schema-v4 save → history/detail read: passed.
- Real Grad-CAM through Express: HTTP 200 PNG, 333,231 bytes, EfficientNet-B0 member 0, Stage 2 target: passed.
- Unauthorized history request: HTTP 401: passed.
- Service outage after one saved record: HTTP 503 and history remained 1 record before/after: passed; no fallback and zero database writes.
- Input image retention: disabled; saved record `imageStored=false` and empty image URL.

## Measured local CPU observations

Direct acceptance startup recorded 99.24 s and approximately 948.5 MB RSS; a later authenticated E2E startup recorded 37.75 s and approximately 954.4 MB RSS, reflecting cache/mount effects. Per-image direct inference for the three fixtures was 13.48 s, 14.97 s and 6.79 s. The authenticated Stage 2 request recorded 5.60 s service inference and 5.91 s Express round trip. These are development-machine observations, not performance guarantees.

The real E2E output was Stage 2 with binary NAFLD score 0.7890; conditional scores were Mild 0.1215, Moderate 0.5155 and Severe 0.3629. The saved gated result matches conditional argmax. The real Stage 0 fixture had NAFLD score 0.0742 and correctly skipped grading.

## Incomplete checks

`npm run build` was attempted but Vite transformation did not complete in the bounded mounted-drive run; it was interrupted and is not reported as passed. Headless browser screenshot automation was attempted through two independent local mechanisms but could not complete a render of the Vite client in this execution environment. No blank probe is included as evidence. Actual UI responsive/keyboard screenshots therefore remain a blocker, not a passing result.

Machine evidence: `test-results/real_model_acceptance.json` and `test-results/authenticated_e2e.json` (local detailed evidence; the packaged copy is privacy reviewed).
