# 08 — Open issues and deployment boundary

1. Recapture dashboard, upload, result, provenance, history, error and mobile screenshots in a normal local browser environment. The automated capture blocker is documented; blank probes were rejected.
2. Complete a production frontend build on an environment where Vite can finish without mounted-drive stalls, then run the same authenticated E2E against the built server.
3. Define an approved production image-retention/access policy before enabling `STORE_UPLOAD_IMAGES`; it remains false.
4. Replace the development authentication/password storage and default JWT secret before any deployment. Conduct dependency/security review; `npm install` reported 17 audit findings (4 moderate, 12 high, 1 critical) that were not auto-fixed because dependency changes were outside this integration task.
5. Provision the exact model runtime reproducibly on the intended host and size CPU/GPU/memory/worker limits. Do not spawn multiple ten-model workers without memory assessment.
6. Obtain independent external clinical validation and qualified governance review before any clinical claim or use.

No public deployment, database migration, external upload, Kaggle job, model training, application release, git commit/push or thesis edit occurred. The old TensorFlow models/service, V3 records, datasets and all V4 experiments remain present and recoverable.
