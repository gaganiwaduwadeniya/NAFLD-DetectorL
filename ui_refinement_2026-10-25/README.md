# NAFLD Research UI Refinement Evidence

Date: 2026-10-25

Scope: local React/Express interface, PyTorch research-v1 inference integration,
responsive behavior, truthful model-output presentation, and local demo state.

## Outcome

The interface was refined from a generic dashboard into a focused research
workflow. The model and datasets were not trained, edited, or replaced.

- Navigation, authentication, scan intake, result interpretation, records,
  detail, and administrator views now share one restrained visual system.
- The model status is read from the backend instead of being hard-coded.
- Mobile navigation and mobile scan-history cards were added.
- Form labels, keyboard focus, inline validation, empty/loading/error states,
  and reduced-motion behavior were improved.
- The password-reset page no longer claims that an email was sent; no recovery
  provider is configured in this prototype.
- External web fonts were removed so the local interface does not depend on a
  third-party font request.

## Model-output contract shown by the UI

The selected research-v1 implementation is a two-stage cascade:

1. Five DenseNet121 fold models are ensembled for the Normal versus NAFLD gate.
2. Only when the gate selects NAFLD, five EfficientNet-B0 fold models are
   ensembled for conditional Mild, Moderate, or Severe grading.

The UI therefore shows the gate probability first and labels severity scores
as conditional within the NAFLD branch. Any derived joint scores are kept in a
secondary provenance section and are explicitly described as uncalibrated.
This avoids presenting the cascade as a single clinically calibrated
four-class model. The interface remains research-only.

## Fresh local database state

The manual-test database was backed up, then only its scan records were
cleared. The two local login accounts were retained. New disposable local
databases now initialize with zero scan records rather than example results.
No Firebase data, research dataset, checkpoint, or source image was changed.

Current clean-state expectation:

| Record type | Count |
|---|---:|
| Local users | 2 |
| Local scans | 0 |

## Actual screenshots

Before:

- [Login](screenshots/before/login.png)
- [Doctor dashboard](screenshots/before/doctor-dashboard.png)
- [Doctor dashboard mobile](screenshots/before/doctor-dashboard-mobile.png)

After:

- [Login](screenshots/after/login.png)
- [Doctor dashboard](screenshots/after/doctor-dashboard.png)
- [Doctor dashboard mobile](screenshots/after/doctor-dashboard-mobile.png)
- [Empty records after reset](screenshots/after/records-empty-fresh-start.png)
- [Empty administrator overview after reset](screenshots/after/admin-empty-fresh-start.png)
- [Genuine Stage 2 result workspace](screenshots/after/genuine-stage2-workspace-clean.png)
- [History with one genuine isolated result](screenshots/after/records-single-genuine-result.png)

The genuine result screenshot came from an approved, de-identified example in
an isolated temporary evidence database. It was not inserted into the clean
manual-test database.

## Verification commands

```bash
npm run test:local-db
npm run test:integration-contract
npm run lint
npm run build
```

The complete local start and troubleshooting guide is in
[`RUN_NEW_UI.md`](../RUN_NEW_UI.md).
