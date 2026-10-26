# Miscellaneous commit scope — 2026-10-26

This note records the safety boundary used for the miscellaneous repository
commit after the application and one-command launcher were completed.

## Included

- Reusable V4 Python source, deterministic tests, and dependency lock files.
- Privacy-reviewed application screenshots and UI refinement evidence.
- Local browser evidence scripts and de-identified integration-test results.
- Three de-identified Grad-CAM evidence images with no visible patient identity.

## Deliberately left local

- Ten V3 Keras worktree model files (approximately 19–30 MB each). These are
  Git LFS-managed artifacts, and Git LFS is not installed in this environment.
- The executed V3 notebook and package-lock peer-metadata churn.
- V4 patient-level reports, approval state, configs, notebooks, and the
  `full_training_patient_results_2026-10-20.DO_NOT_UPLOAD.csv` table.
- The generated thesis-evidence directory and duplicate 22 MB portable ZIP.
- Existing tracked files whose only detected difference is line-ending churn.
- Original images, annotations, databases, checkpoints, runtime environments,
  secrets, and patient-ID mappings.

These exclusions remain present locally. They were not deleted, overwritten,
or staged. Generated archives should be distributed as controlled research
artifacts rather than committed as miscellaneous source code.

## Verification

Before staging the reusable V4 code, the local V4 test suite completed with
124 passing tests and 1 skipped test. The committed UI and launcher had already
passed their production build, type check, model contract, local database, and
end-to-end launcher checks in commit `100fbee`.
