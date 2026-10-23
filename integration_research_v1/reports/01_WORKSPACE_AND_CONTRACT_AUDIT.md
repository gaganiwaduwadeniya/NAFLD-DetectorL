# 01 — Workspace and contract audit

Date: 2026-10-24. The existing application was inspected in `G:\NAFLD-DetectorL`; the selected research release was read from `G:\NAFLD New`. The working tree already contained extensive V3/V4 modifications and untracked V4 evidence before this integration. Those files were not reset, stashed, overwritten, or relabelled.

The application is React 19/TypeScript/Vite with an Express API and a preserved TensorFlow V3 Flask service. V3 remains an explicit provider option. The new implementation adds a separate FastAPI/PyTorch process and an explicit `MODEL_PROVIDER=pytorch_research_v1` route; there is no automatic V3 fallback.

Verified selected release: 109 patients, 1,646 frames, class support 10/35/46/18. Binary model: five DenseNet121 members. Conditional grading model: five EfficientNet-B0 members. Binary manifest SHA-256 is `daa75075d2de25823a2648dcb061b273e46459513657014cb239b5b8d2975cd9`; staging manifest SHA-256 is `12a0894534756dd380d08a4bfe9a6fe8086abde1420fd068140e709d3d83a52a`; dataset version is `48b42abbee97b892591054019cdb4f84a65af9e0cd88d09fcbf8b3ae2a8b9856`.

The exact recorded runtime was provisioned locally in the ignored `.runtime/pytorch-research-v1` directory: Torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, NumPy 2.1.3, Pillow 13.0.0. Model files stay in the restricted research workspace and are not copied into the frontend or evidence ZIP.

Privacy risks identified: source ultrasound images may contain burned-in identifiers; existing historical local database records contain demonstration names and remote image URLs. Acceptance and evidence used only three previously approved anonymized consolidation examples plus synthetic names. Research-v1 image retention defaults off. The requested Chapter 5 thesis ZIP was not located during the bounded audit; no thesis was edited.

Repository instruction files: no applicable project-level `AGENTS.md` was found. The user-provided integration handoff was read completely and governs this work.
