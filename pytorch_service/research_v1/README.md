# Repository-local research-v1 inference bundle

This directory makes the selected PyTorch cascade self-contained inside the
application repository. The running service does not depend on another
workspace.

Included runtime material:

- five selected DenseNet121 binary checkpoints;
- five selected EfficientNet-B0 conditional staging checkpoints;
- the binary and staging manifests and model registries;
- the release model card and system metadata; and
- the exact minimal Python modules required to construct, preprocess, execute,
  and explain the selected models.

The copied payload is 224,091,217 bytes, of which 224,036,160 bytes are the ten
checkpoint files. The checkpoints are byte-for-byte copies of the selected
research-v1 release; they were not converted or retrained.

Integrity anchors:

- binary manifest SHA-256:
  `daa75075d2de25823a2648dcb061b273e46459513657014cb239b5b8d2975cd9`
- staging manifest SHA-256:
  `12a0894534756dd380d08a4bfe9a6fe8086abde1420fd068140e709d3d83a52a`
- dataset version:
  `48b42abbee97b892591054019cdb4f84a65af9e0cd88d09fcbf8b3ae2a8b9856`

Each manifest also contains the expected SHA-256 for every checkpoint. The
service verifies the manifest hashes and then the individual model hashes
before becoming ready. Missing Git LFS objects therefore fail closed.

This remains a research prototype without independent clinical validation.
