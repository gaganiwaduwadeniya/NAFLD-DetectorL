"""Standalone preprocessing comparison. No training, downloads or Kaggle API."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def replay_masked_source(pixels, crop, intensity, source_layout, primitives):
    """Use recorded pre-mask geometry; the privacy mask can erase its UI signature.

    This exercises the approved module's pixel operations, not fresh raw-layout
    detection. The sample layout was observed on the original before masking.
    """
    primitives['require_preview_dtype'](pixels)
    primitives['validate_intensity_config'](intensity)
    layout = crop.get('layouts', {}).get(source_layout, {})
    if (crop.get('output_size', 224) != 224 or layout.get('privacy_reviewed') is not True
            or layout.get('selector') != {'width': pixels.shape[1], 'height': pixels.shape[0]}
            or not layout.get('template_box') or not layout.get('mask_rectangles')):
        raise ValueError('Recorded source layout does not match approved dimensions/geometry.')
    masked = primitives['mask_pixels'](pixels, layout['mask_rectangles'])
    cropped = primitives['crop_pixels'](masked, layout['template_box'])
    gray = primitives['grayscale_preserve'](cropped)
    return primitives['grayscale_preview_tensor'](gray, 224)[0]


def run_smoke(input_dir, expected_manifest_sha256, output_dir):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    manifest_file = root / 'smoke_manifest.json'
    if digest(manifest_file) != expected_manifest_sha256:
        raise ValueError('Smoke manifest hash differs from the reviewed package.')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    names = [item['path'] for item in manifest['assets']]
    sample_ids = [item['sample_id'] for item in manifest['samples']]
    if (manifest.get('schema_version') != 1 or not 1 <= len(manifest['samples']) <= 6
            or len(set(names)) != len(names)
            or len(set(sample_ids)) != len(sample_ids)
            or any(not re.fullmatch(r'S0[1-6]', item) for item in sample_ids)
            or any(Path(name).name != name or name.startswith('.') for name in names)):
        raise ValueError('Invalid smoke manifest paths or sample count.')
    allowed = set(names) | {'smoke_manifest.json', 'dataset-metadata.json'}
    if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in root.iterdir()):
        raise ValueError('Dataset directory differs from the reviewed file whitelist.')
    for item in manifest['assets']:
        path = root / item['path']
        if not path.is_file() or path.is_symlink() or digest(path) != item['sha256']:
            raise ValueError('Smoke asset hash mismatch: ' + item['path'])
    required = {'preprocess.py', 'smoke_runtime.py', 'crop_config.json', 'intensity_config.json', 'pipeline_config.json'}
    for sample in manifest['samples']:
        required.update((sample['source_file'], sample['expected_file']))
    if required != set(names):
        raise ValueError('Smoke manifest must list exactly the approved assets.')
    if output == root or output.is_relative_to(root) or root.is_relative_to(output) or output.exists():
        raise ValueError('Use a new output directory outside the input dataset.')
    # Execute only the hash-verified copy of the same preprocessing module.
    # No input-tree bytecode/cache writes, including during local validation.
    namespace = {'__name__': 'nafld_v4_verified_preprocess', '__file__': str(root / 'preprocess.py')}
    exec(compile((root / 'preprocess.py').read_text(encoding='utf-8'), str(root / 'preprocess.py'), 'exec'), namespace)
    crop = json.loads((root / 'crop_config.json').read_text())
    intensity = json.loads((root / 'intensity_config.json').read_text())
    output.mkdir(parents=True)
    records = []
    for sample in manifest['samples']:
        with Image.open(root / sample['source_file']) as image:
            if image.mode != 'RGB' or image.getexif() or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Smoke sources must be plain RGB single-frame working copies.')
            source = np.asarray(image).copy()
        actual = replay_masked_source(source, crop, intensity, sample['source_layout'], namespace)
        expected = np.load(root / sample['expected_file'], allow_pickle=False)
        if (actual.shape != (224, 224, 3) or actual.dtype != np.float32
                or expected.shape != actual.shape or expected.dtype != np.float32
                or not np.isfinite(actual).all() or not np.isfinite(expected).all()
                or not 0 <= actual.min() <= actual.max() <= 255
                or not np.array_equal(actual[..., 0], actual[..., 1])
                or not np.array_equal(actual[..., 1], actual[..., 2])):
            raise ValueError('Smoke tensor violates the approved shape/dtype/range/channel contract.')
        difference = float(np.max(np.abs(actual - expected)))
        path = output / f"actual_{sample['sample_id']}.npy"
        np.save(path, actual, allow_pickle=False)
        records.append({'sample_id': sample['sample_id'], 'exact_match': bool(np.array_equal(actual, expected)),
                        'maximum_absolute_pixel_difference': difference, 'actual_tensor_sha256': digest(path),
                        'actual_tensor_pixel_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
    gpu_tool = shutil.which('nvidia-smi')
    inventory = {'available': bool(gpu_tool), 'devices': None}
    if gpu_tool:
        try:
            proc = subprocess.run([gpu_tool, '-L'], capture_output=True, text=True, timeout=10, check=False)
            inventory.update(exit_code=proc.returncode, devices=proc.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            inventory['query_status'] = 'unavailable'
    report = {'schema_version': 1, 'status': 'passed' if all(r['exact_match'] for r in records) else 'failed',
              'matched_samples': sum(r['exact_match'] for r in records), 'samples': records,
              'maximum_absolute_pixel_difference': max(r['maximum_absolute_pixel_difference'] for r in records),
              'permitted_absolute_pixel_difference': 0, 'manifest_sha256': expected_manifest_sha256,
              'input_directory': str(root), 'output_directory': str(output),
              'runtime_environment': 'Kaggle' if str(root).startswith('/kaggle/input/') else 'local',
              'software': {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': pillow_version},
              'gpu_requested': False, 'observed_gpu_inventory': inventory,
              'layout_selection': 'Original pre-mask audited layout metadata; this does not test automatic raw-layout detection.',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'training_executed': False, 'model_loaded': False}
    (output / 'smoke_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report
