"""Build a local Gate D review bundle; never uploads or invokes a remote job."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .audit_dataset import sha256, write_json
from .build_review import load_checked_source
from .prepare_dataset import _quality_reasons, require_preparation_approval
from .preprocess import preprocess_decoded

RESOURCE_TITLE = 'NAFLD V4 Preprocess Smoke 20261019'
RESOURCE_SLUG = 'nafld-v4-preprocess-smoke-20261019'
NOTEBOOK_TITLE = 'NAFLD V4 Preprocess Check 20261019'
NOTEBOOK_SLUG = 'nafld-v4-preprocess-check-20261019'


def build_smoke_bundle(context, prepared_directory, selected_ids, output, owner=None):
    workspace = context['workspace']
    require_preparation_approval(context, json.loads((workspace / 'configs/preparation_approval.json').read_text()))
    prepared_directory, output = Path(prepared_directory).resolve(), Path(output).resolve()
    if (not 1 <= len(selected_ids) <= 6 or len(set(selected_ids)) != len(selected_ids)
            or (owner is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,49}', owner))):
        raise ValueError('Select one to six distinct prepared frames and a valid public owner slug.')
    root = context['root']
    if (output.exists() or output == root or output.is_relative_to(root) or root.is_relative_to(output)
            or output == prepared_directory or output.is_relative_to(prepared_directory)):
        raise ValueError('Use a new bundle directory outside source/prepared images.')
    status = json.loads((prepared_directory / 'run_status.json').read_text())
    if (status.get('status') != 'complete'
            or sha256(prepared_directory / 'output_manifest.json') != status.get('output_manifest_sha256')):
        raise ValueError('A complete, unchanged prepared output manifest is required.')
    summary = json.loads((prepared_directory / 'preparation_summary.json').read_text())
    if summary.get('bindings') != context['bindings'] or summary.get('mode') != 'approved_local_preparation':
        raise ValueError('Prepared outputs disagree with the current approved policy.')
    rows = {r['image_id']: r for r in json.loads((prepared_directory / 'output_manifest.json').read_text())}
    selected = []
    for image_id in selected_ids:
        row = rows.get(image_id, {})
        source = context['indexed'].get(image_id)
        if (row.get('status') != 'prepared' or not source
                or _quality_reasons(context, source, row.get('quality_review'))):
            raise ValueError('Only prepared, technically cleared images are eligible for a smoke proposal.')
        tensor_path = (prepared_directory / row['tensor_path']).resolve()
        if (not tensor_path.is_relative_to(prepared_directory) or sha256(tensor_path) != row['tensor_file_sha256']):
            raise ValueError('Prepared tensor path/hash differs from its manifest.')
        _, pixels = load_checked_source(root, source)
        processed = preprocess_decoded(pixels, context['crop'], context['intensity'])
        if not np.array_equal(processed['tensor'], np.load(tensor_path, allow_pickle=False)):
            raise ValueError('Selected prepared pixels differ from the current approved preprocessing.')
        selected.append((row, processed['deidentified'], tensor_path))
    dataset, kernel = output / 'dataset', output / 'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    for source, name in [(Path(__file__).with_name('preprocess.py'), 'preprocess.py'),
                         (Path(__file__).with_name('kaggle_smoke_runtime.py'), 'smoke_runtime.py'),
                         (context['files']['crop'], 'crop_config.json'),
                         (context['files']['intensity'], 'intensity_config.json'),
                         (context['files']['pipeline'], 'pipeline_config.json')]:
        shutil.copyfile(source, dataset / name)
    samples, local_mapping = [], []
    for index, (row, source, tensor) in enumerate(selected, 1):
        sample_id = f'S{index:02d}'
        source_name, expected_name = f'source_{sample_id}.png', f'expected_{sample_id}.npy'
        Image.fromarray(source).save(dataset / source_name)
        shutil.copyfile(tensor, dataset / expected_name)
        samples.append({'sample_id': sample_id, 'source_file': source_name, 'expected_file': expected_name,
                        'source_layout': row['layout']})
        local_mapping.append({'sample_id': sample_id, 'image_id': row['image_id'], 'patient_id': row['patient_id'],
                              'grade': row['grade'], 'source_sha256': row['source_sha256']})
    assets = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(dataset.iterdir())]
    manifest = {'schema_version': 1, 'purpose': 'Six-frame maximum preprocessing equivalence smoke test; no training.',
                'samples': samples, 'assets': assets, 'absolute_pixel_tolerance': 0,
                'source_representation': 'Privacy-masked RGB working copies at source dimensions; no EXIF or raw headers.',
                'layout_selection': 'Original audited pre-mask layout; privacy masking removes some automatic UI signatures.',
                'approved_crop_config_sha256': context['bindings']['crop_config_sha256'],
                'approved_intensity_config_sha256': context['bindings']['intensity_config_sha256']}
    write_json(dataset / 'smoke_manifest.json', manifest)
    manifest_hash, runtime_hash = sha256(dataset / 'smoke_manifest.json'), sha256(dataset / 'smoke_runtime.py')
    code = f'''from pathlib import Path
import hashlib, json
expected_manifest = {manifest_hash!r}
matches = [p.parent for p in Path('/kaggle/input').rglob('smoke_manifest.json')
           if hashlib.sha256(p.read_bytes()).hexdigest() == expected_manifest]
assert len(matches) == 1, 'Expected exactly one mounted, hash-matching smoke dataset.'
root = matches[0]
runtime = root / 'smoke_runtime.py'
assert hashlib.sha256(runtime.read_bytes()).hexdigest() == {runtime_hash!r}
namespace = {{'__name__': 'nafld_v4_smoke_runtime', '__file__': str(runtime)}}
exec(compile(runtime.read_text(), str(runtime), 'exec'), namespace)
result = namespace['run_smoke'](root, expected_manifest, Path('/kaggle/working/smoke_result'))
print(json.dumps(result, indent=2))
assert result['status'] == 'passed', 'Preprocessing differs from the approved local baseline.'
'''
    notebook = {'cells': [{'cell_type': 'markdown', 'id': 'scope', 'metadata': {},
                          'source': ['# NAFLD V4 preprocessing smoke test\n',
                                     'Private CPU job. Internet disabled. No training, model loading or downloads.\n']},
                         {'cell_type': 'code', 'id': 'smoke', 'execution_count': None, 'metadata': {},
                          'outputs': [], 'source': code.splitlines(keepends=True)}],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 5}
    write_json(kernel / 'smoke.ipynb', notebook)
    if owner:
        write_json(dataset / 'dataset-metadata.json', {'title': RESOURCE_TITLE, 'id': f'{owner}/{RESOURCE_SLUG}',
                   'licenses': [{'name': 'other'}],
                   'description': 'Restricted private research preprocessing comparison. Rights remain with original authors. No public redistribution license is granted.'})
        write_json(kernel / 'kernel-metadata.json', {'id': f'{owner}/{NOTEBOOK_SLUG}', 'title': NOTEBOOK_TITLE,
                   'code_file': 'smoke.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                   'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
                   'dataset_sources': [f'{owner}/{RESOURCE_SLUG}'], 'competition_sources': [],
                   'kernel_sources': [], 'model_sources': []})
    inventory = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                 for folder in (dataset, kernel) for p in sorted(folder.iterdir())]
    proposal = {'schema_version': 1, 'sample_count': len(samples), 'smoke_manifest_sha256': manifest_hash,
                'prepared_output_manifest_sha256': status['output_manifest_sha256'], 'bindings': context['bindings'],
                'owner': owner, 'proposed_dataset': f'{owner}/{RESOURCE_SLUG}' if owner else None,
                'proposed_notebook': f'{owner}/{NOTEBOOK_SLUG}' if owner else None,
                'dataset_private_required': True, 'notebook_private': True, 'gpu_requested': False,
                'internet_enabled': False, 'training_approved': False, 'remote_transfer_approved': False,
                'kaggle_execution_approved': False, 'requires_gate': 'D',
                'transfer_inventory': inventory, 'total_proposed_bytes': sum(p['bytes'] for p in inventory),
                'job_timeout_seconds': 300, 'maximum_remote_smoke_runs': 1,
                'pending_checks': ['Explicit Gate D approval of this exact package and job.',
                                   'Verify authenticated account matches owner and destinations do not already exist.',
                                   'Verify resulting dataset/notebook private settings before executing the job.']
                                  + ([] if owner else ['Public Kaggle owner username is missing.'])}
    write_json(output / 'local_sample_mapping.DO_NOT_UPLOAD.json', local_mapping)
    write_json(output / 'transfer_proposal.json', proposal)
    return proposal
