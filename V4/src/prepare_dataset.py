"""Local preparation and QC audit; full processing requires recorded Gate C approval.

Preflight reads metadata only. Pilot validation is limited to the previously
approved cohort. A quality-review template never counts as completed review.
No command changes source images, labels, approvals, training, or remote state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, __version__ as pillow_version

from .audit_dataset import load_labels, sha256, write_csv, write_json
from .build_intensity_review import require_crop_approval
from .build_review import configuration_hash, load_checked_source
from .pilot_preparation import build_preparation_plan, require_intensity_approval
from .preprocess import preprocess_decoded, validate_intensity_config

RUNNER_VERSION = 'local-preparation-1.0'
QC_FIELDS = ('privacy_status', 'diagnostic_text_status', 'geometry_status', 'visibility_status')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _verify_csv(path, expected):
    """Compare an audited JSON table with its existing CSV representation."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Invalid or duplicated manifest/plan CSV columns.')
        actual = list(reader)
    converted = [{key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                        else '' if value is None else str(value)) for key, value in row.items()}
                 for row in expected]
    if actual != converted:
        raise ValueError('Audited manifest/plan JSON and CSV do not match.')


def load_preparation(workspace, paths_file=None):
    """Verify source labels, plans, and appearance approvals without opening images."""
    workspace = Path(workspace).resolve()
    paths = _read(paths_file or workspace / 'configs/paths.local.json')
    location = lambda key, default=None: (workspace / paths.get(key, default)).resolve()
    audit_dir = location('audit_directory')
    plan_dir = location('preparation_plan_directory', 'manifests/gate_c_v1')
    files = {
        'annotations': location('annotation_csv'), 'audit': audit_dir / 'dataset_audit.json',
        'manifest': audit_dir / 'image_manifest.csv', 'plan': plan_dir / 'preparation_plan.json',
        'plan_csv': plan_dir / 'preparation_plan.csv',
        'crop': location('approved_crop_config', 'configs/preprocessing.approved.json'),
        'intensity': location('approved_intensity_config', 'configs/intensity.approved.json'),
        'pipeline': location('approved_pipeline_config', 'configs/pipeline.approved.json'),
        'policy': location('preparation_proposal', 'configs/preparation.proposed.json'),
        'crop_approval': location('crop_approval_record', 'configs/crop_approval.json'),
        'intensity_approval': location('intensity_approval_record', 'configs/intensity_approval.json'),
    }
    context = {key: _read(files[key]) for key in ('audit', 'plan', 'crop', 'intensity', 'pipeline',
                                                 'policy', 'crop_approval', 'intensity_approval')}
    context.update(workspace=workspace, root=location('source_root'), files=files, audit_directory=audit_dir,
                   pilot_directory=location('pilot_directory', 'artifacts/pilot_v1'))
    crop, intensity, pipeline, policy = (context[key] for key in ('crop', 'intensity', 'pipeline', 'policy'))
    require_crop_approval(crop, context['crop_approval'])
    require_intensity_approval(crop, intensity, context['intensity_approval'])
    validate_intensity_config(intensity)
    annotation_hash, manifest_hash = sha256(files['annotations']), sha256(files['manifest'])
    if (load_labels(files['annotations']) != context['audit']['labels']
            or context['audit']['summary']['annotation_sha256'] != annotation_hash
            or policy['source_csv_sha256'] != annotation_hash or policy['image_manifest_sha256'] != manifest_hash):
        raise ValueError('Source annotations or image manifest disagree with the proposed policy/audit.')
    for approval in (context['crop_approval'], context['intensity_approval']):
        if approval.get('source_csv_sha256') != annotation_hash or approval.get('image_manifest_sha256') != manifest_hash:
            raise ValueError('Source hashes differ from the approved appearance evidence.')
    if (policy.get('requires_gate') != 'C'
            or policy['approved_pipeline_sha256'] != configuration_hash(pipeline)
            or pipeline.get('approved_crop_config_sha256') != configuration_hash(crop)
            or pipeline.get('approved_intensity_config_sha256') != configuration_hash(intensity)
            or pipeline.get('output_image_shape') != [224, 224, 3]
            or pipeline.get('dtype') != 'float32' or pipeline.get('pixel_range') != [0, 255]):
        raise ValueError('Pipeline/policy differs from the approved appearance contract.')
    rows = context['audit']['images']
    ids = [row['image_id'] for row in rows]
    source_paths = [row['relative_image_path'] for row in rows]
    if len(set(ids)) != len(ids) or len(set(source_paths)) != len(source_paths):
        raise ValueError('Duplicate image IDs or source paths in audited manifest.')
    for row in rows:
        relative = PurePosixPath(row['relative_image_path'])
        if (not re.fullmatch(r'IMG_[0-9a-f]{16}', row['image_id']) or relative.is_absolute()
                or '..' in relative.parts or '\\' in row['relative_image_path']):
            raise ValueError('Invalid source path or stable image ID.')
        if (row['patient_id'] not in context['audit']['labels']
                or row['grade'] != context['audit']['labels'][row['patient_id']]):
            raise ValueError('Image patient/grade mapping disagrees with the source annotation.')
    _verify_csv(files['manifest'], rows)
    resolution_path = policy.get('source_resolution_file')
    resolution_hash = policy.get('source_resolution_file_sha256')
    if bool(resolution_path) != bool(resolution_hash):
        raise ValueError('Source resolution path/hash must be supplied together.')
    if resolution_path:
        files['source_resolution'] = (workspace / resolution_path).resolve()
        context['source_resolution'] = _read(files['source_resolution'])
        if (sha256(files['source_resolution']) != resolution_hash
                or context['source_resolution'].get('source_csv_sha256') != annotation_hash
                or context['source_resolution'].get('image_manifest_sha256') != manifest_hash):
            raise ValueError('Source resolution evidence differs from the current source/manifest hashes.')
    expected = build_preparation_plan(rows, policy['held_visibility_image_ids'], crop['layouts'],
                                      source_resolution=context.get('source_resolution'))
    if context['plan'] != expected or sha256(files['plan_csv']) != policy['proposed_plan_file_sha256']:
        raise ValueError('Preparation plan differs from the deterministic hold policy.')
    _verify_csv(files['plan_csv'], context['plan'])
    counts = dict(Counter(row['proposed_action'] for row in expected))
    candidates = {row['image_id'] for row in expected if row['proposed_action'] == 'candidate_for_preprocessing'}
    if (counts != policy['proposed_actions'] or len(candidates) != policy['maximum_candidate_frames']
            or len({row['patient_id'] for row in rows if row['image_id'] in candidates}) != policy['candidate_patients']):
        raise ValueError('Candidate counts disagree with the proposed policy.')
    context.update(rows=rows, indexed={row['image_id']: row for row in rows},
                   file_hashes={key: sha256(path) for key, path in files.items()})
    context['bindings'] = {
        'source_csv_sha256': annotation_hash, 'image_manifest_sha256': manifest_hash,
        'audit_file_sha256': sha256(files['audit']), 'preparation_plan_file_sha256': sha256(files['plan_csv']),
        'preparation_plan_json_sha256': sha256(files['plan']),
        'crop_config_sha256': configuration_hash(crop), 'intensity_config_sha256': configuration_hash(intensity),
        'pipeline_config_sha256': configuration_hash(pipeline), 'preparation_policy_sha256': configuration_hash(policy),
    }
    if resolution_path:
        context['bindings']['source_resolution_file_sha256'] = resolution_hash
    return context


def build_quality_queue(context):
    """No approval/QC decisions are inferred from the earlier sampled reviews."""
    reviews = []
    for planned in context['plan']:
        row = context['indexed'][planned['image_id']]
        reviews.append({
            'image_id': row['image_id'], 'patient_id': row['patient_id'], 'grade': row['grade'],
            'relative_image_path': row['relative_image_path'], 'source_sha256': row['source_sha256'],
            'layout': row['layout'], 'crop_config_sha256': context['bindings']['crop_config_sha256'],
            'intensity_config_sha256': context['bindings']['intensity_config_sha256'],
            'proposed_action': planned['proposed_action'], 'policy_reason': planned['reason'],
            'modality': 'pending', **{field: 'pending' for field in QC_FIELDS},
            'reviewed_by': '', 'reviewed_at_utc': '', 'notes': '',
        })
    return {'schema_version': 1, 'bindings': dict(context['bindings']),
            'scope': 'Local technical modality, privacy/diagnostic-text, geometry and visibility review; not clinical grading.',
            'reviews': reviews}


def preflight_summary(context):
    counts = Counter(row['proposed_action'] for row in context['plan'])
    return {'runner_version': RUNNER_VERSION, 'total_images': len(context['rows']),
            'candidate_frames': counts['candidate_for_preprocessing'],
            'candidate_patients': context['policy']['candidate_patients'],
            'held_by_policy': len(context['rows']) - counts['candidate_for_preprocessing'],
            'proposed_actions': dict(counts), 'images_decoded': 0, 'gate_c_approved': False,
            'bindings': context['bindings'], 'quality_review_status': 'All template rows pending; no final QC inferred.',
            'model_input_eligible': False}


def _require_unchanged(context):
    if any(sha256(context['files'][key]) != digest for key, digest in context['file_hashes'].items()):
        raise ValueError('An input annotation/config/manifest changed since preflight; reload and review.')
    objects = ['audit', 'plan', 'crop', 'intensity', 'pipeline', 'policy', 'crop_approval', 'intensity_approval']
    if 'source_resolution' in context:
        objects.append('source_resolution')
    for key in objects:
        if context[key] != _read(context['files'][key]):
            raise ValueError('An in-memory input changed since preflight; reload and review.')
    if (context['rows'] != context['audit']['images']
            or context['indexed'] != {row['image_id']: row for row in context['audit']['images']}):
        raise ValueError('In-memory source mapping changed since preflight.')


def require_preparation_approval(context, approval):
    if not (isinstance(approval, dict) and approval.get('gate') == 'C'
            and approval.get('decision') == 'approved' and approval.get('approved_by') == 'user'
            and isinstance(approval.get('user_instruction'), str) and approval['user_instruction'].strip()
            and approval.get('recorded_at_utc')
            and 'full_local_preparation' in approval.get('permitted_scope', [])
            and approval.get('bindings') == context['bindings']):
        raise PermissionError('Gate C: matching recorded user approval of full local preparation is required.')
    _require_unchanged(context)


def _quality_index(context, quality):
    if not isinstance(quality, dict) or quality.get('schema_version') != 1 or quality.get('bindings') != context['bindings']:
        raise ValueError('Quality review bundle has stale or missing source/config bindings.')
    reviews = quality.get('reviews')
    if not isinstance(reviews, list) or any(not isinstance(row, dict) for row in reviews):
        raise ValueError('Quality reviews must be a list of per-image records.')
    indexed = {}
    for review in reviews:
        image_id = review.get('image_id')
        if image_id not in context['indexed'] or image_id in indexed:
            raise ValueError('Quality review contains an unknown or duplicated image ID.')
        indexed[image_id] = review
    return indexed


def _quality_reasons(context, row, review):
    if not review:
        return ['No per-image quality review is recorded.']
    reasons = []
    for key, value in {'source_sha256': row['source_sha256'], 'layout': row['layout'],
                       'patient_id': row['patient_id'], 'grade': row['grade'],
                       'relative_image_path': row['relative_image_path'],
                       'crop_config_sha256': context['bindings']['crop_config_sha256'],
                       'intensity_config_sha256': context['bindings']['intensity_config_sha256']}.items():
        if review.get(key) != value:
            reasons.append('Stale or mismatched quality evidence: ' + key + '.')
    if review.get('modality') != 'bmode_appearing_no_doppler_seen':
        reasons.append('Modality not cleared for the approved B-mode grayscale pipeline.')
    for field in QC_FIELDS:
        if review.get(field) != 'pass':
            reasons.append(field + ' requires review.')
    if not isinstance(review.get('reviewed_by'), str) or not review['reviewed_by'].strip():
        reasons.append('A local reviewer must be recorded.')
    try:
        if datetime.fromisoformat(review.get('reviewed_at_utc', '')).utcoffset() is None:
            raise ValueError('Timezone missing.')
    except (TypeError, ValueError):
        reasons.append('A review timestamp with timezone must be recorded.')
    return reasons


def _new_output(context, output):
    output = Path(output).resolve()
    root = context['root']
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Preparation output must be separate from the source tree.')
    if output.exists():
        raise ValueError('Existing preparation outputs are preserved; use a new versioned directory.')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    write_json(output / 'run_status.json', {'status': 'in_progress', 'model_input_eligible': False})
    return output


def _prepare_pixels(context, row):
    source, pixels = load_checked_source(context['root'], row)
    processed = preprocess_decoded(pixels, context['crop'], context['intensity'])
    tensor = processed['tensor']
    if processed['layout'] != row['layout']:
        raise ValueError('Observable layout differs from the audited layout; needs review.')
    if (tensor.dtype != np.float32 or tensor.shape != (224, 224, 3) or not np.isfinite(tensor).all()
            or not 0 <= tensor.min() <= tensor.max() <= 255
            or not np.array_equal(tensor[..., 0], tensor[..., 1])
            or not np.array_equal(tensor[..., 1], tensor[..., 2])):
        raise ValueError('Prepared tensor violates the approved pixel contract.')
    if not np.any(tensor):
        raise ValueError('Empty all-zero model view requires review.')
    if sha256(source) != row['source_sha256']:
        raise ValueError('Source hash changed during preparation.')
    active = processed['cropped'].max(axis=2) > 8 if processed['cropped'].ndim == 3 else processed['cropped'] > 8
    gray = processed['gray']
    statistics = {'region': 'Unpadded source max-channel >8 proxy, not anatomical segmentation; never changes pixels.',
                  'proxy_pixel_count': int(active.sum()), 'proxy_fraction_of_crop': float(active.mean()),
                  'grayscale_quantiles_in_proxy': np.quantile(gray[active], [0, .05, .5, .95, 1]).tolist() if active.any() else None,
                  'zero_fraction_unpadded_crop': float((gray == 0).mean()),
                  'saturated_fraction_unpadded_crop': float((gray == 255).mean())}
    return tensor, {'layout': processed['layout'], 'box_ltrb': processed['box_ltrb'],
                    'privacy_mask_rectangles': processed['privacy_mask_rectangles'], 'resize': processed['resize'],
                    'pixel_min': float(tensor.min()), 'pixel_max': float(tensor.max()), 'statistics': statistics,
                    'reporting_flags': [] if active.any() else ['foreground_proxy_empty_not_a_clinical_quality_decision']}


def _write_frame(context, row, tensor, output):
    """Publish a frame only after lossless round trips and source rechecking."""
    with tempfile.TemporaryDirectory(prefix='.tmp_', dir=output / 'frames') as name:
        staging = Path(name)
        np.save(staging / 'tensor.npy', tensor, allow_pickle=False)
        Image.fromarray(tensor[..., 0].astype(np.uint8)).save(staging / 'image.png')
        if not np.array_equal(np.load(staging / 'tensor.npy', allow_pickle=False), tensor):
            raise RuntimeError('Saved tensor round trip failed; preparation aborted.')
        with Image.open(staging / 'image.png') as image:
            if not np.array_equal(np.asarray(image), tensor[..., 0].astype(np.uint8)):
                raise RuntimeError('Saved PNG round trip failed; preparation aborted.')
        if sha256(context['root'] / row['relative_image_path']) != row['source_sha256']:
            raise ValueError('Source hash changed before publishing the derivative.')
        metadata = {'tensor_path': f"frames/{row['image_id']}/tensor.npy",
                    'image_path': f"frames/{row['image_id']}/image.png",
                    'tensor_file_sha256': sha256(staging / 'tensor.npy'),
                    'image_file_sha256': sha256(staging / 'image.png'),
                    'tensor_pixel_sha256': hashlib.sha256(tensor.tobytes()).hexdigest(),
                    'tensor_shape': [224, 224, 3], 'tensor_dtype': 'float32', 'source_hash_unchanged': True}
        staging.rename(output / 'frames' / row['image_id'])
    return metadata


def _record(context, row, planned, review):
    return {key: row[key] for key in ('image_id', 'patient_id', 'grade', 'relative_image_path', 'source_sha256',
                                     'decoded_pixel_sha256')} | {
        'proposed_action': planned['proposed_action'], 'status': 'pending', 'reasons': [],
        'tensor_path': None, 'image_path': None, 'bindings': dict(context['bindings']),
        'quality_review': review, 'source_audit_flags': row['quality_flags'], 'model_input_eligible': False,
    }


def _finish(context, output, records, mode, extra=None):
    prepared = [row for row in records if row['status'] == 'prepared']
    summary = {'runner_version': RUNNER_VERSION, 'mode': mode, 'total_rows': len(records),
               'prepared_frames': len(prepared), 'held_frames': len(records) - len(prepared),
               'statuses': dict(Counter(row['status'] for row in records)),
               'prepared_frames_by_grade': dict(Counter(str(row['grade']) for row in prepared)),
               'prepared_patients': len({row['patient_id'] for row in prepared}),
               'source_preservation_checked_frames': len(prepared), 'bindings': context['bindings'],
               'model_input_eligible': False, 'training_approved': False, 'remote_transfer_approved': False,
               'full_dataset_processing_approved': mode == 'approved_local_preparation',
               'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'Pillow': pillow_version},
               'implementation_sha256': {name: sha256(Path(__file__).with_name(name))
                                         for name in ('prepare_dataset.py', 'preprocess.py', 'build_review.py', 'pilot_preparation.py')},
               **(extra or {})}
    write_json(output / 'output_manifest.json', records)
    fields = list(dict.fromkeys(key for record in records for key in record))
    write_csv(output / 'output_manifest.csv', records, fields=fields)
    write_json(output / 'quality_audit.json', {'summary': summary, 'rows': records})
    write_json(output / 'preparation_summary.json', summary)
    write_json(output / 'run_status.json', {'status': 'complete', 'model_input_eligible': False,
                                          'output_manifest_sha256': sha256(output / 'output_manifest.json')})
    return summary


def run_preparation(context, quality_reviews, approval, output):
    """Only approved candidates with current completed QC can produce derivatives."""
    require_preparation_approval(context, approval)
    quality = _quality_index(context, quality_reviews)
    output = _new_output(context, output)
    records = []
    try:
        write_json(output / 'approval_evidence.json', approval)
        write_json(output / 'quality_review_evidence.json', quality_reviews)
        for planned in context['plan']:
            row = context['indexed'][planned['image_id']]
            review = quality.get(row['image_id'])
            record = _record(context, row, planned, review)
            if planned['proposed_action'] != 'candidate_for_preprocessing':
                record.update(status='held_by_policy', reasons=[planned['reason']])
            elif reasons := _quality_reasons(context, row, review):
                record.update(status='held_quality_review', reasons=reasons)
            else:
                try:
                    tensor, details = _prepare_pixels(context, row)
                except (OSError, ValueError, SyntaxError) as error:
                    record.update(status='held_processing_failure', reasons=[str(error)])
                else:
                    try:
                        metadata = _write_frame(context, row, tensor, output)
                    except ValueError as error:
                        record.update(status='held_processing_failure', reasons=[str(error)])
                    else:
                        record.update(details, **metadata, status='prepared')
            records.append(record)
        _require_unchanged(context)
        return _finish(context, output, records, 'approved_local_preparation',
                       {'quality_review_sha256': configuration_hash(quality_reviews)})
    except Exception:
        write_json(output / 'run_status.json', {'status': 'failed', 'model_input_eligible': False,
                                              'reason': 'Preparation interrupted; outputs are incomplete and must not be consumed.'})
        raise


def validate_pilot(context, output):
    """Exercise the same writer on the approved 26-frame cohort, never final QC."""
    _require_unchanged(context)
    subset_file = context['audit_directory'] / 'review_subset_extended.json'
    rows = _read(subset_file)
    a, b = context['crop_approval'], context['intensity_approval']
    reserved_file = context['audit_directory'] / 'reserved_patients.json'
    reserved = _read(reserved_file) if reserved_file.exists() else []
    if (not rows or len(rows) > 40 or len({row['patient_id'] for row in rows}) > 28
            or len({row['image_id'] for row in rows}) != len(rows)
            or sha256(subset_file) != a.get('review_subset_sha256')
            or sha256(subset_file) != b.get('review_subset_sha256')
            or any(context['indexed'].get(row['image_id']) != row
                   or row['image_id'] not in b.get('reviewed_image_ids', [])
                   or row['patient_id'] in reserved for row in rows)):
        raise PermissionError('Pilot must be the bounded, previously approved development subset.')
    modality = _read(context['audit_directory'] / 'modality_review_for_gate_b.json')
    for row in rows:
        evidence = modality.get(row['image_id'], {})
        if (evidence.get('source_sha256') != row['source_sha256']
                or evidence.get('classification') != 'bmode_appearing_no_doppler_seen'
                or evidence.get('grayscale_preview_allowed') is not True):
            raise ValueError('Pilot modality review is missing or stale.')
    pilot_dir = context['pilot_directory']
    reference_info = _read(pilot_dir / 'pilot_results.json')
    if sha256(pilot_dir / 'pilot_tensors.npy') != reference_info['pilot_tensors_file_sha256']:
        raise ValueError('Stored pilot reference hash changed.')
    reference = np.load(pilot_dir / 'pilot_tensors.npy', allow_pickle=False)
    reference_rows = _read(pilot_dir / 'pilot_manifest.json')
    if ([row['image_id'] for row in reference_rows] != [row['image_id'] for row in rows]
            or reference.shape != (len(rows), 224, 224, 3) or reference.dtype != np.float32):
        raise ValueError('Stored pilot reference mapping/shape differs from the approved subset.')
    output = _new_output(context, output)
    plan = {row['image_id']: row for row in context['plan']}
    records = []
    try:
        for index, row in enumerate(rows):
            tensor, details = _prepare_pixels(context, row)
            if not np.array_equal(tensor, reference[index]):
                raise ValueError('Runner pixels differ from the previously approved pilot.')
            record = _record(context, row, plan[row['image_id']], {'status': 'pilot_only_not_final_QC'})
            record.update(details, **_write_frame(context, row, tensor, output), status='prepared')
            records.append(record)
        _require_unchanged(context)
        return _finish(context, output, records, 'technical_pilot_validation',
                       {'matches_approved_pilot_exactly': True, 'full_dataset_processing_approved': False,
                        'final_per_image_quality_approved': False,
                        'reference_tensors_file_sha256': reference_info['pilot_tensors_file_sha256']})
    except Exception:
        write_json(output / 'run_status.json', {'status': 'failed', 'model_input_eligible': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'validate-pilot', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
        command.add_argument('--paths', type=Path)
        command.add_argument('--output', required=True, type=Path)
        if name == 'run':
            command.add_argument('--quality-reviews', required=True, type=Path)
            command.add_argument('--approval', type=Path)
    args = parser.parse_args(argv)
    try:
        context = load_preparation(args.workspace, args.paths)
        if args.command == 'preflight':
            output = _new_output(context, args.output)
            summary = preflight_summary(context)
            queue = build_quality_queue(context)
            write_json(output / 'quality_reviews.pending.json', queue)
            write_csv(output / 'quality_reviews.pending.csv', queue['reviews'])
            write_json(output / 'preflight_summary.json', summary)
            write_json(output / 'run_status.json', {'status': 'complete', 'images_decoded': 0,
                                                  'model_input_eligible': False})
        elif args.command == 'validate-pilot':
            summary = validate_pilot(context, args.output)
        else:
            approval_file = args.approval or Path(args.workspace) / 'configs/preparation_approval.json'
            approval = _read(approval_file) if approval_file.is_file() else None
            # Approval is checked before the QC file is opened or any output/source image is created/read.
            require_preparation_approval(context, approval)
            summary = run_preparation(context, _read(args.quality_reviews), approval, args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
