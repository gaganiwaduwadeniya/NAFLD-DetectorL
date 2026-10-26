"""Validate approved preprocessing on the existing bounded review cohort only."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .audit_dataset import sha256, write_csv, write_json
from .build_intensity_review import require_crop_approval
from .build_review import configuration_hash, load_checked_source
from .preprocess import preprocess_decoded

PIPELINE_VERSION = 'approved-operations-pilot-1.0'


def require_intensity_approval(crop, intensity, approval):
    if not (approval.get('gate') == 'B' and approval.get('decision') == 'approved'
            and approval.get('candidate_id') == 'I0_preserve'
            and approval.get('config_sha256') == configuration_hash(intensity)
            and approval.get('approved_geometry_sha256') == configuration_hash(crop)
            and approval.get('approved_by') == 'user' and bool(approval.get('user_instruction', '').strip())
            and bool(approval.get('recorded_at_utc'))
            and 'bounded_pilot_preparation' in approval.get('permitted_scope', [])):
        raise ValueError('Matching user intensity approval for both configs and bounded pilot scope is required.')


def preprocess_review_batch(root, rows, crop, intensity, crop_approval, intensity_approval, modality_reviews):
    if len(rows) > 40 or len({r['patient_id'] for r in rows}) > 28:
        raise PermissionError('Full-dataset processing requires Gate C approval; this pilot runner permits at most 40 frames / 28 patients.')
    if not rows:
        raise ValueError('A nonempty reviewed pilot subset is required.')
    require_crop_approval(crop, crop_approval)
    require_intensity_approval(crop, intensity, intensity_approval)
    ids = {r['image_id'] for r in rows}
    if len(ids) != len(rows) or not ids.issubset(intensity_approval.get('reviewed_image_ids', [])):
        raise ValueError('Pilot IDs must be unique and previously reviewed.')
    root = Path(root).resolve()
    images, metadata = [], []
    for row in rows:
        modality = modality_reviews.get(row['image_id'], {})
        if (modality.get('source_sha256') != row['source_sha256'] or modality.get('grayscale_preview_allowed') is not True
                or modality.get('classification') != 'bmode_appearing_no_doppler_seen'):
            raise ValueError('Unreviewed modality cannot enter the pilot grayscale pipeline.')
        source, pixels = load_checked_source(root, row)
        result = preprocess_decoded(pixels, crop, intensity)
        tensor = result['tensor']
        if (tensor.dtype != np.float32 or tensor.shape != (224, 224, 3) or not np.isfinite(tensor).all()
                or tensor.min() < 0 or tensor.max() > 255 or not np.array_equal(tensor[..., 0], tensor[..., 1])
                or not np.array_equal(tensor[..., 1], tensor[..., 2])):
            raise ValueError('Preprocessing tensor contract failed.')
        if sha256(source) != row['source_sha256']:
            raise ValueError('Source changed during pilot preprocessing.')
        images.append(tensor)
        metadata.append({'image_id': row['image_id'], 'patient_id': row['patient_id'], 'array_index': len(images) - 1,
                         'source_sha256': row['source_sha256'], 'source_decoded_pixel_sha256': row['decoded_pixel_sha256'],
                         'tensor_pixel_sha256': hashlib.sha256(tensor.tobytes()).hexdigest(), 'shape': [224, 224, 3],
                         'dtype': 'float32', 'pixel_min': float(tensor.min()), 'pixel_max': float(tensor.max()),
                         'crop_config_sha256': configuration_hash(crop), 'intensity_config_sha256': configuration_hash(intensity),
                         'box_ltrb': result['box_ltrb'], 'privacy_mask_rectangles': result['privacy_mask_rectangles'],
                         'resize': result['resize'], 'quality_flags': row['quality_flags'], 'model_input_eligible': False})
    return np.stack(images), metadata


def confirmed_duplicate_exclusions(rows, resolution):
    """Validate recorded ownership clarification; never rewrite source patients/grades."""
    if resolution is None:
        return {}
    if not (isinstance(resolution, dict) and resolution.get('schema_version') == 1
            and resolution.get('confirmed_by') == 'user'
            and isinstance(resolution.get('user_instruction'), str) and resolution['user_instruction'].strip()
            and resolution.get('recorded_at_utc') and isinstance(resolution.get('pairs'), list)
            and resolution['pairs']):
        raise ValueError('Source resolution requires recorded user confirmation and exact duplicate pairs.')
    indexed = {row['image_id']: row for row in rows}
    excluded = {}
    for pair in resolution['pairs']:
        if not isinstance(pair, dict):
            raise ValueError('Invalid duplicate resolution pair.')
        discard, keeper = indexed.get(pair.get('exclude_image_id')), indexed.get(pair.get('keep_image_id'))
        if (not discard or not keeper or discard['image_id'] == keeper['image_id']
                or discard['image_id'] in excluded or discard['patient_id'] == keeper['patient_id']
                or discard.get('decode_status') != 'ok' or keeper.get('decode_status') != 'ok'
                or not pair.get('source_sha256') or not pair.get('decoded_pixel_sha256')
                or any(row['source_sha256'] != pair['source_sha256']
                       or row['decoded_pixel_sha256'] != pair['decoded_pixel_sha256'] for row in (discard, keeper))
                or keeper['patient_id'] != pair.get('canonical_patient_id')
                or type(pair.get('canonical_grade')) is not int or keeper['grade'] != pair['canonical_grade']):
            raise ValueError('Duplicate resolution does not match exact source hashes and canonical patient/grade.')
        excluded[discard['image_id']] = keeper['image_id']
    if set(excluded) & set(excluded.values()):
        raise ValueError('Duplicate resolution cannot also exclude a selected canonical copy.')
    return excluded


def build_preparation_plan(rows, held_image_ids=(), known_layouts=(), source_resolution=None):
    """Metadata-only proposal; never deletes, relabels, or preprocesses sources."""
    excluded = confirmed_duplicate_exclusions(rows, source_resolution)
    groups = defaultdict(list)
    for row in rows:
        if row.get('decoded_pixel_sha256') and row['image_id'] not in excluded:
            groups[row['decoded_pixel_sha256']].append(row)
    plan = []
    for row in sorted(rows, key=lambda r: r['image_id']):
        group = groups.get(row.get('decoded_pixel_sha256'), [row])
        patients = {r['patient_id'] for r in group if r.get('patient_id')}
        action, reason, keeper = 'candidate_for_preprocessing', 'Approved appearance operations; final pixel privacy/quality QC still required.', None
        if row['image_id'] in excluded:
            action = 'hold_resolved_duplicate_copy'
            reason = 'Confirmed misassigned exact copy; exclude from working cohort, preserve original and canonical patient copy.'
            keeper = excluded[row['image_id']]
        elif row.get('decode_status') != 'ok' or row.get('patient_id') is None or row.get('grade') is None:
            action, reason = 'hold_mapping_or_decode', 'Unresolved mapping or decode.'
        elif known_layouts and row.get('layout') not in known_layouts:
            action, reason = 'hold_unapproved_layout', 'Unknown layout requires full-field review.'
        elif len(patients) > 1:
            action, reason = 'hold_cross_patient_duplicate', 'Identical decoded pixels occur under multiple patient IDs; no label/ownership decision inferred.'
        elif row['image_id'] in held_image_ids:
            action, reason = 'hold_manual_review', 'Previously flagged visibility case; retain original and hold this frame for review.'
        elif len(group) > 1:
            keeper = min(r['image_id'] for r in group)
            if row['image_id'] != keeper:
                action, reason = 'hold_redundant_exact_copy', 'Propose one exact copy per patient, selected by smallest stable image ID.'
        plan.append({'image_id': row['image_id'], 'patient_id': row['patient_id'], 'grade': row['grade'],
                     'proposed_action': action, 'reason': reason, 'retained_exact_copy_image_id': keeper,
                     'policy_approved': False, 'model_input_eligible': False,
                     'pixel_privacy_quality_status': 'needs_final_QC_before_any_training_or_transfer'})
    return plan


def run_pilot(root, rows, crop, intensity, a, b, modality, gate_b_gallery, output):
    root, output, gate_b_gallery = Path(root).resolve(), Path(output).resolve(), Path(gate_b_gallery).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Pilot output must be separate from source tree.')
    if output.exists():
        raise ValueError('Existing pilot outputs are preserved; use a new directory.')
    if sha256(gate_b_gallery / 'intensity_candidates.jsonl') != b.get('gate_b_candidates_sha256'):
        raise ValueError('Gate B evidence differs from the approved snapshot.')
    start = time.perf_counter()
    first, metadata = preprocess_review_batch(root, rows, crop, intensity, a, b, modality)
    elapsed = time.perf_counter() - start
    second, repeated_metadata = preprocess_review_batch(root, rows, crop, intensity, a, b, modality)
    if not np.array_equal(first, second) or metadata != repeated_metadata:
        raise ValueError('Local preprocessing reproducibility failed.')
    for index, row in enumerate(rows):
        with Image.open(gate_b_gallery / row['image_id'] / 'I0_preserve_padded224.png') as preview:
            if not np.array_equal(first[index], np.asarray(preview).astype(np.float32)):
                raise ValueError('Pilot output differs from the intensity pixels approved at Gate B.')
    output.mkdir(parents=True)
    np.save(output / 'pilot_tensors.npy', first, allow_pickle=False)
    if not np.array_equal(np.load(output / 'pilot_tensors.npy', allow_pickle=False), first):
        raise ValueError('Saved pilot tensor round trip failed.')
    write_json(output / 'pilot_manifest.json', metadata)
    write_csv(output / 'pilot_manifest.csv', metadata)
    overview = Image.new('RGB', (6 * 224, ((len(rows) + 5) // 6) * 254), '#15191f')
    draw = ImageDraw.Draw(overview)
    cards = []
    for index, row in enumerate(rows):
        image = Image.fromarray(first[index].astype(np.uint8))
        image.save(output / f"{row['image_id']}.png")
        x, y = (index % 6) * 224, (index // 6) * 254
        draw.text((x + 3, y + 4), f"{row['patient_id']} / pilot only", fill='white')
        overview.paste(image, (x, y + 27))
        cards.append(f'<figure><img src="{row["image_id"]}.png" alt="Approved pilot pixels"><figcaption>{row["patient_id"]} / {row["image_id"]}</figcaption></figure>')
    overview.save(output / 'pilot_contact_sheet.png')
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Gate C pilot</title><style>body{font:16px system-ui;background:#10151b;color:white;margin:24px}main{display:flex;flex-wrap:wrap}figure{margin:12px}img{width:224px;height:224px}</style><h1>Gate C pilot — full-dataset approval pending</h1><p>' + str(len(rows)) + ' previously reviewed frames only. Exact approved C2/I0 pixels. Pilot arrays are for technical validation, not an approved training dataset. Duplicate/visibility/overlay cases remain flagged. No bulk preparation, transfer, or training has run.</p><main>' + ''.join(cards) + '</main></html>')
    result = {'pipeline_version': PIPELINE_VERSION, 'frames': len(rows), 'patients': len({r['patient_id'] for r in rows}),
              'batch_shape': list(first.shape), 'batch_dtype': str(first.dtype), 'pixel_min': float(first.min()),
              'pixel_max': float(first.max()), 'all_finite': True, 'channels_identical': True,
              'repeat_runs_bit_identical': True, 'pilot_matches_gate_b_pixels_exactly': True,
              'npy_round_trip_exact': True, 'pilot_tensors_file_sha256': sha256(output / 'pilot_tensors.npy'),
              'first_pass_seconds': elapsed, 'model_input_eligible': False, 'full_dataset_processing_approved': False,
              'cross_runtime_validation': 'not_run; local exact identity only; remote work requires Gate D'}
    write_json(output / 'pilot_results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'annotations', 'audit', 'subset', 'crop-config', 'crop-approval', 'intensity-config', 'intensity-approval', 'modality-reviews', 'gate-b-gallery', 'output']:
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    crop, intensity, a, b = map(read, [args.crop_config, args.intensity_config, args.crop_approval, args.intensity_approval])
    if (sha256(args.annotations) != b.get('source_csv_sha256') or sha256(args.subset) != b.get('review_subset_sha256')
            or sha256(args.audit.parent / 'image_manifest.csv') != b.get('image_manifest_sha256')):
        parser.error('Source/manifest/subset hashes differ from approved evidence.')
    audited, rows = read(args.audit), read(args.subset)
    indexed = {r['image_id']: r for r in audited['images']}
    reserved_file = args.audit.parent / 'reserved_patients.json'
    reserved = read(reserved_file) if reserved_file.exists() else []
    if audited['summary']['mapping_status'] == 'blocked' or any(indexed.get(r['image_id']) != r or r['patient_id'] in reserved for r in rows):
        parser.error('Pilot must match the audited reviewed cohort and exclude reserved patients.')
    result = run_pilot(args.root, rows, crop, intensity, a, b, read(args.modality_reviews), args.gate_b_gallery, args.output)
    print(json.dumps(result, indent=2))
    print('STOP at Gate C. Full-dataset materialization, remote work, and training remain disabled.')


if __name__ == '__main__':
    main()
