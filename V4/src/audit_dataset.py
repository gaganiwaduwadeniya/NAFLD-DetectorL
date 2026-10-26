"""Read-only source audit and reproducible development-only review selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
from .preprocess import classify_layout

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp', '.dcm', '.dicom', '.avi', '.mp4'}
EXPECTED_COUNTS = {'0': 11, '1': 35, '2': 46, '3': 18}
EXPECTED_CSV_HASH = 'b3c437b2ebb514f6b9cd9f5178dc57e0cd4cd9cf90ea3935f8f757660b32a479'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def load_labels(path):
    labels = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = [h.strip() for h in (reader.fieldnames or [])]
        required = ['Patient ID', 'NAFLD', 'NON-NAFLD', *[f'STAGE {g}' for g in range(4)]]
        if len(fields) != len(set(fields)) or set(fields) != set(required):
            raise ValueError('Unexpected or duplicated annotation columns.')
        reader.fieldnames = fields
        for number, row in enumerate(reader, 2):
            pid = row.get('Patient ID', '').strip()
            if not re.fullmatch(r'P\d{3}', pid) or pid == 'P000' or pid in labels:
                raise ValueError(f'Invalid or duplicate patient ID at CSV row {number}.')
            if None in row or any(row.get(k) is None for k in required):
                raise ValueError(f'Wrong column count at CSV row {number}.')
            flags = [row[k].strip() for k in required[1:]]
            if any(v not in {'0', '1'} for v in flags):
                raise ValueError(f'Nonbinary flag at CSV row {number}.')
            stages = [int(row[f'STAGE {g}'].strip()) for g in range(4)]
            if sum(stages) != 1:
                raise ValueError(f'Zero or multiple selected grades at CSV row {number}.')
            grade = stages.index(1)
            if int(row['NAFLD']) != int(grade > 0) or int(row['NON-NAFLD']) != int(grade == 0):
                raise ValueError(f'Inconsistent binary flags at CSV row {number}.')
            labels[pid] = grade
    if not labels:
        raise ValueError('Annotation contains no patients.')
    return labels


def map_patient(relative_path):
    tokens = set(re.findall(r'(?<![A-Za-z0-9])P\d{3}(?![A-Za-z0-9])', Path(relative_path).as_posix(), re.IGNORECASE))
    tokens = {t.upper() for t in tokens}
    if len(tokens) > 1:
        raise ValueError('Conflicting patient tokens in image path.')
    return next(iter(tokens), None)


def pixel_digest(pixels):
    header = f'{pixels.dtype}:{pixels.shape}:'.encode()
    return hashlib.sha256(header + pixels.tobytes()).hexdigest()


def difference_hash(pixels):
    # Integrity screening only; this conversion never enters saved image previews.
    small = np.asarray(Image.fromarray(pixels).convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return f'{int.from_bytes(np.packbits(bits).tobytes(), "big"):016x}'


def audit(root, annotation_path):
    root = Path(root).resolve()
    annotation_path = Path(annotation_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError('New dataset source directory is absent.')
    labels = load_labels(annotation_path)
    rows = []
    unsupported = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Source image symlink escapes the dataset.')
        rel = path.relative_to(root).as_posix()
        flags = []
        try:
            pid = map_patient(Path(rel))
        except ValueError:
            pid = None
            flags.append('conflicting_patient_tokens')
        if pid is None:
            flags.append('unmapped_patient')
        elif pid not in labels:
            flags.append('patient_absent_from_annotation')
        row = dict(image_id='IMG_' + hashlib.sha256(rel.encode()).hexdigest()[:16], patient_id=pid,
                   examination_id=None, relative_image_path=rel, grade=labels.get(pid),
                   source_sha256=sha256(path), decoded_pixel_sha256=None, width=None, height=None,
                   channels=None, dtype=None, source_format=None, decode_status='failed',
                   source_group=path.parent.parent.relative_to(root).as_posix() if path.parent != root else '.',
                   quality_flags=flags, label_scope='inherited_patient_grade_not_independent_frame_read',
                   scanner_vendor=None, orientation=None, layout=None, dhash=None)
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['source_sha256']:
                raise ValueError('Source changed while auditing.')
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                pixels = np.asarray(image)
                row.update(width=image.width, height=image.height, source_format=image.format,
                           channels=len(image.getbands()), dtype=str(pixels.dtype), mode=image.mode,
                           orientation=int(image.getexif().get(274, 1)), frame_count=getattr(image, 'n_frames', 1),
                           compression=image.info.get('compression'),
                           has_icc_profile=bool(image.info.get('icc_profile')))
                row.update(decoded_pixel_sha256=pixel_digest(pixels), decode_status='ok',
                           layout=classify_layout(pixels),
                           pixel_min=int(pixels.min()), pixel_max=int(pixels.max()),
                           pixel_quantiles=[float(v) for v in np.quantile(pixels, [0, .01, .5, .99, 1])],
                           aspect_ratio=image.width / image.height)
                if pixels.dtype == np.uint8 and image.mode in {'L', 'RGB'}:
                    row['dhash'] = difference_hash(pixels)
                    if pixels.ndim == 3:
                        colored = pixels.max(axis=2).astype(int) - pixels.min(axis=2).astype(int) > 20
                        row['chromatic_pixel_fraction'] = float(colored.mean())
                        if colored.mean() > .001:
                            flags.append('color_overlay_or_doppler_needs_review')
                else:
                    flags.append('unsupported_preview_mode_or_bit_depth')
                if row['orientation'] != 1:
                    flags.append('nonstandard_orientation_no_auto_rotation')
                if row['frame_count'] != 1:
                    flags.append('multiframe_requires_format_specific_review')
                flags.append('burned_in_overlay_status_unreviewed')
        except (OSError, ValueError, SyntaxError):
            flags.append('decode_failed_or_format_specific_review_needed')
            unsupported.append(row['image_id'])
        rows.append(row)

    exact = defaultdict(list)
    for row in rows:
        if row['decoded_pixel_sha256']:
            exact[row['decoded_pixel_sha256']].append(row)
    duplicate_groups = []
    for digest, group in exact.items():
        if len(group) < 2:
            continue
        patients = {r['patient_id'] for r in group}
        cross = len(patients - {None}) > 1
        duplicate_groups.append(dict(decoded_pixel_sha256=digest, image_ids=[r['image_id'] for r in group],
                                     patient_ids=sorted(p for p in patients if p), cross_patient=cross,
                                     conflicting_grades=len({r['grade'] for r in group}) > 1))
        for row in group:
            row['quality_flags'].append('cross_patient_exact_duplicate_quarantine' if cross else 'within_patient_exact_duplicate')
    near = []
    hashed = [(row, int(row['dhash'], 16)) for row in rows if row['dhash']]
    for i, (a, ah) in enumerate(hashed):
        for b, bh in hashed[i + 1:]:
            if (a['width'], a['height']) != (b['width'], b['height']) or a['decoded_pixel_sha256'] == b['decoded_pixel_sha256']:
                continue
            distance = (ah ^ bh).bit_count()
            if distance <= 3:
                near.append(dict(image_id_a=a['image_id'], image_id_b=b['image_id'], patient_id_a=a['patient_id'],
                                 patient_id_b=b['patient_id'], hamming_distance=distance,
                                 cross_patient=a['patient_id'] != b['patient_id'], status='candidate_not_proven_duplicate'))
    present = {r['patient_id'] for r in rows if r['patient_id'] in labels}
    decoded = {r['patient_id'] for r in rows if r['patient_id'] in labels and r['decode_status'] == 'ok'}
    invalid = [r for r in rows if r['patient_id'] not in labels or r['decode_status'] != 'ok']
    counts = {str(g): sum(v == g for v in labels.values()) for g in range(4)}
    summary = dict(patients_in_csv=len(labels), images=len(rows), patient_label_counts=counts,
                   label_counts_match_handoff=counts == EXPECTED_COUNTS,
                   annotation_sha256=sha256(annotation_path), annotation_matches_handoff_hash=sha256(annotation_path) == EXPECTED_CSV_HASH,
                   mapped_patients=len(present), missing_patient_images=sorted(set(labels) - present),
                   patients_without_decodable_images=sorted(set(labels) - decoded),
                   unexpected_patient_ids=sorted(set(labels) - {f'P{i:03d}' for i in range(1, 111)}),
                   decode_failures=sum(r['decode_status'] != 'ok' for r in rows),
                   unmapped_images=sum(r['patient_id'] not in labels for r in rows),
                   mapping_status='verified_by_explicit_patient_id_tokens' if not invalid and present == set(labels) else 'blocked',
                   patient_identity='user_confirmed', clinical_label_provenance='not_documented_in_supplied_files',
                   frames_per_patient=dict(sorted(Counter(r['patient_id'] for r in rows if r['patient_id'] in labels).items())),
                   dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows if r['width'])),
                   layouts=dict(Counter(r['layout'] for r in rows if r['layout'])),
                   formats=dict(Counter(r['source_format'] for r in rows if r['source_format'])),
                   modes=dict(Counter(r.get('mode') for r in rows if r.get('mode'))),
                   source_groups_by_grade={str(g): dict(Counter(r['source_group'] for r in rows if r['grade'] == g)) for g in range(4)},
                   layouts_by_grade={str(g): dict(Counter(r['layout'] for r in rows if r['grade'] == g)) for g in range(4)},
                   exact_duplicate_groups=len(duplicate_groups),
                   source_byte_duplicate_groups=sum(v > 1 for v in Counter(r['source_sha256'] for r in rows).values()),
                   cross_patient_exact_duplicate_groups=sum(g['cross_patient'] for g in duplicate_groups),
                   likely_near_duplicate_pairs=len(near), cross_patient_near_duplicate_pairs=sum(p['cross_patient'] for p in near),
                   near_duplicate_rule='64-bit full-display dHash Hamming <=3, same dimensions; candidate only',
                   software={'numpy': np.__version__, 'Pillow': pillow_version},
                   reserved_cohort_status='No V4 reserved cohort manifest found; legacy Test_Data excluded entirely')
    return {'summary': summary, 'labels': labels, 'images': rows, 'exact_duplicates': duplicate_groups, 'near_duplicates': near}


def refresh_layouts(root, annotation_path, previous):
    """Reuse unchanged metadata; verify every source byte and decoded-pixel hash.

    This avoids repeating the earlier quantile and duplicate calculations when
    refining a display-layout signature after visual inspection.
    """
    root = Path(root).resolve()
    if sha256(annotation_path) != previous['summary']['annotation_sha256'] or load_labels(annotation_path) != previous['labels']:
        raise ValueError('Annotations changed; perform a fresh audit.')
    result = json.loads(json.dumps(previous))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if actual != {r['relative_image_path'] for r in result['images']}:
        raise ValueError('Source file inventory changed; perform a fresh audit.')
    for index, row in enumerate(result['images']):
        path = root / row['relative_image_path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['source_sha256']:
            raise ValueError('Source bytes changed; perform a fresh audit.')
        with Image.open(io.BytesIO(raw)) as im:
            pixels = np.asarray(im)
        if pixel_digest(pixels) != row['decoded_pixel_sha256']:
            raise ValueError('Decoded source changed; perform a fresh audit.')
        row['layout'] = classify_layout(pixels)
        if index % 250 == 0:
            print(f'Verified layout metadata for {index + 1} source images.', flush=True)
    result['summary']['layouts'] = dict(Counter(r['layout'] for r in result['images']))
    result['summary']['layouts_by_grade'] = {str(g): dict(Counter(r['layout'] for r in result['images'] if r['grade'] == g)) for g in range(4)}
    result['summary']['source_byte_duplicate_groups'] = sum(v > 1 for v in Counter(r['source_sha256'] for r in result['images']).values())
    result['summary']['layout_signature_version'] = 'amber-footer-or-yellow-marker-v1'
    return result


def select_subset(rows, labels, seed=42, reserved=(), per_grade=5):
    """One frame per development patient, balance grades, prefer layout coverage.

    Selection is stable under file enumeration changes. Within a layout pick
    challenging flagged examples first, then seed-shuffled patients. No scores.
    """
    rng = random.Random(seed)
    by_patient = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['image_id']):
        if row['patient_id'] in labels and row['patient_id'] not in reserved and row['decode_status'] == 'ok':
            by_patient[row['patient_id']].append(row)
    selected = []
    for grade in range(4):
        patients = sorted(pid for pid in by_patient if labels[pid] == grade)
        rng.shuffle(patients)
        seen_layouts = set()
        while patients and sum(labels[r['patient_id']] == grade for r in selected) < per_grade:
            index = next((i for i, pid in enumerate(patients) if any(r['layout'] not in seen_layouts for r in by_patient[pid])), 0)
            pid = patients.pop(index)
            candidates = by_patient[pid]
            row = max(candidates, key=lambda r: (r['layout'] not in seen_layouts, len(r['quality_flags']), r['image_id']))
            selected.append(dict(row))
            seen_layouts.add(row['layout'])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reuse-audit', type=Path, help='Reclassify observable layouts only after verifying unchanged inventory, source bytes, and decoded pixels')
    parser.add_argument('--reserved-patients', type=Path, help='JSON list of V4 held-out patient IDs; excluded from preview selection')
    args = parser.parse_args()
    if args.output.resolve() == args.root.resolve() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error('Audit output must be outside source tree.')
    if args.output.exists():
        parser.error('Use a new output directory; existing audits are preserved.')
    result = refresh_layouts(args.root, args.annotations, json.loads(args.reuse_audit.read_text())) if args.reuse_audit else audit(args.root, args.annotations)
    reserved = json.loads(args.reserved_patients.read_text()) if args.reserved_patients else []
    subset = select_subset(result['images'], result['labels'], reserved=reserved)
    write_json(args.output / 'dataset_audit.json', result)
    fields = sorted(set().union(*(r.keys() for r in result['images']))) if result['images'] else ['image_id']
    write_csv(args.output / 'image_manifest.csv', [{k: r.get(k) for k in fields} for r in result['images']], fields)
    write_csv(args.output / 'patient_labels.csv', [dict(patient_id=p, grade=g) for p, g in sorted(result['labels'].items())])
    write_csv(args.output / 'review_subset.csv', [dict(patient_id=r['patient_id'], image_id=r['image_id'], grade=r['grade'],
              relative_image_path=r['relative_image_path'], selection_seed=42, usage='preprocessing_development') for r in subset],
              ['patient_id', 'image_id', 'grade', 'relative_image_path', 'selection_seed', 'usage'])
    write_json(args.output / 'review_subset.json', subset)
    write_json(args.output / 'reserved_patients.json', reserved)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
