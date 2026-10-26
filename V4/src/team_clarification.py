"""Export and check label-clarification CSVs; never apply responses to data.

Uses an existing audit snapshot only. It does not open source images, establish
clinical truth, release holds, or authorize preprocessing. The two-file table
format deliberately refuses larger/decoded-only conflict groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

GRADE_NAMES = {'0': 'Normal', '1': 'Mild', '2': 'Moderate', '3': 'Severe'}
IDENTITY_FIELDS = (
    'pair_number', 'patient_a', 'patient_a_file', 'patient_a_grade', 'patient_a_label',
    'patient_b', 'patient_b_file', 'patient_b_grade', 'patient_b_label', 'source_file_sha256',
)
RESPONSE_FIELDS = ('team_confirmed_patient', 'team_confirmed_grade', 'team_resolution_notes')
TABLE_FIELDS = (
    *IDENTITY_FIELDS[:-1], 'issue', 'verification', 'source_file_sha256',
    'clarification_requested', 'status', *RESPONSE_FIELDS,
)


def build_conflict_table(audit):
    """Build the existing team-table format deterministically from audited metadata."""
    labels = {patient: str(grade) for patient, grade in audit['labels'].items()}
    groups = defaultdict(list)
    for row in audit['images']:
        if row.get('decoded_pixel_sha256'):
            groups[row['decoded_pixel_sha256']].append(row)
    pairs = []
    for group in groups.values():
        if len({r.get('patient_id') for r in group}) < 2 or len({r.get('grade') for r in group}) < 2:
            continue
        if len(group) != 2:
            raise ValueError('Conflict group has more than two files; a separate multi-entry review is required.')
        pair = sorted(group, key=lambda r: (r['patient_id'], r['relative_image_path']))
        for row in pair:
            if (labels.get(row['patient_id']) != str(row['grade'])
                    or str(row['grade']) not in GRADE_NAMES):
                raise ValueError('Conflict image mapping disagrees with the audited patient labels.')
            if not row['relative_image_path']:
                raise ValueError('Conflict source file path is missing.')
        if (pair[0]['source_sha256'] != pair[1]['source_sha256']
                or not re.fullmatch(r'[0-9a-f]{64}', pair[0]['source_sha256'])):
            raise ValueError('Decoded-only conflict or invalid file hashes; a separate review format is required.')
        pairs.append(pair)
    pairs.sort(key=lambda pair: tuple(r['relative_image_path'] for r in pair))
    rows = []
    for number, pair in enumerate(pairs, 1):
        a, b = pair
        ga, gb = str(a['grade']), str(b['grade'])
        rows.append({
            'pair_number': number,
            'patient_a': a['patient_id'], 'patient_a_file': a['relative_image_path'],
            'patient_a_grade': int(ga), 'patient_a_label': GRADE_NAMES[ga],
            'patient_b': b['patient_id'], 'patient_b_file': b['relative_image_path'],
            'patient_b_grade': int(gb), 'patient_b_label': GRADE_NAMES[gb],
            'issue': 'Byte-for-byte identical image assigned to two patients with conflicting grades: '
                     f'{ga} {GRADE_NAMES[ga]} versus {gb} {GRADE_NAMES[gb]}.',
            'verification': 'Original file bytes and decoded image pixels both match.',
            'source_file_sha256': a['source_sha256'],
            'clarification_requested': 'Confirm the source patient and examination for this image and the corresponding '
                                       'verified grade. State which dataset entries need correction or exclusion.',
            'status': 'Awaiting team clarification; no source labels or images changed.',
            **{field: '' for field in RESPONSE_FIELDS},
        })
    return rows


def validate_responses(expected, responses, labels):
    """Check response coverage, immutable identities, and proposed mappings only."""
    expected_by_number = {str(row['pair_number']): row for row in expected}
    if len(expected_by_number) != len(expected):
        raise ValueError('Expected table contains duplicated pair numbers.')
    results = {number: {'pair_number': row['pair_number'], 'status': 'missing', 'errors': [],
                        'patient_a_file': row['patient_a_file'], 'patient_b_file': row['patient_b_file'],
                        **{field: '' for field in RESPONSE_FIELDS}}
               for number, row in expected_by_number.items()}
    labels = {patient: str(grade) for patient, grade in labels.items()}
    errors, seen = [], set()
    for csv_row, response in enumerate(responses, 2):
        number = str(response.get('pair_number', '')).strip()
        if number not in expected_by_number:
            errors.append(f'CSV row {csv_row}: unknown pair number {number!r}.')
            continue
        result = results[number]
        if number in seen:
            result['errors'].append('Duplicated pair number in returned CSV.')
            result['status'] = 'invalid'
            continue
        seen.add(number)
        for field in RESPONSE_FIELDS:
            result[field] = str(response.get(field, '')).strip()
        for field in IDENTITY_FIELDS:
            if str(response.get(field, '')).strip() != str(expected_by_number[number][field]):
                result['errors'].append(f'Changed or missing source identity field: {field}.')
        patient, grade, notes = (result[field] for field in RESPONSE_FIELDS)
        if patient or grade:
            if patient not in labels:
                result['errors'].append('Confirmed patient must be a known audited patient ID.')
            if grade not in GRADE_NAMES:
                result['errors'].append('Confirmed grade must be exactly 0, 1, 2, or 3.')
            result['status'] = 'ready_for_manual_review'
            if patient in labels and grade in GRADE_NAMES:
                result['proposes_annotation_grade_change'] = grade != labels[patient]
        else:
            result['status'] = 'unstructured_response' if notes else 'pending'
        if result['errors']:
            result['status'] = 'invalid'

    proposed_by_patient = defaultdict(list)
    for number, result in results.items():
        if number not in seen:
            result['errors'].append('Expected pair is missing from returned CSV.')
        if result['status'] == 'ready_for_manual_review':
            proposed_by_patient[result['team_confirmed_patient']].append(result)
    for patient, proposed in proposed_by_patient.items():
        if len({row['team_confirmed_grade'] for row in proposed}) > 1:
            for row in proposed:
                row['errors'].append(f'Inconsistent proposed grades for {patient} across image pairs.')
                row['status'] = 'invalid'
    for number, result in results.items():
        errors.extend(f'Pair {number}: {error}' for error in result['errors'])
    counts = {status: 0 for status in ('pending', 'ready_for_manual_review', 'unstructured_response', 'invalid', 'missing')}
    counts.update(Counter(row['status'] for row in results.values()))
    return {
        'schema_version': 1, 'metadata_only': True,
        'expected_pairs': len(expected), 'submitted_rows': len(responses), 'counts': counts,
        'structurally_valid': not errors,
        'all_mappings_complete': bool(expected) and not errors and counts['ready_for_manual_review'] == len(expected),
        'labels_applied': False, 'holds_released': False, 'gate_c_approved': False,
        'interpretation': 'Completed mappings are team proposals for manual review. No clinical verification, '
                          'source mutation, hold release, or gate approval is performed. Notes-only responses '
                          '(including exclusion requests) require manual interpretation.',
        'errors': errors, 'rows': list(results.values()),
    }


def _parse_response_csv(data):
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    fields = [field.strip() for field in next(reader, [])]
    if len(fields) != len(set(fields)):
        raise ValueError('Duplicated CSV header columns.')
    missing = set(IDENTITY_FIELDS + RESPONSE_FIELDS) - set(fields)
    if missing:
        raise ValueError('Missing required CSV columns: ' + ', '.join(sorted(missing)))
    rows = []
    for number, cells in enumerate(reader, 2):
        if not cells or not any(cell.strip() for cell in cells):
            continue
        if len(cells) != len(fields):
            raise ValueError(f'CSV row {number} has the wrong number of columns.')
        rows.append(dict(zip(fields, cells)))
    return rows


def read_response_csv(path):
    return _parse_response_csv(Path(path).read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'check'):
        command = commands.add_parser(name)
        command.add_argument('--audit', required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
        if name == 'check':
            command.add_argument('--response', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        audit_bytes = args.audit.read_bytes()
        audit = json.loads(audit_bytes)
        expected = build_conflict_table(audit)
        if args.command == 'check':
            response_bytes = args.response.read_bytes()
            result = validate_responses(expected, _parse_response_csv(response_bytes), audit['labels'])
            result.update(audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
                          response_sha256=hashlib.sha256(response_bytes).hexdigest(),
                          annotation_sha256_from_audit=audit['summary']['annotation_sha256'],
                          evidence_scope='Existing audit snapshot; source images and current annotations are not reopened.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'export':
            with args.output.open('x', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
                writer.writeheader()
                writer.writerows(expected)
            print(f'Exported {len(expected)} conflict pairs; team-response fields are blank. Originals unchanged.')
            return 0
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write('\n')
        print(json.dumps({'counts': result['counts'], 'structurally_valid': result['structurally_valid'],
                          'all_mappings_complete': result['all_mappings_complete'], 'labels_applied': False}))
        return 0 if result['structurally_valid'] else 2
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
