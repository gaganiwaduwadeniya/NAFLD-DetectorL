"""Synthetic metadata only: responses must never change data or release holds."""
import copy
import csv
import hashlib
import json

import pytest

from V4.src.team_clarification import (
    build_conflict_table, main, read_response_csv, validate_responses,
)


def fixture_audit():
    def row(image_id, patient, grade, filename, digest):
        return {'image_id': image_id, 'patient_id': patient, 'grade': grade,
                'relative_image_path': f'frames/{patient}/{filename}',
                'source_sha256': digest, 'decoded_pixel_sha256': digest}
    first, second, within = 'a' * 64, 'b' * 64, 'c' * 64
    return {'labels': {'P001': 0, 'P002': 3, 'P003': 1},
            'summary': {'annotation_sha256': 'd' * 64},
            'images': [row('A', 'P001', 0, 'a.jpg', first),
                       row('B', 'P002', 3, 'b.jpg', first),
                       row('C', 'P001', 0, 'c.jpg', second),
                       row('D', 'P002', 3, 'd.jpg', second),
                       row('E', 'P003', 1, 'e.jpg', within),
                       row('F', 'P003', 1, 'f.jpg', within)]}


def answered(rows, patient='P002', grade='3'):
    return [dict(row, team_confirmed_patient=patient, team_confirmed_grade=grade,
                 team_resolution_notes='Synthetic test response, not a real clinical decision.')
            for row in rows]


def test_export_is_deterministic_and_only_lists_conflicting_cross_patient_pairs():
    audit = fixture_audit()
    before = copy.deepcopy(audit)
    rows = build_conflict_table(audit)
    assert len(rows) == 2
    assert rows[0]['patient_a_file'] == 'frames/P001/a.jpg'
    assert rows[0]['patient_b_file'] == 'frames/P002/b.jpg'
    assert rows[0]['patient_a_label'] == 'Normal'
    assert rows[0]['patient_b_label'] == 'Severe'
    assert all(row['team_confirmed_patient'] == row['team_confirmed_grade'] == '' for row in rows)
    audit['images'].reverse()
    assert build_conflict_table(audit) == rows
    audit['images'].reverse()
    assert audit == before


def test_pending_and_complete_responses_remain_proposals_and_do_not_mutate_inputs():
    audit = fixture_audit()
    expected = build_conflict_table(audit)
    pending = validate_responses(expected, expected, audit['labels'])
    assert pending['counts']['pending'] == 2 and pending['structurally_valid']
    assert not pending['all_mappings_complete']
    responses = answered(expected)
    snapshot = copy.deepcopy((audit, expected, responses))
    complete = validate_responses(expected, list(reversed(responses)), audit['labels'])
    assert complete['counts']['ready_for_manual_review'] == 2
    assert complete['all_mappings_complete'] and complete['structurally_valid']
    assert complete['labels_applied'] is False and complete['holds_released'] is False
    assert complete['gate_c_approved'] is False
    assert (audit, expected, responses) == snapshot


@pytest.mark.parametrize('kind', ['multiple_entries', 'different_file_bytes', 'label_mismatch'])
def test_export_refuses_conflicts_that_the_two_file_table_cannot_represent(kind):
    audit = fixture_audit()
    if kind == 'multiple_entries':
        audit['images'].append(dict(audit['images'][0], image_id='G', patient_id='P003', grade=1,
                                    relative_image_path='frames/P003/g.jpg'))
    elif kind == 'different_file_bytes':
        audit['images'][1]['source_sha256'] = 'e' * 64
    else:
        audit['labels']['P002'] = 2
    with pytest.raises(ValueError, match='more than two|Decoded-only|disagrees'):
        build_conflict_table(audit)


@pytest.mark.parametrize('field,value', [
    ('patient_a_file', 'frames/P001/another.jpg'),
    ('source_file_sha256', 'f' * 64),
    ('patient_a_grade', '3'),
])
def test_changed_source_identity_is_flagged(field, value):
    audit = fixture_audit()
    expected = build_conflict_table(audit)
    response = answered(expected)
    response[0][field] = value
    result = validate_responses(expected, response, audit['labels'])
    assert not result['structurally_valid']
    assert result['rows'][0]['status'] == 'invalid'
    assert any(field in error for error in result['rows'][0]['errors'])


@pytest.mark.parametrize('responses_kind', ['missing', 'duplicate', 'unknown'])
def test_pair_coverage_errors_cannot_become_complete(responses_kind):
    audit = fixture_audit()
    expected = build_conflict_table(audit)
    responses = answered(expected)
    if responses_kind == 'missing':
        responses.pop()
    elif responses_kind == 'duplicate':
        responses.append(dict(responses[0]))
    else:
        responses.append(dict(responses[0], pair_number='999'))
    result = validate_responses(expected, responses, audit['labels'])
    assert not result['structurally_valid'] and not result['all_mappings_complete']
    assert result['errors']


@pytest.mark.parametrize('patient,grade', [('P999', '3'), ('P002', '4'), ('P002', '3.0'), ('P002', '')])
def test_unknown_patient_invalid_grade_and_partial_mapping_are_flagged(patient, grade):
    audit = fixture_audit()
    expected = build_conflict_table(audit)
    result = validate_responses(expected, answered(expected, patient, grade), audit['labels'])
    assert result['counts']['invalid'] == 2 and not result['all_mappings_complete']


def test_inconsistent_proposed_patient_grades_are_detected_across_pairs():
    audit = fixture_audit()
    expected = build_conflict_table(audit)
    responses = answered(expected)
    responses[1]['team_confirmed_grade'] = '0'
    result = validate_responses(expected, responses, audit['labels'])
    assert result['counts']['invalid'] == 2
    assert all(any('inconsistent' in error.lower() for error in row['errors']) for row in result['rows'])


def test_free_text_exclusion_is_preserved_for_manual_interpretation():
    audit = fixture_audit()
    expected = build_conflict_table(audit)
    responses = [dict(row, team_resolution_notes='Exclude both entries; ownership could not be verified.')
                 for row in expected]
    result = validate_responses(expected, responses, audit['labels'])
    assert result['counts']['unstructured_response'] == 2
    assert not result['all_mappings_complete'] and not result['holds_released']
    assert result['rows'][0]['team_resolution_notes'] == responses[0]['team_resolution_notes']


@pytest.mark.parametrize('header', [
    'pair_number,team_confirmed_grade\n1,3\n',
    'pair_number,pair_number\n1,1\n',
])
def test_malformed_csv_schema_is_rejected(tmp_path, header):
    path = tmp_path / 'response.csv'
    path.write_text(header)
    with pytest.raises(ValueError, match='columns|header'):
        read_response_csv(path)


def test_cli_csv_round_trip_binds_inputs_preserves_files_and_refuses_overwrite(tmp_path):
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(fixture_audit()))
    source_csv = tmp_path / 'Dataset_Annotation.csv'
    source_csv.write_text('sentinel original annotation: do not modify\n')
    state = tmp_path / 'AGENT_STATE.json'
    state.write_text('{"full_dataset_processing_approved": false}')
    originals = {p: p.read_bytes() for p in (audit, source_csv, state)}
    template, result = tmp_path / 'team.csv', tmp_path / 'result.json'
    assert main(['export', '--audit', str(audit), '--output', str(template)]) == 0
    rows = read_response_csv(template)
    assert len(rows) == 2
    assert main(['check', '--audit', str(audit), '--response', str(template), '--output', str(result)]) == 0
    report = json.loads(result.read_text())
    assert report['audit_sha256'] == hashlib.sha256(audit.read_bytes()).hexdigest()
    assert report['response_sha256'] == hashlib.sha256(template.read_bytes()).hexdigest()
    assert report['counts']['pending'] == 2 and report['metadata_only'] is True
    assert report['annotation_sha256_from_audit'] == 'd' * 64
    with pytest.raises(SystemExit) as error:
        main(['export', '--audit', str(audit), '--output', str(template)])
    assert error.value.code == 2
    assert read_response_csv(template) == rows
    with pytest.raises(SystemExit):
        main(['export', '--audit', str(audit), '--output', str(source_csv)])
    for path, data in originals.items():
        assert path.read_bytes() == data

    rows[0]['patient_b_file'] = 'tampered.jpg'
    changed = tmp_path / 'changed.csv'
    with changed.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert main(['check', '--audit', str(audit), '--response', str(changed),
                 '--output', str(tmp_path / 'changed_result.json')]) == 2
