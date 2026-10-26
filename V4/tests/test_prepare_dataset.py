"""Full runner checks use synthetic sources and approvals only; no real cohort is prepared."""
import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from V4.src.audit_dataset import audit, sha256, write_csv, write_json
from V4.src.build_review import configuration_hash
from V4.src.pilot_preparation import build_preparation_plan
from V4.src.prepare_dataset import build_quality_queue, load_preparation, main, run_preparation, validate_pilot
from V4.src.preprocess import preprocess_decoded


def preparation_fixture(tmp_path, cross_patient=False):
    workspace = tmp_path / 'V4'
    raw = workspace / 'raw'
    raw.mkdir(parents=True)
    annotations = raw / 'Dataset_Annotation.csv'
    annotations.write_text('Patient ID,NAFLD,NON-NAFLD ,STAGE 0,STAGE 1,STAGE 2,STAGE 3\n'
                           'P001,0,1,1,0,0,0\nP002,1,0,0,0,0,1\nP003,1,0,0,1,0,0\n')
    for pid, level in [('P001', 60), ('P002', 90), ('P003', 120)]:
        (raw / pid).mkdir()
        Image.fromarray(np.full((60, 100, 3), level, dtype=np.uint8)).save(raw / pid / 'frame.png')
    # A within-patient exact copy produces one held row; keepers are deterministic.
    (raw / 'P003/copy.png').write_bytes((raw / 'P003/frame.png').read_bytes())
    if cross_patient:
        (raw / 'P002/frame.png').write_bytes((raw / 'P001/frame.png').read_bytes())
    audited = audit(raw, annotations)
    audit_dir = workspace / 'manifests/audit'
    write_json(audit_dir / 'dataset_audit.json', audited)
    write_csv(audit_dir / 'image_manifest.csv', audited['images'])
    write_json(audit_dir / 'reserved_patients.json', [])
    crop = {'version': 'synthetic_fixture', 'output_size': 224, 'layouts': {'100x60': {
        'selector': {'width': 100, 'height': 60}, 'privacy_reviewed': True,
        'mask_rectangles': [[0, 0, 100, 10]], 'template_box': [0, 10, 100, 60]}}}
    intensity = {'version': 'synthetic_fixture', 'candidates': ['I0_preserve'],
                 'optional_intensity_operation': 'identity',
                 'grayscale_method': 'BT601_integer_299_587_114_half_up', 'output_size': 224}
    annotation_hash, manifest_hash = sha256(annotations), sha256(audit_dir / 'image_manifest.csv')
    crop_approval = {'gate': 'A', 'decision': 'approved', 'candidate_id': 'C2_layout_specific_crop',
                     'config_sha256': configuration_hash(crop), 'approved_by': 'user',
                     'user_instruction': 'SYNTHETIC TEST APPROVAL ONLY', 'recorded_at_utc': '2026-10-19T00:00:00Z',
                     'permitted_scope': ['crop_geometry', 'prepare_gate_b_intensity_previews'],
                     'source_csv_sha256': annotation_hash, 'image_manifest_sha256': manifest_hash}
    intensity_approval = {'gate': 'B', 'decision': 'approved', 'candidate_id': 'I0_preserve',
                         'config_sha256': configuration_hash(intensity), 'approved_geometry_sha256': configuration_hash(crop),
                         'approved_by': 'user', 'user_instruction': 'SYNTHETIC TEST APPROVAL ONLY',
                         'recorded_at_utc': '2026-10-19T00:00:00Z', 'permitted_scope': ['bounded_pilot_preparation'],
                         'source_csv_sha256': annotation_hash, 'image_manifest_sha256': manifest_hash}
    pipeline = {'version': 'synthetic_fixture', 'approved_crop_config_sha256': configuration_hash(crop),
                'approved_intensity_config_sha256': configuration_hash(intensity),
                'output_image_shape': [224, 224, 3], 'dtype': 'float32', 'pixel_range': [0, 255]}
    plan = build_preparation_plan(audited['images'], known_layouts=crop['layouts'])
    plan_dir = workspace / 'manifests/gate_c_v1'
    write_json(plan_dir / 'preparation_plan.json', plan)
    write_csv(plan_dir / 'preparation_plan.csv', plan)
    candidates = {row['image_id'] for row in plan if row['proposed_action'] == 'candidate_for_preprocessing'}
    policy = {'requires_gate': 'C', 'status': 'awaiting_user_approval',
              'approved_pipeline_sha256': configuration_hash(pipeline), 'source_csv_sha256': annotation_hash,
              'image_manifest_sha256': manifest_hash, 'held_visibility_image_ids': [],
              'maximum_candidate_frames': len(candidates),
              'candidate_patients': len({row['patient_id'] for row in audited['images'] if row['image_id'] in candidates}),
              'proposed_actions': dict(Counter(row['proposed_action'] for row in plan)),
              'proposed_plan_file_sha256': sha256(plan_dir / 'preparation_plan.csv')}
    for name, value in [('preprocessing.approved.json', crop), ('intensity.approved.json', intensity),
                        ('crop_approval.json', crop_approval), ('intensity_approval.json', intensity_approval),
                        ('pipeline.approved.json', pipeline), ('preparation.proposed.json', policy),
                        ('paths.local.json', {'source_root': 'raw', 'annotation_csv': 'raw/Dataset_Annotation.csv',
                                              'audit_directory': 'manifests/audit'})]:
        write_json(workspace / 'configs' / name, value)
    return workspace


def gate_c(context):
    return {'gate': 'C', 'decision': 'approved', 'approved_by': 'user',
            'user_instruction': 'SYNTHETIC TEST APPROVAL ONLY', 'recorded_at_utc': '2026-10-19T00:00:00Z',
            'permitted_scope': ['full_local_preparation'], 'bindings': copy.deepcopy(context['bindings'])}


def passing_reviews(context):
    queue = build_quality_queue(context)
    for row in queue['reviews']:
        row.update(modality='bmode_appearing_no_doppler_seen', privacy_status='pass', diagnostic_text_status='pass',
                   geometry_status='pass', visibility_status='pass', reviewed_by='synthetic test reviewer',
                   reviewed_at_utc='2026-10-19T00:00:00Z')
    return queue


def test_metadata_preflight_produces_pending_queue_for_every_row_without_decoding(tmp_path, monkeypatch):
    workspace = preparation_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Metadata preflight must not decode an image.')
    monkeypatch.setattr(Image, 'open', forbidden)
    context = load_preparation(workspace)
    before = copy.deepcopy(context['plan'])
    queue = build_quality_queue(context)
    assert len(queue['reviews']) == 4
    assert queue['bindings'] == context['bindings']
    assert all(row['privacy_status'] == 'pending' and not row['reviewed_by'] for row in queue['reviews'])
    assert context['plan'] == before


@pytest.mark.parametrize('kind', ['missing', 'wrong_scope', 'wrong_hash'])
def test_gate_c_failure_precedes_source_access_or_output_creation(tmp_path, monkeypatch, kind):
    context = load_preparation(preparation_fixture(tmp_path))
    approval = gate_c(context)
    if kind == 'missing':
        approval = None
    elif kind == 'wrong_scope':
        approval['permitted_scope'] = ['bounded_pilot_preparation']
    else:
        approval['bindings']['source_csv_sha256'] = 'f' * 64
    monkeypatch.setattr(Image, 'open', lambda *args, **kwargs: pytest.fail('Gate must block before decoding.'))
    output = tmp_path / 'prepared'
    with pytest.raises(PermissionError, match='Gate C'):
        run_preparation(context, passing_reviews(context), approval, output)
    assert not output.exists()


def test_preparation_writes_exact_shared_tensors_lineage_and_all_hold_rows(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    original_hashes = {row['image_id']: sha256(context['root'] / row['relative_image_path']) for row in context['rows']}
    output = tmp_path / 'prepared'
    result = run_preparation(context, passing_reviews(context), gate_c(context), output)
    assert result['prepared_frames'] == 3 and result['held_frames'] == 1
    assert result['model_input_eligible'] is False and result['training_approved'] is False
    records = json.loads((output / 'output_manifest.json').read_text())
    assert len(records) == 4
    by_id = {row['image_id']: row for row in context['rows']}
    for record in records:
        assert record['model_input_eligible'] is False
        if record['status'] != 'prepared':
            assert record['status'] == 'held_by_policy' and record['tensor_path'] is None
            continue
        source = by_id[record['image_id']]
        with Image.open(context['root'] / source['relative_image_path']) as image:
            expected = preprocess_decoded(np.asarray(image), context['crop'], context['intensity'])['tensor']
        tensor = np.load(output / record['tensor_path'], allow_pickle=False)
        assert tensor.dtype == np.float32 and tensor.shape == (224, 224, 3)
        assert np.array_equal(tensor, expected)
        assert record['tensor_file_sha256'] == sha256(output / record['tensor_path'])
        assert record['tensor_pixel_sha256'] == hashlib.sha256(tensor.tobytes()).hexdigest()
        assert record['bindings'] == context['bindings']
        assert record['quality_review']['reviewed_by'] == 'synthetic test reviewer'
    for row in context['rows']:
        assert sha256(context['root'] / row['relative_image_path']) == original_hashes[row['image_id']]
    assert json.loads((output / 'run_status.json').read_text())['status'] == 'complete'
    with pytest.raises(ValueError, match='Existing'):
        run_preparation(context, passing_reviews(context), gate_c(context), output)


@pytest.mark.parametrize('kind', ['missing', 'privacy', 'diagnostic_text', 'doppler', 'geometry', 'visibility',
                                  'stale_hash', 'misassigned_patient', 'changed_path'])
def test_unreviewed_or_failed_quality_is_held_without_decoding(tmp_path, monkeypatch, kind):
    context = load_preparation(preparation_fixture(tmp_path))
    quality = passing_reviews(context)
    target = next(row for row in quality['reviews'] if row['proposed_action'] == 'candidate_for_preprocessing')
    if kind == 'missing':
        quality['reviews'].remove(target)
    elif kind == 'stale_hash':
        target['source_sha256'] = 'f' * 64
    elif kind == 'doppler':
        target['modality'] = 'possible_doppler'
    elif kind == 'misassigned_patient':
        target['patient_id'] = 'P999'
    elif kind == 'changed_path':
        target['relative_image_path'] = 'P999/another.png'
    else:
        target[kind + '_status'] = 'needs_review'
    from V4.src import prepare_dataset
    original_load = prepare_dataset.load_checked_source
    def checked(root, row):
        assert row['image_id'] != target['image_id'], 'Uncleared image must not be decoded.'
        return original_load(root, row)
    monkeypatch.setattr(prepare_dataset, 'load_checked_source', checked)
    output = tmp_path / 'prepared'
    result = run_preparation(context, quality, gate_c(context), output)
    assert result['prepared_frames'] == 2 and result['held_frames'] == 2
    record = next(row for row in json.loads((output / 'output_manifest.json').read_text()) if row['image_id'] == target['image_id'])
    assert record['status'] == 'held_quality_review' and record['tensor_path'] is None
    assert record['reasons']


def test_changed_source_becomes_a_reported_failure_and_other_rows_continue(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    target = next(row for row in context['rows'] if row['patient_id'] == 'P001')
    (context['root'] / target['relative_image_path']).write_bytes(b'changed fixture bytes')
    output = tmp_path / 'prepared'
    result = run_preparation(context, passing_reviews(context), gate_c(context), output)
    assert result['prepared_frames'] == 2
    records = json.loads((output / 'output_manifest.json').read_text())
    failed = next(row for row in records if row['image_id'] == target['image_id'])
    assert failed['status'] == 'held_processing_failure' and 'hash' in failed['reasons'][0].lower()
    assert not list((output / 'frames').glob('.tmp_*'))


def test_tampered_plan_is_rejected_even_if_its_file_hash_is_updated_in_policy(tmp_path):
    workspace = preparation_fixture(tmp_path)
    plan_dir = workspace / 'manifests/gate_c_v1'
    plan = json.loads((plan_dir / 'preparation_plan.json').read_text())
    for row in plan:
        row['proposed_action'] = 'candidate_for_preprocessing'
    write_json(plan_dir / 'preparation_plan.json', plan)
    write_csv(plan_dir / 'preparation_plan.csv', plan)
    policy_path = workspace / 'configs/preparation.proposed.json'
    policy = json.loads(policy_path.read_text())
    policy['proposed_plan_file_sha256'] = sha256(plan_dir / 'preparation_plan.csv')
    write_json(policy_path, policy)
    with pytest.raises(ValueError, match='plan|Plan'):
        load_preparation(workspace)


def test_duplicate_quality_ids_and_source_tree_output_are_rejected(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    quality = passing_reviews(context)
    quality['reviews'].append(copy.deepcopy(quality['reviews'][0]))
    with pytest.raises(ValueError, match='quality|Quality'):
        run_preparation(context, quality, gate_c(context), tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    with pytest.raises(ValueError, match='source tree'):
        run_preparation(context, passing_reviews(context), gate_c(context), context['root'] / 'new')


def test_cross_patient_conflicts_stay_held_even_with_passing_quality(tmp_path, monkeypatch):
    context = load_preparation(preparation_fixture(tmp_path, cross_patient=True))
    from V4.src import prepare_dataset
    original_load = prepare_dataset.load_checked_source
    def checked(root, row):
        assert row['patient_id'] == 'P003', 'Conflicting patient assignments must not be decoded.'
        return original_load(root, row)
    monkeypatch.setattr(prepare_dataset, 'load_checked_source', checked)
    output = tmp_path / 'prepared'
    result = run_preparation(context, passing_reviews(context), gate_c(context), output)
    assert result['prepared_frames'] == 1 and result['held_frames'] == 3
    records = json.loads((output / 'output_manifest.json').read_text())
    assert all(row['status'] == 'held_by_policy' and row['tensor_path'] is None
               for row in records if row['patient_id'] in {'P001', 'P002'})


def test_pending_template_does_not_clear_any_source_for_preparation(tmp_path, monkeypatch):
    context = load_preparation(preparation_fixture(tmp_path))
    monkeypatch.setattr(Image, 'open', lambda *args, **kwargs: pytest.fail('Pending QC must block decoding.'))
    result = run_preparation(context, build_quality_queue(context), gate_c(context), tmp_path / 'prepared')
    assert result['prepared_frames'] == 0 and result['held_frames'] == 4
    assert result['statuses'] == {'held_by_policy': 1, 'held_quality_review': 3}


def test_changed_annotation_after_loading_is_rejected_before_output(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    with context['files']['annotations'].open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='changed since preflight'):
        run_preparation(context, passing_reviews(context), gate_c(context), tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


@pytest.mark.parametrize('kind', ['crop', 'plan'])
def test_notebook_cannot_silently_change_loaded_geometry_or_release_holds(tmp_path, kind):
    context = load_preparation(preparation_fixture(tmp_path))
    if kind == 'crop':
        context['crop']['layouts']['100x60']['template_box'] = [10, 10, 90, 60]
    else:
        for row in context['plan']:
            row['proposed_action'] = 'candidate_for_preprocessing'
    with pytest.raises(ValueError, match='in-memory input changed'):
        run_preparation(context, passing_reviews(context), gate_c(context), tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_output_io_failure_marks_run_incomplete_and_removes_temporary_frame(tmp_path, monkeypatch):
    context = load_preparation(preparation_fixture(tmp_path))
    def fail_save(*args, **kwargs):
        raise OSError('Synthetic disk-write failure')
    monkeypatch.setattr(np, 'save', fail_save)
    output = tmp_path / 'prepared'
    with pytest.raises(OSError, match='disk-write'):
        run_preparation(context, passing_reviews(context), gate_c(context), output)
    assert json.loads((output / 'run_status.json').read_text())['status'] == 'failed'
    assert not (output / 'preparation_summary.json').exists()
    assert not list((output / 'frames').iterdir())


def pilot_context(tmp_path):
    workspace = preparation_fixture(tmp_path)
    context = load_preparation(workspace)
    audit_dir = context['audit_directory']
    rows = context['rows']
    subset = audit_dir / 'review_subset_extended.json'
    write_json(subset, rows)
    for name in ['crop_approval', 'intensity_approval']:
        approval = context[name]
        approval.update(review_subset_sha256=sha256(subset), reviewed_image_ids=[row['image_id'] for row in rows])
        write_json(context['files'][name], approval)
    write_json(audit_dir / 'modality_review_for_gate_b.json', {
        row['image_id']: {'classification': 'bmode_appearing_no_doppler_seen', 'grayscale_preview_allowed': True,
                          'source_sha256': row['source_sha256']} for row in rows})
    tensors = []
    for row in rows:
        with Image.open(context['root'] / row['relative_image_path']) as image:
            tensors.append(preprocess_decoded(np.asarray(image), context['crop'], context['intensity'])['tensor'])
    pilot = context['pilot_directory']
    pilot.mkdir(parents=True)
    np.save(pilot / 'pilot_tensors.npy', np.stack(tensors), allow_pickle=False)
    write_json(pilot / 'pilot_results.json', {'pilot_tensors_file_sha256': sha256(pilot / 'pilot_tensors.npy')})
    write_json(pilot / 'pilot_manifest.json', [{'image_id': row['image_id']} for row in rows])
    return load_preparation(workspace)


def test_pilot_uses_same_writer_and_preserves_final_qc_and_gate_c_boundary(tmp_path):
    context = pilot_context(tmp_path)
    output = tmp_path / 'runner_pilot'
    result = validate_pilot(context, output)
    assert result['prepared_frames'] == 4 and result['matches_approved_pilot_exactly']
    assert result['full_dataset_processing_approved'] is False
    assert result['final_per_image_quality_approved'] is False
    records = json.loads((output / 'output_manifest.json').read_text())
    assert all(row['quality_review'] == {'status': 'pilot_only_not_final_QC'} for row in records)


@pytest.mark.parametrize('kind', ['unreviewed_id', 'changed_reference'])
def test_pilot_rejects_unreviewed_ids_and_modified_reference_before_output(tmp_path, kind):
    context = pilot_context(tmp_path)
    if kind == 'unreviewed_id':
        approval = context['intensity_approval']
        approval['reviewed_image_ids'].pop()
        write_json(context['files']['intensity_approval'], approval)
        context = load_preparation(context['workspace'])
    else:
        with (context['pilot_directory'] / 'pilot_tensors.npy').open('ab') as f:
            f.write(b'changed reference')
    with pytest.raises((ValueError, PermissionError), match='Pilot|pilot'):
        validate_pilot(context, tmp_path / 'runner_pilot')
    assert not (tmp_path / 'runner_pilot').exists()


def test_cli_preflight_is_metadata_only_and_materialize_requires_gate_c(tmp_path):
    workspace = preparation_fixture(tmp_path)
    output = tmp_path / 'preflight'
    assert main(['preflight', '--workspace', str(workspace), '--output', str(output)]) == 0
    summary = json.loads((output / 'preflight_summary.json').read_text())
    assert summary['candidate_frames'] == 3 and summary['held_by_policy'] == 1
    assert summary['images_decoded'] == 0 and summary['gate_c_approved'] is False
    with pytest.raises(SystemExit) as error:
        main(['run', '--workspace', str(workspace), '--quality-reviews', str(output / 'quality_reviews.pending.json'),
              '--output', str(tmp_path / 'full')])
    assert error.value.code == 2 and not (tmp_path / 'full').exists()


def duplicate_resolution(context):
    excluded = next(row for row in context['rows'] if row['patient_id'] == 'P001')
    keeper = next(row for row in context['rows'] if row['patient_id'] == 'P002')
    return {'schema_version': 1, 'confirmed_by': 'user',
            'user_instruction': 'SYNTHETIC TEST: keep P002 grade 3; reject P001 misassigned duplicate.',
            'recorded_at_utc': '2026-10-19T00:00:00Z',
            'source_csv_sha256': context['bindings']['source_csv_sha256'],
            'image_manifest_sha256': context['bindings']['image_manifest_sha256'],
            'pairs': [{'exclude_image_id': excluded['image_id'], 'keep_image_id': keeper['image_id'],
                       'canonical_patient_id': 'P002', 'canonical_grade': 3,
                       'source_sha256': keeper['source_sha256'],
                       'decoded_pixel_sha256': keeper['decoded_pixel_sha256']}]}


def resolved_context(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path, cross_patient=True))
    resolution = duplicate_resolution(context)
    plan = build_preparation_plan(context['rows'], known_layouts=context['crop']['layouts'],
                                  source_resolution=resolution)
    path = context['workspace'] / 'configs/source_resolution.json'
    write_json(path, resolution)
    write_json(context['files']['plan'], plan)
    write_csv(context['files']['plan_csv'], plan)
    policy = context['policy']
    policy.update(source_resolution_file='configs/source_resolution.json',
                  source_resolution_file_sha256=sha256(path), maximum_candidate_frames=2, candidate_patients=2,
                  proposed_plan_file_sha256=sha256(context['files']['plan_csv']),
                  proposed_actions=dict(Counter(row['proposed_action'] for row in plan)))
    write_json(context['files']['policy'], policy)
    return load_preparation(context['workspace'])


def test_confirmed_misassignment_excludes_copy_and_retains_canonical_patient_without_relabelling(tmp_path):
    context = resolved_context(tmp_path)
    original = {row['image_id']: row['grade'] for row in context['rows']}
    output = tmp_path / 'resolved_prepared'
    result = run_preparation(context, passing_reviews(context), gate_c(context), output)
    assert result['prepared_frames'] == 2 and result['held_frames'] == 2
    assert 'source_resolution_file_sha256' in context['bindings']
    records = json.loads((output / 'output_manifest.json').read_text())
    rejected = next(row for row in records if row['patient_id'] == 'P001')
    canonical = next(row for row in records if row['patient_id'] == 'P002')
    assert rejected['status'] == 'held_by_policy' and rejected['tensor_path'] is None
    assert rejected['proposed_action'] == 'hold_resolved_duplicate_copy'
    assert canonical['status'] == 'prepared' and canonical['grade'] == 3
    assert {row['image_id']: row['grade'] for row in context['rows']} == original
    assert json.loads(context['files']['audit'].read_text())['labels']['P001'] == 0


@pytest.mark.parametrize('kind', ['unconfirmed', 'wrong_hash', 'wrong_patient', 'wrong_grade', 'different_images'])
def test_duplicate_resolution_rejects_unconfirmed_or_inexact_patient_claims(tmp_path, kind):
    context = load_preparation(preparation_fixture(tmp_path, cross_patient=True))
    resolution = duplicate_resolution(context)
    pair = resolution['pairs'][0]
    if kind == 'unconfirmed':
        resolution['user_instruction'] = ''
    elif kind == 'wrong_hash':
        pair['source_sha256'] = 'f' * 64
    elif kind == 'wrong_patient':
        pair['canonical_patient_id'] = 'P003'
    elif kind == 'wrong_grade':
        pair['canonical_grade'] = 2
    else:
        pair['keep_image_id'] = next(row['image_id'] for row in context['rows'] if row['patient_id'] == 'P003')
    with pytest.raises(ValueError, match='resolution|Resolution'):
        build_preparation_plan(context['rows'], known_layouts=context['crop']['layouts'], source_resolution=resolution)


def test_resolution_record_is_bound_and_cannot_change_after_preflight(tmp_path):
    context = resolved_context(tmp_path)
    with context['files']['source_resolution'].open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='changed since preflight'):
        run_preparation(context, passing_reviews(context), gate_c(context), tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_source_identity_confirmation_does_not_grant_gate_c(tmp_path):
    context = resolved_context(tmp_path)
    with pytest.raises(PermissionError, match='Gate C'):
        run_preparation(context, passing_reviews(context), context['source_resolution'], tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
