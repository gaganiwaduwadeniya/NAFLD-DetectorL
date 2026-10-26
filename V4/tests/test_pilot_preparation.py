"""Gate C pilot tests; fixtures are software checks, never added patient data."""
import json

import numpy as np
import pytest
from PIL import Image

from V4.src.audit_dataset import audit, sha256
from V4.src.build_review import configuration_hash
from V4.src.preprocess import preprocess_decoded
from V4.src.pilot_preparation import require_intensity_approval, preprocess_review_batch, build_preparation_plan


def pilot_fixture(tmp_path):
    root = tmp_path / 'raw'
    (root / 'P001').mkdir(parents=True)
    (root / 'Dataset_Annotation.csv').write_text('Patient ID,NAFLD,NON-NAFLD ,STAGE 0,STAGE 1,STAGE 2,STAGE 3\nP001,0,1,1,0,0,0\n')
    Image.fromarray(np.full((60, 100, 3), 90, dtype=np.uint8)).save(root / 'P001/frame.png')
    row = audit(root, root / 'Dataset_Annotation.csv')['images'][0]
    crop = {'version': 'fixture', 'output_size': 224, 'layouts': {'100x60': {
        'selector': {'width': 100, 'height': 60}, 'privacy_reviewed': True,
        'mask_rectangles': [[0, 0, 100, 10]], 'template_box': [0, 10, 100, 60]}}}
    intensity = {'version': 'fixture', 'candidates': ['I0_preserve'], 'optional_intensity_operation': 'identity',
                 'grayscale_method': 'BT601_integer_299_587_114_half_up', 'output_size': 224}
    a = {'gate': 'A', 'decision': 'approved', 'candidate_id': 'C2_layout_specific_crop',
         'config_sha256': configuration_hash(crop), 'approved_by': 'user', 'user_instruction': 'approved',
         'recorded_at_utc': '2026-10-18T17:00:00+00:00', 'permitted_scope': ['crop_geometry', 'prepare_gate_b_intensity_previews']}
    b = {'gate': 'B', 'decision': 'approved', 'candidate_id': 'I0_preserve',
         'config_sha256': configuration_hash(intensity), 'approved_geometry_sha256': configuration_hash(crop),
         'approved_by': 'user', 'user_instruction': 'yup', 'recorded_at_utc': '2026-10-18T17:00:00+00:00',
         'permitted_scope': ['bounded_pilot_preparation'], 'reviewed_image_ids': [row['image_id']]}
    modality = {row['image_id']: {'source_sha256': row['source_sha256'], 'classification': 'bmode_appearing_no_doppler_seen', 'grayscale_preview_allowed': True}}
    return root, row, crop, intensity, a, b, modality


def test_unified_pipeline_geometry_grayscale_dtype_and_no_input_mutation(tmp_path):
    _, _, crop, intensity, _, _, _ = pilot_fixture(tmp_path)
    source = np.full((60, 100, 3), 90, dtype=np.uint8)
    saved = source.copy()
    result = preprocess_decoded(source, crop, intensity)
    assert result['cropped'].shape == (50, 100, 3)
    assert result['gray'].shape == (50, 100) and np.all(result['gray'] == 90)
    assert result['tensor'].shape == (224, 224, 3) and result['tensor'].dtype == np.float32
    assert result['tensor'].max() == 90 and result['tensor'].min() == 0
    assert np.array_equal(result['tensor'][..., 0], result['tensor'][..., 2])
    assert np.array_equal(source, saved)


def test_unknown_layout_blocks_instead_of_silently_cropping(tmp_path):
    _, _, crop, intensity, _, _, _ = pilot_fixture(tmp_path)
    with pytest.raises(ValueError, match='Unknown|unapproved'):
        preprocess_decoded(np.zeros((768, 1024, 3), dtype=np.uint8), crop, intensity)


def test_intensity_approval_bound_to_both_configs(tmp_path):
    _, _, crop, intensity, _, b, _ = pilot_fixture(tmp_path)
    require_intensity_approval(crop, intensity, b)
    for changes in [{'decision': 'pending'}, {'config_sha256': 'changed'}, {'approved_geometry_sha256': 'changed'},
                    {'user_instruction': ''}, {'permitted_scope': ['training']}]:
        with pytest.raises(ValueError, match='approval'):
            require_intensity_approval(crop, intensity, dict(b, **changes))


def test_pilot_batch_is_reproducible_and_preserves_source(tmp_path):
    root, row, crop, intensity, a, b, modality = pilot_fixture(tmp_path)
    before = sha256(root / row['relative_image_path'])
    first, metadata = preprocess_review_batch(root, [row], crop, intensity, a, b, modality)
    second, again = preprocess_review_batch(root, [row], crop, intensity, a, b, modality)
    assert first.shape == (1, 224, 224, 3) and first.dtype == np.float32
    assert np.array_equal(first, second) and metadata == again
    assert metadata[0]['model_input_eligible'] is False
    assert sha256(root / row['relative_image_path']) == before


def test_full_materialization_and_unreviewed_cohort_are_blocked(tmp_path):
    root, row, crop, intensity, a, b, modality = pilot_fixture(tmp_path)
    with pytest.raises(PermissionError, match='Gate C'):
        preprocess_review_batch(root, [row] * 1669, crop, intensity, a, b, modality)
    different = dict(row, image_id='IMG_NOT_PREVIOUSLY_REVIEWED')
    with pytest.raises(ValueError, match='reviewed'):
        preprocess_review_batch(root, [different], crop, intensity, a, b, modality)


def test_processed_cache_outside_raw_source_is_not_reprocessed(tmp_path):
    root, row, crop, intensity, a, b, modality = pilot_fixture(tmp_path)
    with pytest.raises(ValueError, match='escapes source'):
        preprocess_review_batch(root, [dict(row, relative_image_path='../cached.png')], crop, intensity, a, b, modality)


def test_palette_indices_are_not_silently_treated_as_grayscale(tmp_path):
    from V4.src.build_review import load_checked_source
    root, row, *_ = pilot_fixture(tmp_path)
    palette = Image.new('P', (100, 60))
    palette.putpalette([value for i in range(256) for value in (i, 0, 255 - i)])
    palette.save(root / 'P001/frame.png')
    row = audit(root, root / 'Dataset_Annotation.csv')['images'][0]
    with pytest.raises(ValueError, match='mode|format'):
        load_checked_source(root, row)


def test_proposed_inclusion_holds_cross_patient_conflicts_and_keeps_one_within_patient_copy():
    def row(i, pid, grade, digest):
        return {'image_id': f'IMG_{i}', 'patient_id': pid, 'grade': grade, 'decoded_pixel_sha256': digest,
                'source_sha256': digest, 'decode_status': 'ok', 'layout': 'known', 'quality_flags': []}
    rows = [row(1, 'P001', 0, 'cross'), row(2, 'P002', 3, 'cross'), row(3, 'P003', 1, 'within'),
            row(4, 'P003', 1, 'within'), row(5, 'P004', 2, 'hold')]
    plan = build_preparation_plan(rows, held_image_ids=['IMG_5'], known_layouts=['known'])
    byid = {r['image_id']: r for r in plan}
    assert byid['IMG_1']['proposed_action'] == 'hold_cross_patient_duplicate'
    assert byid['IMG_2']['proposed_action'] == 'hold_cross_patient_duplicate'
    assert byid['IMG_3']['proposed_action'] == 'candidate_for_preprocessing'
    assert byid['IMG_4']['proposed_action'] == 'hold_redundant_exact_copy'
    assert byid['IMG_5']['proposed_action'] == 'hold_manual_review'
    assert all(r['model_input_eligible'] is False and r['policy_approved'] is False for r in plan)
    assert build_preparation_plan(list(reversed(rows)), held_image_ids=['IMG_5'], known_layouts=['known']) == plan
