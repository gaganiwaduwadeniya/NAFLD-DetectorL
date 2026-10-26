"""Gate B acceptance tests; synthetic software fixtures, not training data."""
import json

import numpy as np
import pytest

from V4.src.build_review import configuration_hash
from V4.src.preprocess import grayscale_preserve, grayscale_preview_tensor
from V4.src.build_intensity_review import require_crop_approval, intensity_statistics, validate_intensity_config


def test_fixed_grayscale_formula_preserves_neutral_pixels_and_source():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
    saved = rgb.copy()
    assert grayscale_preserve(rgb).tolist() == [[76, 150, 29, 128, 255]]
    assert np.array_equal(rgb, saved)
    neutral = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(grayscale_preserve(neutral), neutral)
    assert np.array_equal(grayscale_preserve(np.repeat(neutral[..., None], 3, axis=2)), neutral)
    with pytest.raises(ValueError, match='uint8'):
        grayscale_preserve(neutral.astype(np.uint16))


def test_grayscale_preview_tensor_no_double_rescale_and_geometry():
    a = np.full((100, 200), 180, dtype=np.uint8)
    tensor, resize = grayscale_preview_tensor(a)
    assert tensor.shape == (224, 224, 3) and tensor.dtype == np.float32
    assert np.isfinite(tensor).all() and tensor.min() == 0 and tensor.max() == 180
    assert np.array_equal(tensor[..., 0], tensor[..., 1])
    assert np.array_equal(tensor[..., 1], tensor[..., 2])
    assert resize['padding_ltrb'] == [0, 56, 0, 56]
    assert np.all(tensor[56:168] == 180)


def test_geometry_approval_bound_to_user_scope_candidate_and_hash():
    cfg = {'version': 'fixture'}
    approval = {'gate': 'A', 'decision': 'approved', 'candidate_id': 'C2_layout_specific_crop',
                'config_sha256': configuration_hash(cfg), 'approved_by': 'user', 'user_instruction': 'approved',
                'recorded_at_utc': '2026-10-18T17:00:00+00:00',
                'permitted_scope': ['crop_geometry', 'prepare_gate_b_intensity_previews']}
    require_crop_approval(cfg, approval)
    for key, value in [('config_sha256', 'changed'), ('decision', 'pending'), ('candidate_id', 'C0_full_field'),
                       ('user_instruction', ''), ('permitted_scope', ['training'])]:
        with pytest.raises(ValueError, match='approval'):
            require_crop_approval(cfg, dict(approval, **{key: value}))
    with pytest.raises(ValueError, match='approval'):
        require_crop_approval(dict(cfg, new_crop=True), approval)


def test_gate_b_config_does_not_silently_enable_optional_enhancement():
    cfg = {'version': 'fixture', 'candidates': ['I0_preserve'], 'optional_intensity_operation': 'identity',
           'grayscale_method': 'BT601_integer_299_587_114_half_up', 'output_size': 224}
    validate_intensity_config(cfg)
    with pytest.raises(ValueError, match='I0_preserve'):
        validate_intensity_config(dict(cfg, candidates=['I0_preserve', 'I3_CLAHE']))
    with pytest.raises(ValueError, match='identity'):
        validate_intensity_config(dict(cfg, optional_intensity_operation='gamma'))


def test_statistics_exclude_canvas_and_do_not_remap_dark_pixels():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[40:60, 40:60] = 25
    gray = grayscale_preserve(rgb)
    stats, hist, mask = intensity_statistics(rgb, gray)
    assert stats['statistic_pixels'] == 400
    assert stats['grayscale_quantiles']['p50'] == 25
    assert stats['grayscale_mean'] == 25
    assert stats['rounding_max_abs_error'] == 0
    assert stats['optional_transform'] == 'identity'
    assert sum(hist['grayscale']) == 400
    assert np.array_equal(gray[40:60, 40:60], np.full((20, 20), 25, dtype=np.uint8))
    assert mask.sum() == 400


def test_empty_statistics_region_is_flagged_not_autocontrasted():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    stats, histogram, mask = intensity_statistics(rgb, grayscale_preserve(rgb))
    assert stats['status'] == 'needs_review'
    assert stats['statistic_pixels'] == 0 and not mask.any()
    assert stats['grayscale_quantiles'] is None


def gate_b_fixture(tmp_path):
    from PIL import Image
    from V4.src.audit_dataset import audit, sha256
    from V4.src.build_review import build_review
    root = tmp_path / 'raw'
    (root / 'P001').mkdir(parents=True)
    (root / 'Dataset_Annotation.csv').write_text('Patient ID,NAFLD,NON-NAFLD ,STAGE 0,STAGE 1,STAGE 2,STAGE 3\nP001,0,1,1,0,0,0\n')
    Image.fromarray(np.full((60, 100, 3), 90, dtype=np.uint8)).save(root / 'P001' / 'frame.png')
    row = audit(root, root / 'Dataset_Annotation.csv')['images'][0]
    crop = {'version': 'fixture', 'output_size': 224, 'layouts': {'100x60': {
        'selector': {'width': 100, 'height': 60}, 'privacy_reviewed': True,
        'mask_rectangles': [[0, 0, 100, 10]], 'template_box': [0, 10, 100, 60]}}}
    gallery = tmp_path / 'gate_a'
    build_review(root, [row], crop, gallery)
    approval = {'gate': 'A', 'decision': 'approved', 'candidate_id': 'C2_layout_specific_crop',
                'config_sha256': configuration_hash(crop), 'approved_by': 'user', 'user_instruction': 'approved',
                'recorded_at_utc': '2026-10-18T17:00:00+00:00', 'permitted_scope': ['crop_geometry', 'prepare_gate_b_intensity_previews'],
                'reviewed_image_ids': [row['image_id']], 'gate_a_candidates_sha256': sha256(gallery / 'crop_candidates.jsonl')}
    policy = {'version': 'fixture', 'candidates': ['I0_preserve'], 'optional_intensity_operation': 'identity',
              'grayscale_method': 'BT601_integer_299_587_114_half_up', 'output_size': 224}
    modality = {row['image_id']: {'source_sha256': row['source_sha256'], 'classification': 'bmode_appearing_no_doppler_seen', 'grayscale_preview_allowed': True}}
    return root, row, crop, approval, policy, gallery, modality


def test_actual_gate_b_gallery_approval_lineage_and_preservation(tmp_path):
    from PIL import Image
    from V4.src.build_intensity_review import build_intensity_review
    from V4.src.audit_dataset import sha256
    root, row, crop, approval, policy, gallery, modality = gate_b_fixture(tmp_path)
    source_hash = sha256(root / row['relative_image_path'])
    output = tmp_path / 'gate_b'
    records = build_intensity_review(root, [row], crop, approval, policy, gallery, modality, output)
    assert len(records) == 1 and records[0]['status'] == 'pending'
    assert records[0]['model_input_eligible'] is False
    assert records[0]['box_ltrb'] == [0, 10, 100, 60]
    assert sha256(root / row['relative_image_path']) == source_hash
    assert (output / 'index.html').is_file() and 'Normal' not in (output / 'index.html').read_text()
    padded = np.asarray(Image.open(output / row['image_id'] / 'I0_preserve_padded224.png'))
    assert padded.shape == (224, 224, 3) and padded.max() == 90
    assert np.array_equal(padded[..., 0], padded[..., 2])
    with pytest.raises(ValueError, match='Existing review'):
        build_intensity_review(root, [row], crop, approval, policy, gallery, modality, output)


def test_unreviewed_modality_and_changed_gate_a_pixels_block_gray_gallery(tmp_path):
    from PIL import Image
    from V4.src.build_intensity_review import build_intensity_review
    root, row, crop, approval, policy, gallery, modality = gate_b_fixture(tmp_path)
    with pytest.raises(ValueError, match='modality review'):
        build_intensity_review(root, [row], crop, approval, policy, gallery, {}, tmp_path / 'unknown')
    assert not (tmp_path / 'unknown').exists()
    Image.new('RGB', (100, 50), 'white').save(gallery / row['image_id'] / 'C2_layout_specific_crop_crop.png')
    with pytest.raises(ValueError, match='Crop pixels changed'):
        build_intensity_review(root, [row], crop, approval, policy, gallery, modality, tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()
