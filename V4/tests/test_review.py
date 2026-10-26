"""Synthetic software fixtures only; these are never patient training images."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from V4.src.audit_dataset import load_labels, map_patient, audit, select_subset
from V4.src.preprocess import classify_layout, crop_pixels, letterbox, mask_pixels, propose_crop
from V4.src.build_review import build_review


def annotation(path, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Patient ID', 'NAFLD', 'NON-NAFLD ', 'STAGE 0', 'STAGE 1', 'STAGE 2', 'STAGE 3'])
        w.writerows(rows)


def test_labels_strip_headers_and_reject_ambiguous_stages(tmp_path):
    p = tmp_path / 'Dataset_Annotation.csv'
    annotation(p, [['P001', 0, 1, 1, 0, 0, 0], ['P002', 1, 0, 0, 0, 1, 0]])
    assert load_labels(p) == {'P001': 0, 'P002': 2}
    for flags in [[0, 0, 0, 0], [0, 1, 1, 0], [0, 2, 0, 0]]:
        annotation(p, [['P001', 1, 0, *flags]])
        with pytest.raises(ValueError):
            load_labels(p)


def test_labels_reject_duplicate_ids_and_inconsistent_binary(tmp_path):
    p = tmp_path / 'labels.csv'
    annotation(p, [['P001', 0, 1, 1, 0, 0, 0]] * 2)
    with pytest.raises(ValueError):
        load_labels(p)
    annotation(p, [['P001', 0, 1, 0, 1, 0, 0]])
    with pytest.raises(ValueError):
        load_labels(p)


def test_mapping_uses_id_tokens_and_fails_on_conflict():
    assert map_patient(Path('Normal/P002/frame_42.jpg')) == 'P002'
    assert map_patient(Path('images/P003_frame.jpg')) == 'P003'
    assert map_patient(Path('images/frame_42.jpg')) is None
    with pytest.raises(ValueError):
        map_patient(Path('P001/P002_frame.jpg'))
    assert map_patient(Path('P0012/frame.jpg')) is None
    assert map_patient(Path('P001abc/frame.jpg')) is None


def test_same_dimensions_different_display_layouts_are_not_same_template():
    banner = np.zeros((768, 1024, 3), dtype=np.uint8)
    banner[65:75, 130:145] = [230, 220, 40]
    footer = np.zeros_like(banner)
    footer[740:750, 900:950] = [240, 175, 30]
    assert classify_layout(banner) == 'header_banner_1024x768'
    assert classify_layout(footer) == 'amber_footer_1024x768'
    assert classify_layout(np.zeros_like(banner)) == 'unknown_1024x768'


def test_coordinate_convention_and_no_source_mutation():
    a = np.arange(60, dtype=np.uint8).reshape(6, 10)
    original = a.copy()
    assert np.array_equal(crop_pixels(a, [2, 1, 8, 5]), a[1:5, 2:8])
    for box in [[-1, 0, 5, 5], [0, 0, 11, 6], [3, 1, 3, 5], [0, 0, 2.5, 4]]:
        with pytest.raises(ValueError):
            crop_pixels(a, box)
    masked = mask_pixels(a, [[1, 1, 4, 3]])
    assert not masked[1:3, 1:4].any()
    assert np.array_equal(a, original)


def test_letterbox_preserves_geometry_intensity_and_channels():
    a = np.full((100, 200, 3), 180, dtype=np.uint8)
    padded, meta = letterbox(a)
    assert padded.shape == (224, 224, 3)
    assert meta['resized_size'] == [224, 112]
    assert meta['padding_ltrb'] == [0, 56, 0, 56]
    assert np.all(padded[56:168] == 180)
    assert not padded[:56].any()
    assert np.array_equal(a, np.full((100, 200, 3), 180, dtype=np.uint8))


def test_high_depth_is_blocked_instead_of_truncated():
    with pytest.raises(ValueError, match='uint8'):
        letterbox(np.full((4, 4), 4000, dtype=np.uint16))


def test_conservative_crop_contains_all_nonblack_pixels_and_flags_empty():
    a = np.zeros((100, 200, 3), dtype=np.uint8)
    a[20:80, 30:170] = 120
    box, warnings = propose_crop(a, black_threshold=0, margin_px=5)
    assert box == [25, 15, 175, 85]
    assert not warnings
    box, warnings = propose_crop(np.zeros_like(a))
    assert box == [0, 0, 200, 100]
    assert 'empty_detection_full_field_fallback' in warnings


def test_subset_reproducible_balanced_and_excludes_reserved():
    labels = {f'P{i:03d}': (i - 1) % 4 for i in range(1, 29)}
    rows = [dict(patient_id=p, image_id=p + '_a', decode_status='ok', layout='A', quality_flags=[]) for p in labels]
    selected = select_subset(rows, labels, seed=42, reserved=['P001'])
    assert len(selected) == 20
    assert selected == select_subset(list(reversed(rows)), labels, seed=42, reserved=['P001'])
    assert 'P001' not in {r['patient_id'] for r in selected}
    assert all(sum(labels[r['patient_id']] == g for r in selected) == 5 for g in range(4))


def make_dataset(tmp_path):
    root = tmp_path / 'raw'
    root.mkdir()
    annotation(root / 'Dataset_Annotation.csv', [['P001', 0, 1, 1, 0, 0, 0], ['P002', 1, 0, 0, 1, 0, 0]])
    for pid in ['P001', 'P002']:
        (root / pid).mkdir()
        Image.fromarray(np.full((60, 100, 3), 90, dtype=np.uint8)).save(root / pid / 'a.png')
    (root / 'P001' / 'broken.jpg').write_bytes(b'broken')
    Image.new('RGB', (100, 60)).save(root / 'unmapped.png')
    return root


def test_audit_reports_corrupt_unmapped_duplicate_and_preserves_source(tmp_path):
    root = make_dataset(tmp_path)
    result = audit(root, root / 'Dataset_Annotation.csv')
    assert len(result['images']) == 4
    assert result['summary']['decode_failures'] == 1
    assert result['summary']['unmapped_images'] == 1
    assert result['summary']['cross_patient_exact_duplicate_groups'] == 1
    assert result['summary']['patient_label_counts'] == {'0': 1, '1': 1, '2': 0, '3': 0}
    assert result['summary']['mapping_status'] == 'blocked'


def test_gallery_blocks_unreviewed_privacy_and_retains_pending_gate(tmp_path):
    root = make_dataset(tmp_path)
    result = audit(root, root / 'Dataset_Annotation.csv')
    row = next(r for r in result['images'] if r['patient_id'] == 'P001' and r['decode_status'] == 'ok')
    config = {'version': 'test', 'output_size': 224, 'black_threshold': 0, 'margin_px': 2, 'layouts': {}}
    with pytest.raises(ValueError, match='privacy'):
        build_review(root, [row], config, tmp_path / 'gallery')
    config['layouts']['100x60'] = {
        'mask_rectangles': [[0, 0, 100, 10]],
        'privacy_reviewed': True,
        'template_box': [0, 10, 100, 60],
        'selector': {'width': 100, 'height': 60},
    }
    out = tmp_path / 'gallery_ok'
    records = build_review(root, [row], config, out)
    assert (out / 'index.html').is_file()
    assert len(records) == 3
    assert {r['candidate_id'] for r in records} == {'C0_full_field', 'C1_conservative_field_crop', 'C2_layout_specific_crop'}
    assert all(r['status'] == 'pending' and r['model_input_eligible'] is False for r in records)
    assert (out / row['image_id'] / 'deidentified_original.png').is_file()
    assert 'Normal' not in (out / 'index.html').read_text()
    for r in records:
        crop = np.asarray(Image.open(out / r['crop_path']))
        assert crop.max() == 90
    again = audit(root, root / 'Dataset_Annotation.csv')
    assert [r['source_sha256'] for r in result['images']] == [r['source_sha256'] for r in again['images']]


def test_output_cannot_overwrite_source_tree(tmp_path):
    root = make_dataset(tmp_path)
    with pytest.raises(ValueError, match='source'):
        build_review(root, [], {}, root / 'previews')


def test_modified_source_and_bulk_render_are_blocked(tmp_path):
    root = make_dataset(tmp_path)
    result = audit(root, root / 'Dataset_Annotation.csv')
    row = next(r for r in result['images'] if r['decode_status'] == 'ok' and r['patient_id'] == 'P001')
    with pytest.raises(ValueError, match='at most'):
        build_review(root, [row] * 41, {}, tmp_path / 'bulk')
    (root / row['relative_image_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash changed'):
        build_review(root, [row], {}, tmp_path / 'tampered')
    assert not (tmp_path / 'tampered').exists()


def test_refresh_rejects_changed_inventory(tmp_path):
    from V4.src.audit_dataset import refresh_layouts
    root = make_dataset(tmp_path)
    result = audit(root, root / 'Dataset_Annotation.csv')
    Image.new('RGB', (10, 10)).save(root / 'new.png')
    with pytest.raises(ValueError, match='inventory changed'):
        refresh_layouts(root, root / 'Dataset_Annotation.csv', result)
