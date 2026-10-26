"""Local-only smoke package checks; no Kaggle client or network is invoked."""
import json

import numpy as np
import pytest
from PIL import Image

from V4.src.audit_dataset import sha256, write_json
from V4.src.kaggle_smoke import build_smoke_bundle
from V4.src.kaggle_smoke_runtime import run_smoke
from V4.src.prepare_dataset import load_preparation, run_preparation
from V4.tests.test_prepare_dataset import gate_c, passing_reviews, preparation_fixture


def test_masked_copy_replays_when_privacy_mask_removed_layout_signature():
    from V4.src import preprocess
    from V4.src.kaggle_smoke_runtime import replay_masked_source
    pixels = np.full((768, 1024, 3), [50, 60, 70], dtype=np.uint8)
    pixels[740:, 900:] = [255, 160, 0]
    layout = 'amber_footer_1024x768'
    crop = {'output_size': 224, 'layouts': {layout: {
        'selector': {'width': 1024, 'height': 768}, 'privacy_reviewed': True,
        'mask_rectangles': [[0, 0, 1024, 38], [0, 724, 1024, 768]],
        'template_box': [0, 38, 1024, 768]}}}
    intensity = {'candidates': ['I0_preserve'], 'optional_intensity_operation': 'identity',
                 'grayscale_method': 'BT601_integer_299_587_114_half_up', 'output_size': 224}
    original = preprocess.preprocess_decoded(pixels, crop, intensity)
    assert preprocess.classify_layout(original['deidentified']) != layout
    with pytest.raises(ValueError, match='Unknown'):
        preprocess.preprocess_decoded(original['deidentified'], crop, intensity)
    actual = replay_masked_source(original['deidentified'], crop, intensity, layout, vars(preprocess))
    np.testing.assert_array_equal(actual, original['tensor'])


def prepared_fixture(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    write_json(context['workspace'] / 'configs/preparation_approval.json', gate_c(context))
    prepared = tmp_path / 'prepared'
    run_preparation(context, passing_reviews(context), gate_c(context), prepared)
    rows = json.loads((prepared / 'output_manifest.json').read_text())
    selected = [r['image_id'] for r in rows if r['status'] == 'prepared']
    return context, prepared, selected, rows


def test_bundle_contains_only_selected_passing_frames_and_private_offline_cpu_job(tmp_path):
    context, prepared, selected, _ = prepared_fixture(tmp_path)
    output = tmp_path / 'bundle'
    result = build_smoke_bundle(context, prepared, selected, output, owner='syntheticowner')
    assert result['sample_count'] == 3 and result['remote_transfer_approved'] is False
    manifest = json.loads((output / 'dataset/smoke_manifest.json').read_text())
    assert 'patient_id' not in json.dumps(manifest) and 'relative_image_path' not in json.dumps(manifest)
    dataset_metadata = json.loads((output / 'dataset/dataset-metadata.json').read_text())
    assert dataset_metadata['licenses'] == [{'name': 'other'}]
    assert 'Restricted private research' in dataset_metadata['description']
    assert 'No public redistribution license is granted.' in dataset_metadata['description']
    metadata = json.loads((output / 'kernel/kernel-metadata.json').read_text())
    assert metadata['title'] != dataset_metadata['title']
    assert metadata['id'] != dataset_metadata['id']
    assert metadata['id'] == result['proposed_notebook']
    assert metadata['dataset_sources'] == [dataset_metadata['id']]
    assert metadata['is_private'] is True
    assert metadata['enable_gpu'] is False and metadata['enable_internet'] is False
    assert metadata['competition_sources'] == metadata['kernel_sources'] == metadata['model_sources'] == []
    for sample in manifest['samples']:
        with Image.open(output / 'dataset' / sample['source_file']) as image:
            pixels = np.asarray(image)
            assert image.mode == 'RGB' and not image.getexif()
            assert np.all(pixels[:10] == 0)  # Approved synthetic privacy mask.
    before = {p.name for p in (output / 'dataset').iterdir()}
    report = run_smoke(output / 'dataset', result['smoke_manifest_sha256'], tmp_path / 'smoke_result')
    assert report['status'] == 'passed' and report['matched_samples'] == 3
    assert report['maximum_absolute_pixel_difference'] == 0
    assert {p.name for p in (output / 'dataset').iterdir()} == before
    assert report['training_executed'] is False


def test_bundle_rejects_policy_holds_before_creating_output(tmp_path):
    context, prepared, selected, rows = prepared_fixture(tmp_path)
    held = next(r['image_id'] for r in rows if r['status'] != 'prepared')
    with pytest.raises(ValueError, match='prepared|eligible'):
        build_smoke_bundle(context, prepared, [held], tmp_path / 'bad', owner='syntheticowner')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('tamper', ['pixels', 'preprocess_code', 'unlisted_file', 'wrong_manifest'])
def test_smoke_rejects_tampering_before_output_or_code_execution(tmp_path, tamper):
    context, prepared, selected, _ = prepared_fixture(tmp_path)
    output = tmp_path / 'bundle'
    result = build_smoke_bundle(context, prepared, selected, output, owner='syntheticowner')
    dataset = output / 'dataset'
    expected_hash = result['smoke_manifest_sha256']
    if tamper == 'pixels':
        (dataset / 'source_S01.png').write_bytes(b'changed source')
    elif tamper == 'preprocess_code':
        (dataset / 'preprocess.py').write_text('raise RuntimeError("MUST NOT EXECUTE")')
    elif tamper == 'unlisted_file':
        (dataset / 'private_annotations.csv').write_text('must never be included')
    else:
        expected_hash = '0' * 64
    with pytest.raises(ValueError, match='hash|whitelist|manifest'):
        run_smoke(dataset, expected_hash, tmp_path / 'bad_result')
    assert not (tmp_path / 'bad_result').exists()


def test_pixel_difference_is_reported_as_failed_without_relaxing_tolerance(tmp_path):
    context, prepared, selected, _ = prepared_fixture(tmp_path)
    output = tmp_path / 'bundle'
    build_smoke_bundle(context, prepared, selected, output, owner='syntheticowner')
    dataset = output / 'dataset'
    expected = np.load(dataset / 'expected_S01.npy', allow_pickle=False)
    expected[0, 0] += 1
    np.save(dataset / 'expected_S01.npy', expected, allow_pickle=False)
    path = dataset / 'smoke_manifest.json'
    manifest = json.loads(path.read_text())
    for item in manifest['assets']:
        if item['path'] == 'expected_S01.npy':
            item['sha256'] = sha256(dataset / item['path'])
    write_json(path, manifest)
    report = run_smoke(dataset, sha256(path), tmp_path / 'mismatch')
    assert report['status'] == 'failed'
    assert report['maximum_absolute_pixel_difference'] == 1
    assert report['permitted_absolute_pixel_difference'] == 0
