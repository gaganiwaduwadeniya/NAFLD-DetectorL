"""Review package excludes outer test and source identifiers; no remote work."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from V4.src.audit_dataset import sha256, write_json
from V4.src.training_bundle import stage_training_bundle, load_prepared_cohort
from V4.src.training_plan import make_splits
from V4.src.training_data import FrameStore
from V4.src.prepare_dataset import load_preparation, run_preparation
from V4.tests.test_prepare_dataset import preparation_fixture, passing_reviews, gate_c


def prepared_rows(tmp_path):
    root = tmp_path/'prepared'; root.mkdir()
    rows = []
    for i in range(20):
        gray = np.full((224,224), i+20, np.uint8)
        tensor = np.repeat(gray[...,None], 3, axis=-1).astype(np.float32)
        Image.fromarray(gray).save(root/f'{i}.png'); np.save(root/f'{i}.npy',tensor)
        value = hashlib.sha256(tensor.tobytes()).hexdigest()
        rows.append({'image_id':f'PRIVATE_IMAGE_{i}', 'patient_id':f'P{i:03d}', 'grade':i//5,
                     'status':'prepared','image_path':f'{i}.png','tensor_path':f'{i}.npy',
                     'image_file_sha256':sha256(root/f'{i}.png'),'tensor_file_sha256':sha256(root/f'{i}.npy'),
                     'source_sha256':value,'decoded_pixel_sha256':value,'tensor_pixel_sha256':value})
    return root, rows


def test_package_reconstructs_canonical_pixels_and_excludes_outer_test_and_identifiers(tmp_path):
    weights = Path(__file__).resolve().parents[1]/'artifacts/pretrained/efficientnetb0_notop.h5'
    if not weights.is_file(): pytest.skip('Official public weights are not staged locally.')
    root, rows = prepared_rows(tmp_path); plan = make_splits(rows); output = tmp_path/'bundle'
    proposal = stage_training_bundle(rows,plan,root,weights,output)
    assert proposal['frames']==16 and proposal['patient_groups']==16
    manifest = json.loads((output/'dataset/training_manifest.json').read_text())
    assert 'PRIVATE_IMAGE' not in json.dumps(manifest) and 'patient_id' not in json.dumps(manifest)
    mapping = json.loads((output/'local_mapping.DO_NOT_UPLOAD.json').read_text())
    assert {r['patient_id'] for r in mapping}.isdisjoint(plan['folds'][0]['test'])
    assert all('local_mapping' not in r['path'] for r in proposal['transfer_inventory'])
    assert not proposal['training_approved'] and not proposal['remote_transfer_approved']
    assert json.loads((output/'dataset/gate_e_approval.json').read_text())['decision']=='pending_user_approval'
    store = FrameStore(output/'dataset', proposal['manifest_sha256'])
    original = {r['image_id']:r for r in rows}
    for row in store.rows:
        mapped = next(r for r in mapping if r['frame_id']==row['frame_id'])
        np.testing.assert_array_equal(store.load(row), np.load(root/original[mapped['image_id']]['tensor_path']))
    meta = json.loads((output/'kernel/kernel-metadata.json').read_text())
    data = json.loads((output/'dataset/dataset-metadata.json').read_text())
    assert meta['title'] != data['title'] and meta['id'] != data['id']
    assert meta['enable_gpu'] and meta['is_private'] and not meta['enable_internet']
    assert meta['dataset_sources']==[data['id']]


def test_held_rows_cannot_enter_pilot_and_existing_output_is_preserved(tmp_path):
    root, rows = prepared_rows(tmp_path); plan = make_splits(rows); output = tmp_path/'bundle'
    rows[0]['status']='held_quality_review'
    with pytest.raises(ValueError,match='prepared'):
        stage_training_bundle(rows,plan,root,tmp_path/'unused-weights',output)
    assert not output.exists()


def test_prepared_loader_keeps_existing_holds_and_rejects_changed_manifest(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    write_json(context['workspace']/'configs/preparation_approval.json',gate_c(context))
    prepared = tmp_path/'prepared'
    run_preparation(context,passing_reviews(context),gate_c(context),prepared)
    assert len(load_prepared_cohort(context,prepared))==3
    path=prepared/'output_manifest.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='unchanged'):
        load_prepared_cohort(context,prepared)
