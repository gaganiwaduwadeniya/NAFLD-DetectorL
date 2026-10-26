"""New full-experiment contracts only; no fitting or existing suite reruns."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from V4.src.training_data import digest, patient_predictions
from V4.src.full_training import (
    FullFrameStore, full_config, require_full_approval, validate_oof,
    bootstrap_intervals, majority_predictions, predict_outer_fold, export_archive,
)
from V4.tests.test_training_bundle import prepared_rows
from V4.src.training_plan import make_splits
from V4.src.training_runtime import config_hash


def full_fixture(tmp_path):
    root, rows = prepared_rows(tmp_path)
    plan = make_splits(rows)
    # Reuse the plain synthetic PNGs; exclude their tensor companions from mount.
    mount = tmp_path / 'mount'; mount.mkdir()
    groups = {r['patient_id']: f'G{i:04d}' for i, r in enumerate(rows, 1)}
    frames = []
    for i, row in enumerate(rows, 1):
        name = f'F{i:06d}.png'
        (mount/name).write_bytes((root/row['image_path']).read_bytes())
        frames.append(dict(frame_id=name[:-4], file=name, group_id=groups[row['patient_id']],
                           grade=row['grade'], sha256=digest(mount/name)))
    manifest = dict(schema_version=2, purpose='five_fold_internal_cv', frames=frames,
                    folds=[dict(fold=f['fold'], **{role: [groups[p] for p in f[role]]
                           for role in ('train', 'validation', 'test')}) for f in plan['folds']], assets=[])
    (mount/'training_manifest.json').write_text(json.dumps(manifest))
    return mount, manifest


def store_for(mount, manifest):
    (mount/'training_manifest.json').write_text(json.dumps(manifest))
    return FullFrameStore(mount, digest(mount/'training_manifest.json'))


def perfect_oof(store):
    patients, frames = [], []
    for fold in range(5):
        rows = store.partition(fold, 'test')
        probs = np.eye(4)[[r['grade'] for r in rows]]
        patients.extend(dict(p, fold=fold) for p in patient_predictions(rows, probs))
        frames.extend(dict(frame_id=r['frame_id'], group_id=r['group_id'], grade=r['grade'],
                           prediction=r['grade'], probabilities=p.tolist(), fold=fold)
                      for r, p in zip(rows, probs))
    return patients, frames


def test_full_store_roles_are_disjoint_cover_cohort_and_reuse_exact_loader(tmp_path):
    mount, manifest = full_fixture(tmp_path); store = store_for(mount, manifest)
    outer = []
    for f in range(5):
        groups = [{r['group_id'] for r in store.partition(f, role)}
                  for role in ('train', 'validation', 'test')]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert len(set.union(*groups)) == 20
        outer.extend(groups[2])
    assert len(outer) == len(set(outer)) == 20
    x, y, w = next(store.batches(store.partition(0, 'train'), 4))
    assert x.shape == (4,224,224,3) and x.dtype == np.float32
    assert np.array_equal(x[...,0], x[...,2]) and x.min() >= 20 and x.max() <= 39
    broken = copy.deepcopy(manifest)
    broken['folds'][0]['train'].append(broken['folds'][0]['test'][0])
    with pytest.raises(ValueError, match='partition|overlap'):
        store_for(mount, broken)


def test_actual_full_approval_required_before_output_or_framework_import(tmp_path):
    from V4.src.full_training import run_experiment
    config = full_config('0'*64)
    with pytest.raises(PermissionError, match='approval'):
        run_experiment(tmp_path/'not-read', 'manifest', config,
                       {'decision':'pending_user_approval'}, tmp_path/'output')
    assert not (tmp_path/'output').exists()
    approval = dict(gate='E', decision='approved', approved_by='user', user_instruction='SYNTHETIC ONLY',
                    recorded_at_utc='fixture', permitted_scope=['one_private_gpu_pilot'],
                    manifest_sha256='manifest', maximum_remote_runs=1, timeout_seconds=7200)
    with pytest.raises(PermissionError):
        require_full_approval(approval, config, 'manifest')


def test_oof_requires_once_per_patient_correct_fold_and_frame_aggregation(tmp_path):
    mount, manifest = full_fixture(tmp_path); store = store_for(mount, manifest)
    patients, frames = perfect_oof(store)
    metrics = validate_oof(store, patients, frames)
    assert metrics['patient_metrics']['balanced_accuracy'] == 1
    assert metrics['frame_metrics']['frames'] == 20
    with pytest.raises(ValueError, match='coverage|duplicate'):
        validate_oof(store, patients+[patients[0]], frames)
    wrong = copy.deepcopy(patients); wrong[0]['fold'] = (wrong[0]['fold']+1)%5
    with pytest.raises(ValueError, match='fold'):
        validate_oof(store, wrong, frames)
    wrong = copy.deepcopy(frames); wrong[0]['probabilities'] = [.25]*4
    with pytest.raises(ValueError, match='probab|aggregate|prediction'):
        validate_oof(store, patients, wrong)


def test_bootstrap_and_majority_baseline_use_patient_units_and_train_only(tmp_path):
    mount, manifest = full_fixture(tmp_path); store = store_for(mount, manifest)
    patients, _ = perfect_oof(store)
    intervals = bootstrap_intervals(patients, draws=100, seed=42)
    assert intervals['balanced_accuracy'] == [1.,1.]
    assert intervals['macro_f1'] == [1.,1.]
    train = [dict(group_id='a',grade=1)]*10 + [dict(group_id='b',grade=2),dict(group_id='c',grade=2)]
    baseline = majority_predictions(train, patients)
    assert {p['prediction'] for p in baseline} == {2}  # two patients beat ten frames from one patient


def test_outer_inference_rejects_missing_or_changed_checkpoint_lock(tmp_path):
    mount, manifest = full_fixture(tmp_path); store = store_for(mount, manifest)
    output=tmp_path/'fold'; output.mkdir()
    def forbidden(*args, **kwargs):
        pytest.fail('Outer inference ran before checkpoint lock validation')
    with pytest.raises(ValueError, match='lock'):
        predict_outer_fold(forbidden, store, 0, output)
    (output/'selected.keras').write_bytes(b'checkpoint bytes')
    (output/'selection_lock.json').write_text(json.dumps(dict(fold=0, checkpoint='selected.keras',
        checkpoint_sha256='wrong', selected_using='inner_validation_only')))
    with pytest.raises(ValueError, match='checkpoint'):
        predict_outer_fold(forbidden, store, 0, output)


def test_export_contains_models_plots_and_logs_but_excludes_cache(tmp_path):
    output=tmp_path/'results'; output.mkdir()
    (output/'model.keras').write_bytes(b'checkpoint')
    (output/'run.log').write_text('actual log')
    (output/'figure.svg').write_text('<svg/>')
    cache=output/'.keras/models'; cache.mkdir(parents=True)
    (cache/'private-cache.bin').write_bytes(b'not an output')
    archive=export_archive(output)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
    assert 'full_cv_results/model.keras' in names and 'full_cv_results/run.log' in names
    assert not any('.keras/' in name for name in names)


def test_full_guard_accepts_exact_scope_and_rejects_a_rebound_recipe_change():
    config=full_config('0'*64)
    approval=dict(gate='E',decision='approved',approved_by='user',user_instruction='SYNTHETIC ONLY',
        recorded_at_utc='fixture',permitted_scope=['one_private_five_fold_experiment'],
        manifest_sha256='fixture',experiment_config_sha256=config_hash(config),maximum_remote_runs=1,
        timeout_seconds=7200,full_cohort_private_upload_approved=True)
    require_full_approval(approval,config,'fixture')
    config['head_epochs']=999
    approval['experiment_config_sha256']=config_hash(config)
    with pytest.raises(ValueError,match='configuration'):
        require_full_approval(approval,config,'fixture')


def test_oof_uses_probability_means_for_a_patient_with_multiple_frames(tmp_path):
    mount,manifest=full_fixture(tmp_path)
    for number in (21,22):
        r=dict(manifest['frames'][0],frame_id=f'F{number:06d}',file=f'F{number:06d}.png')
        (mount/r['file']).write_bytes((mount/manifest['frames'][0]['file']).read_bytes())
        manifest['frames'].append(r)
    store=store_for(mount,manifest)
    patients,frames=perfect_oof(store)
    selected=sorted([p for p in frames if p['group_id']=='G0001'],key=lambda p:p['frame_id'])
    values=[[.51,.49,0,0],[.51,.49,0,0],[0,1,0,0]]
    for p,probs in zip(selected,values):
        p.update(probabilities=probs,prediction=int(np.argmax(probs)))
    actual=next(p for p in patients if p['group_id']=='G0001')
    actual.update(probabilities=np.mean(values,axis=0).tolist(),prediction=1)
    validate_oof(store,patients,frames)
    actual['prediction']=0  # majority frame argmax differs from probability averaging
    with pytest.raises(ValueError,match='aggregate'):
        validate_oof(store,patients,frames)
