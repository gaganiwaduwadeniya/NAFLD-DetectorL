"""Patient separation and portable loader contracts; no fitting or network."""
import builtins
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pytest
from PIL import Image

from V4.src.training_plan import make_splits, validate_splits
from V4.src.training_data import FrameStore, patient_weights, patient_predictions, classification_metrics
from V4.src.training_runtime import require_training_approval, run_training


def cohort():
    rows = []
    for grade, count in enumerate([10, 35, 46, 18]):
        for patient in range(count):
            pid = f'P{grade}{patient:03d}'
            for frame in range(1 + patient % 3):
                unique = f'{pid}-{frame}'
                digest = hashlib.sha256(unique.encode()).hexdigest()
                rows.append({'image_id': unique, 'patient_id': pid, 'grade': grade,
                             'source_sha256': digest, 'decoded_pixel_sha256': digest,
                             'tensor_pixel_sha256': digest})
    return rows


def test_five_folds_are_reproducible_patient_stratified_and_outer_test_is_disjoint():
    rows = cohort()
    plan = make_splits(rows, seed=42)
    assert plan == make_splits(list(reversed(rows)), seed=42)
    assert plan != make_splits(rows, seed=43)
    validate_splits(rows, plan)
    all_patients = {r['patient_id'] for r in rows}
    scored = []
    for fold in plan['folds']:
        groups = {role: set(fold[role]) for role in ['train', 'validation', 'test']}
        assert set.union(*groups.values()) == all_patients
        assert not (groups['train'] & groups['validation'] or groups['test'] & groups['train']
                    or groups['test'] & groups['validation'])
        grades = {r['patient_id']: r['grade'] for r in rows}
        assert Counter(grades[p] for p in groups['test'])[0] == 2
        assert Counter(grades[p] for p in groups['validation'])[0] == 2
        assert Counter(grades[p] for p in groups['train'])[0] == 6
        assert all(Counter(grades[p] for p in groups[role])[g] > 0
                   for role in groups for g in range(4))
        scored.extend(groups['test'])
    assert Counter(scored) == Counter({p: 1 for p in all_patients})
    assert sorted(len(f['test']) for f in plan['folds']) == [21, 22, 22, 22, 22]


@pytest.mark.parametrize('kind', ['source', 'pixels', 'tensor', 'grade', 'image_id'])
def test_conflicts_or_cross_patient_duplicates_fail_before_splitting(kind):
    rows = cohort()
    if kind in ['source', 'pixels', 'tensor']:
        field = {'source': 'source_sha256', 'pixels': 'decoded_pixel_sha256',
                 'tensor': 'tensor_pixel_sha256'}[kind]
        rows[-1][field] = rows[0][field]
    elif kind == 'grade':
        rows.append(dict(rows[0], image_id='other', grade=1))
    else:
        rows[-1]['image_id'] = rows[0]['image_id']
    with pytest.raises(ValueError):
        make_splits(rows)


def test_modified_split_or_membership_is_rejected():
    rows = cohort()
    plan = make_splits(rows)
    plan['folds'][0]['train'].append(plan['folds'][0]['test'][0])
    with pytest.raises(ValueError):
        validate_splits(rows, plan)


def test_training_weights_balance_classes_and_patients_without_using_validation():
    train = [{'group_id': 'A', 'grade': 0}] * 2 + [{'group_id': 'B', 'grade': 0}] * 5
    train += [{'group_id': 'C', 'grade': 1}] * 3
    weights, class_weights = patient_weights(train)
    assert np.mean(weights) == pytest.approx(1)
    totals = defaultdict(float)
    for row, weight in zip(train, weights): totals[row['group_id']] += float(weight)
    assert totals['A'] == pytest.approx(totals['B'])
    assert totals['C'] == pytest.approx(totals['A'] + totals['B'])
    val = [{'group_id': 'V', 'grade': 0}] * 9 + [{'group_id': 'W', 'grade': 1}]
    val_weights, reused = patient_weights(val, class_weights)
    assert reused == class_weights
    assert val_weights.sum() > 0


def portable_fixture(tmp_path):
    rows = []
    for i, role in enumerate(['train', 'train', 'validation']):
        path = tmp_path / f'F{i:06d}.png'
        Image.fromarray(np.full((224, 224), 60 + i, np.uint8)).save(path)
        rows.append({'frame_id': f'F{i:06d}', 'group_id': f'G{i:04d}', 'grade': i % 2,
                     'role': role, 'file': path.name,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'schema_version': 1, 'fold': 0, 'allowed_roles': ['train', 'validation'],
                'frames': rows}
    path = tmp_path / 'training_manifest.json'
    path.write_text(json.dumps(manifest))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_is_pixel_exact_float32_integer_labels_and_cannot_open_outer_test(tmp_path):
    path, digest = portable_fixture(tmp_path)
    store = FrameStore(path.parent, digest)
    rows = store.partition('train')
    batches = list(store.batches(rows, batch_size=2, shuffle=False))
    x, y, w = batches[0]
    assert x.shape == (2, 224, 224, 3) and x.dtype == np.float32
    np.testing.assert_array_equal(x[0], np.full((224, 224, 3), 60, np.float32))
    assert y.tolist() == [0, 1] and np.issubdtype(y.dtype, np.integer)
    assert w.dtype == np.float32
    with pytest.raises(ValueError, match='test|role'):
        store.partition('test')
    with pytest.raises(ValueError):
        list(store.batches([dict(rows[0], file='../outside.png')], 1))


@pytest.mark.parametrize('kind', ['hash', 'traversal', 'label', 'overlap', 'test_role'])
def test_loader_rejects_tampered_or_invalid_manifest(tmp_path, kind):
    path, digest = portable_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    if kind == 'hash':
        (tmp_path / manifest['frames'][0]['file']).write_bytes(b'corrupt')
    else:
        if kind == 'traversal': manifest['frames'][0]['file'] = '../private.png'
        elif kind == 'label': manifest['frames'][0]['grade'] = 4
        elif kind == 'overlap': manifest['frames'][-1]['group_id'] = manifest['frames'][0]['group_id']
        else: manifest['frames'][0]['role'] = 'test'
        path.write_text(json.dumps(manifest));digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        FrameStore(path.parent, digest)


def test_patient_outputs_average_probabilities_and_metrics_have_four_class_counts():
    rows = [{'group_id': 'A', 'grade': 0}, {'group_id': 'A', 'grade': 0},
            {'group_id': 'B', 'grade': 1}, {'group_id': 'C', 'grade': 2}, {'group_id': 'D', 'grade': 3}]
    probs = np.array([[.8,.1,.05,.05],[.6,.2,.1,.1],[.05,.8,.1,.05],
                      [.1,.1,.7,.1],[.1,.1,.1,.7]])
    patients = patient_predictions(rows, probs)
    np.testing.assert_allclose(patients[0]['probabilities'], [.7,.15,.075,.075])
    metrics = classification_metrics(patients)
    assert metrics['confusion_matrix'] == np.eye(4, dtype=int).tolist()
    assert metrics['balanced_accuracy'] == metrics['macro_f1'] == 1


def test_missing_approval_cannot_import_framework_create_output_or_fit(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == 'tensorflow': pytest.fail('Framework import before Gate E approval.')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    with pytest.raises(PermissionError, match='Gate E'):
        require_training_approval({}, {}, 'manifest')
    with pytest.raises(PermissionError, match='Gate E'):
        run_training(tmp_path, 'manifest', {}, {}, tmp_path/'output')
    assert not (tmp_path/'output').exists()
