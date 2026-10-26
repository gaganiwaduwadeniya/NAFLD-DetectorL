"""Deterministic patient splits for review. This module cannot train or upload."""
from __future__ import annotations
import hashlib
import math


def _rank(seed, scope, value):
    return hashlib.sha256(f'{seed}|{scope}|{value}'.encode()).hexdigest()


def make_splits(rows, seed=42, n_folds=5, validation_fraction=.2):
    if n_folds != 5 or not isinstance(seed, int) or validation_fraction != .2:
        raise ValueError('This protocol fixes five outer folds and 20% inner validation.')
    patients, ids = {}, set()
    hashes = {key: {} for key in ('source_sha256', 'decoded_pixel_sha256', 'tensor_pixel_sha256')}
    for row in rows:
        pid, grade, image_id = row['patient_id'], row['grade'], row['image_id']
        if (type(grade) is not int or grade not in range(4) or image_id in ids
                or (pid in patients and patients[pid] != grade)):
            raise ValueError('Duplicate image ID or conflicting patient grades.')
        ids.add(image_id); patients[pid] = grade
        for key, index in hashes.items():
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError('Missing source/decoded/tensor integrity hash.')
            if value in index and index[value] != pid:
                raise ValueError('Cross-patient duplicate requires resolution before splitting.')
            index[value] = pid
    by_grade = {g: sorted(p for p, grade in patients.items() if grade == g) for g in range(4)}
    if any(len(group) < n_folds for group in by_grade.values()):
        raise ValueError('Every class needs at least five patients for this protocol.')
    outer = [[] for _ in range(n_folds)]
    for grade, group in by_grade.items():
        assigned = [0] * n_folds
        for pid in sorted(group, key=lambda p: _rank(seed, f'outer-{grade}', p)):
            fold = min(range(n_folds), key=lambda f: (assigned[f], len(outer[f]), _rank(seed, pid, f)))
            outer[fold].append(pid); assigned[fold] += 1
    folds = []
    for index, test in enumerate(outer):
        remaining = set(patients) - set(test)
        validation = []
        for grade in range(4):
            group = sorted((p for p in remaining if patients[p] == grade),
                           key=lambda p: _rank(seed, f'inner-{index}-{grade}', p))
            count = max(1, math.ceil(len(group) * validation_fraction))
            if count >= len(group):
                raise ValueError('No training patients remain after inner validation.')
            validation.extend(group[:count])
        fold = {'fold': index, 'train': sorted(remaining - set(validation)),
                'validation': sorted(validation), 'test': sorted(test)}
        fold['patient_counts_by_grade'] = {
            role: {str(g): sum(patients[p] == g for p in fold[role]) for g in range(4)}
            for role in ('train', 'validation', 'test')}
        fold['frame_counts'] = {role: sum(r['patient_id'] in set(fold[role]) for r in rows)
                                for role in ('train', 'validation', 'test')}
        folds.append(fold)
    return {'schema_version': 1, 'protocol': 'five_outer_patient_folds_fixed_inner_holdout_v1',
            'seed': seed, 'validation_fraction_within_outer_training': validation_fraction,
            'patient_count': len(patients), 'frame_count': len(rows), 'folds': folds,
            'evaluation_scope': 'Internal validation; all canonical patients were exposed to preprocessing QC.',
            'model_selection': 'Inner patient validation only; outer test is never used for early stopping or pilot feedback.',
            'training_approved': False}


def validate_splits(rows, plan):
    if plan != make_splits(rows, plan.get('seed'), validation_fraction=plan.get('validation_fraction_within_outer_training')):
        raise ValueError('Split manifest differs from the deterministic patient protocol.')
