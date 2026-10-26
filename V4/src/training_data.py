"""Portable lossless model inputs and patient metrics; no framework or fitting."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def patient_weights(rows, class_weights=None):
    if not rows:
        raise ValueError('Cannot weight an empty partition.')
    groups = {}
    for row in rows:
        g, y = row['group_id'], row['grade']
        if type(y) is not int or y not in range(4) or (g in groups and groups[g] != y):
            raise ValueError('Invalid patient grade.')
        groups[g] = y
    counts = Counter(groups.values())
    if class_weights is None:
        class_weights = {str(y): len(groups) / (len(counts) * count) for y, count in counts.items()}
    if any(str(y) not in class_weights or not np.isfinite(class_weights[str(y)])
           or class_weights[str(y)] <= 0 for y in counts):
        raise ValueError('Class weights must come from the training patients.')
    frames = Counter(r['group_id'] for r in rows)
    weights = np.array([class_weights[str(r['grade'])] / frames[r['group_id']] for r in rows], np.float32)
    weights /= weights.mean()
    return weights, dict(class_weights)


class FrameStore:
    def __init__(self, root, manifest_sha256):
        self.root = Path(root).resolve()
        path = self.root / 'training_manifest.json'
        if digest(path) != manifest_sha256:
            raise ValueError('Training manifest hash differs from the reviewed package.')
        self.manifest = json.loads(path.read_text())
        if (self.manifest.get('schema_version') != 1 or self.manifest.get('fold') != 0
                or self.manifest.get('allowed_roles') != ['train', 'validation']):
            raise ValueError('Pilot allows only fold 0 train and inner-validation roles.')
        self.rows = self.manifest['frames']
        assets = self.manifest.get('assets', [])
        if (len({a['file'] for a in assets}) != len(assets)
                or any(Path(a['file']).name != a['file'] for a in assets)):
            raise ValueError('Unsafe or repeated asset path.')
        allowed = {'training_manifest.json', 'dataset-metadata.json', 'gate_e_approval.json'}
        allowed.update(a['file'] for a in assets)
        allowed.update(r['file'] for r in self.rows)
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in self.root.iterdir()):
            raise ValueError('Unexpected file in the private training directory.')
        ids, groups = set(), {}
        for row in self.rows:
            key, group, role = row['frame_id'], row['group_id'], row['role']
            if (not re.fullmatch(r'F\d{6}', key) or key in ids
                    or not re.fullmatch(r'G\d{4}', group)
                    or role not in ['train', 'validation'] or type(row['grade']) is not int
                    or row['grade'] not in range(4) or row['file'] != key + '.png'
                    or (group in groups and groups[group] != (role, row['grade']))):
                raise ValueError('Invalid frame, label, path, or overlapping patient roles.')
            ids.add(key); groups[group] = (role, row['grade'])
            image = self.root / row['file']
            if not image.is_file() or image.is_symlink() or digest(image) != row['sha256']:
                raise ValueError('Image path/hash differs from reviewed derivative.')
        if not self.rows or {r['role'] for r in self.rows} != {'train', 'validation'}:
            raise ValueError('Both pilot partitions must be nonempty.')
        self._indexed = {r['frame_id']: dict(r) for r in self.rows}

    def partition(self, role):
        if role not in ['train', 'validation']:
            raise ValueError('Outer test role is inaccessible in the pilot package.')
        return [r for r in self.rows if r['role'] == role]

    def load(self, row):
        if row != self._indexed.get(row.get('frame_id')):
            raise ValueError('Unknown or changed frame record.')
        path = self.root / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('Frame changed after preflight.')
        with Image.open(path) as image:
            if image.mode != 'L' or image.size != (224, 224) or image.getexif():
                raise ValueError('Expected lossless 224x224 grayscale PNG without EXIF.')
            gray = np.asarray(image).copy()
        return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)

    def batches(self, rows, batch_size=16, shuffle=False, seed=42, class_weights=None):
        if not rows or batch_size < 1 or any(r != self._indexed.get(r.get('frame_id')) for r in rows):
            raise ValueError('Invalid batch rows or batch size.')
        weights, _ = patient_weights(rows, class_weights)
        indices = np.arange(len(rows))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(rows), batch_size):
            selected = indices[start:start + batch_size]
            yield (np.stack([self.load(rows[i]) for i in selected]),
                   np.array([rows[i]['grade'] for i in selected], np.int64), weights[selected])


def patient_predictions(rows, probabilities):
    probs = np.asarray(probabilities, dtype=np.float64)
    if (probs.shape != (len(rows), 4) or not np.isfinite(probs).all()
            or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Expected normalized four-class frame probabilities.')
    grouped = defaultdict(list); labels = {}
    for row, p in zip(rows, probs):
        group, grade = row['group_id'], row['grade']
        if group in labels and labels[group] != grade:
            raise ValueError('Conflicting patient labels.')
        labels[group] = grade; grouped[group].append(p)
    return [{'group_id': group, 'grade': labels[group], 'frames': len(values),
             'probabilities': np.mean(values, axis=0).tolist(), 'prediction': int(np.argmax(np.mean(values, axis=0)))}
            for group, values in sorted(grouped.items())]


def classification_metrics(patients):
    if not patients:
        raise ValueError('No patient predictions.')
    confusion = np.zeros((4, 4), np.int64)
    for row in patients: confusion[row['grade'], row['prediction']] += 1
    tp = np.diag(confusion).astype(float); support = confusion.sum(axis=1); predicted = confusion.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(4), where=predicted > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(4), where=(precision+recall) > 0)
    return {'patients': len(patients), 'confusion_matrix': confusion.tolist(),
            'balanced_accuracy': float(recall[support > 0].mean()), 'macro_f1': float(f1.mean()),
            'support': support.tolist(), 'class_precision': precision.tolist(), 'class_recall': recall.tolist()}
