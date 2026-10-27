"""YAML loading, initial training contracts, and content-addressed experiment IDs."""

import hashlib
import json
from pathlib import Path

import yaml

BACKBONES = {'resnet18', 'densenet121', 'efficientnet_b0', 'efficientnetv2_s', 'convnext_tiny'}
TASK_LABELS = {'binary': ['Non-NAFLD', 'NAFLD'],
               'staging': ['Stage 1', 'Stage 2', 'Stage 3'],
               'four_class': ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3']}


def load_config(path):
    with Path(path).open(encoding='utf-8') as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return value


def validate_training(cfg):
    if cfg.get('task') not in TASK_LABELS:
        raise ValueError('Unknown task')
    if cfg.get('backbone') not in BACKBONES:
        raise ValueError('Unknown backbone')
    if cfg.get('phase') not in {'screening', 'finetune'}:
        raise ValueError('phase must be screening or finetune')
    for key in ['image_size', 'batch_size', 'epochs', 'patience']:
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['fold', 'workers', 'seed']:
        if type(cfg.get(key)) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if cfg['fold'] >= 5:
        raise ValueError('fold must be between 0 and 4')
    if cfg.get('learning_rate', 0) <= 0 or cfg.get('weight_decay', -1) < 0:
        raise ValueError('Invalid learning rate or weight decay')
    if cfg.get('gradient_clip', 0) <= 0:
        raise ValueError('gradient_clip must be positive')
    from nafld.preprocessing.images import validate_preprocessing

    validate_preprocessing(cfg)
    if not cfg.get('pretrained', False):
        raise ValueError('Pretrained weights required for planned benchmarks')


def experiment_identity_config(cfg):
    ignored = {'resume_path', 'output_root', 'allow_local_training', 'weights_path', 'models_config',
               'feature_cache_root', 'feature_cache_read_roots', 'kaggle_run_reference',
               'interruption_confirmed', 'restart_unstarted'}
    return {k: v for k, v in cfg.items() if k not in ignored}


def experiment_id(cfg, dataset_version, code_version):
    """Include all settings except filesystem/execution-only fields.

    Code version must be an immutable source digest or commit, not a mutable name.
    No successful-result or checkpoint validity claims are made by this helper.
    """
    settings = experiment_identity_config(cfg)
    payload = {'config': settings, 'dataset_version': dataset_version, 'code_version': code_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

