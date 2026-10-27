"""Explicit backends, offline native weights, and one consistent linear head.

Architecture checks may initialize random parameters for dimensions/interfaces
only. Research training requires verified pretrained weights. The factory never
downloads weights or silently switches implementation backends.
"""

import copy
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

import torch
from torch import nn

from nafld.preprocessing.images import preprocessing_metadata, validate_preprocessing
from nafld.utils.config import BACKBONES, TASK_LABELS, load_config


class MissingPretrainedWeights(ValueError):
    """A real experiment cannot start until offline weights are supplied."""


def resolve_model_config(cfg: dict) -> dict:
    resolved = copy.deepcopy(cfg)
    backend = resolved.get('backend', 'timm')
    if backend not in {'timm', 'torchvision'}:
        raise ValueError('Unknown model backend')
    if resolved.get('backbone') not in BACKBONES or resolved.get('task') not in TASK_LABELS:
        raise ValueError('Unknown backbone or task')
    if resolved.get('phase') not in {'screening', 'finetune'}:
        raise ValueError('Unknown model phase')
    if type(resolved.get('image_size')) is not int or resolved['image_size'] < 32:
        raise ValueError('These backbones require image_size >= 32')
    dropout = resolved.get('head_dropout', .1)
    if type(dropout) not in {int, float} or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('head_dropout must be finite and in [0, 1)')
    if type(resolved.get('pretrained')) is not bool:
        raise ValueError('pretrained must be a boolean')
    if type(resolved.get('seed')) is not int or resolved['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    registry = load_config(resolved.get('models_config', 'configs/models.yaml'))
    if registry.get('tasks') != TASK_LABELS:
        raise ValueError('Model registry task labels contradict the protocol')
    definition = copy.deepcopy(registry['backbones'][resolved['backbone']])
    profile = definition[backend]
    policy = resolved.get('normalization_policy', 'pretrained')
    if policy == 'pretrained':
        resolved.update(copy.deepcopy(profile['normalization']))
    elif policy == 'custom':
        if not resolved.get('normalization_override_reason'):
            raise ValueError('A custom normalization comparison requires a reason')
    else:
        raise ValueError('normalization_policy must be pretrained or custom')
    resolved.update(backend=backend, head_dropout=float(dropout), normalization_policy=policy,
                    resolved_model_name=profile['name'], feature_dim=definition['features'],
                    num_classes=len(TASK_LABELS[resolved['task']]),
                    pretraining_data=profile['pretraining_data'],
                    registry_sha256=hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest())
    validate_preprocessing(resolved)
    return resolved


def create_head(feature_dim: int, task: str, *, dropout: float = .1) -> nn.Module:
    if task not in TASK_LABELS or type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError('Invalid classifier task or feature dimension')
    if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('Invalid classifier dropout')
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, len(TASK_LABELS[task])))


def read_native_weights(path, expected_sha256):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('A full lowercase SHA256 is required for pretrained weights')
    path = Path(path)
    if not path.is_file():
        raise MissingPretrainedWeights(f'Pretrained weights file is missing: {path}')
    payload_bytes = path.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Pretrained weights SHA256 mismatch')
    if path.suffix == '.safetensors':
        try:
            from safetensors.torch import load
        except ImportError as exc:
            raise RuntimeError('Install safetensors to read this weight cache') from exc
        payload = load(payload_bytes)
    else:
        payload = torch.load(BytesIO(payload_bytes), map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, dict) or not payload or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
        raise ValueError('Weights must be a native tensor state_dict, optionally wrapped by state_dict')
    return payload, {'path': str(path), 'sha256': digest, 'format': path.suffix}


def _torchvision_native(cfg):
    from torchvision import models

    registry = load_config(cfg.get('models_config', 'configs/models.yaml'))
    profile = registry['backbones'][cfg['backbone']]['torchvision']
    weights_enum = models.get_model_weights(profile['name'])[profile['weights']]
    actual_transform = weights_enum.transforms()
    if (list(actual_transform.mean) != profile['normalization']['mean']
            or list(actual_transform.std) != profile['normalization']['std']):
        raise ValueError('Installed torchvision weight normalization differs from the registry')
    return models.get_model(profile['name'], weights=None), {'weight_variant': profile['weights'],
                                                            'registered_weight_url': weights_enum.url}


def _timm_native(cfg):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError('timm is unavailable; install requirements. No backend substitution was made.') from exc
    native = timm.create_model(cfg['resolved_model_name'], pretrained=False, num_classes=1000)
    profile = native.pretrained_cfg
    registry = load_config(cfg.get('models_config', 'configs/models.yaml'))
    expected = registry['backbones'][cfg['backbone']]['timm']['normalization']
    if list(profile['mean']) != expected['mean'] or list(profile['std']) != expected['std']:
        raise ValueError('Installed timm weight normalization differs from the registry')
    return native, {'weight_variant': cfg['resolved_model_name'], 'native_pretrained_cfg': copy.deepcopy(profile)}


def _remove_native_classifier(native, cfg):
    backbone = cfg['backbone']
    if cfg['backend'] == 'timm':
        features = int(native.num_features)
        native.reset_classifier(0)
    elif backbone == 'resnet18':
        features = native.fc.in_features
        native.fc = nn.Identity()
    elif backbone == 'densenet121':
        features = native.classifier.in_features
        native.classifier = nn.Identity()
    elif backbone in {'efficientnet_b0', 'efficientnetv2_s'}:
        features = native.classifier[-1].in_features
        native.classifier = nn.Identity()  # common head owns classifier dropout
    else:
        features = native.classifier[-1].in_features
        native.classifier[-1] = nn.Identity()  # retain ConvNeXt's final normalization/flatten
    if features != cfg['feature_dim']:
        raise ValueError('Native model feature dimension differs from the registry')
    return native


class ClassificationModel(nn.Module):
    def __init__(self, extractor, cfg, metadata):
        super().__init__()
        self.extractor = extractor
        self.resolved_config = copy.deepcopy(cfg)
        self.feature_dim = cfg['feature_dim']
        self.num_classes = cfg['num_classes']
        self.classifier = create_head(self.feature_dim, cfg['task'], dropout=cfg['head_dropout'])
        self.metadata = metadata
        self.set_phase(cfg['phase'])

    def set_phase(self, phase):
        if phase not in {'screening', 'finetune'}:
            raise ValueError('Unknown model phase')
        self.phase = phase
        self.resolved_config['phase'] = phase
        self.metadata['phase'] = phase
        self.extractor.requires_grad_(phase == 'finetune')
        self.classifier.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'phase', None) == 'screening':
            self.extractor.eval()
        return self

    def extract_features(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError('Expected images shaped [N, 3, H, W]')
        if self.phase == 'screening':
            with torch.no_grad():
                features = self.extractor(pixels)
        else:
            features = self.extractor(pixels)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError('Unexpected backbone feature shape')
        return features

    def classify_features(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError('Expected pooled feature vectors')
        return self.classifier(features)

    def forward(self, pixels=None, *, features=None):
        if features is not None:
            if self.phase != 'screening' or pixels is not None:
                raise ValueError('Cached features are supported only for frozen screening')
            return self.classify_features(features)
        if pixels is None:
            raise ValueError('Provide images or frozen screening features')
        return self.classify_features(self.extract_features(pixels))


def build_model(cfg: dict, *, architecture_check: bool = False) -> ClassificationModel:
    cfg = resolve_model_config(cfg)
    if cfg['pretrained'] is False and not architecture_check:
        raise ValueError('Random initialization is allowed only with architecture_check=True')
    if cfg['pretrained'] is True and architecture_check:
        raise ValueError('architecture_check must explicitly set pretrained=False')
    state, weights_info = None, None
    if cfg['pretrained']:
        if not cfg.get('weights_path'):
            raise MissingPretrainedWeights('Set weights_path and weights_sha256 to verified native pretrained weights; automatic downloads are disabled')
        state, weights_info = read_native_weights(cfg['weights_path'], cfg.get('weights_sha256'))
    # Deterministic classifier initialization without perturbing the caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'])
        native, native_info = _torchvision_native(cfg) if cfg['backend'] == 'torchvision' else _timm_native(cfg)
        adaptation = None
        if state is not None:
            if cfg['backend'] == 'torchvision' and cfg['backbone'] == 'densenet121':
                # torchvision's documented legacy DenseNet checkpoint conversion.
                pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$')
                if any(pattern.match(key) for key in state):
                    converted = {pattern.sub(r'\1\2', key): value for key, value in state.items()}
                    if len(converted) != len(state):
                        raise ValueError('DenseNet legacy conversion would collide with existing keys')
                    state = converted
                    adaptation = 'torchvision_legacy_densenet_keys'
            try:
                native.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ValueError('Weights are incompatible with this exact backend/variant') from exc
        native = _remove_native_classifier(native, cfg)
        task = cfg['task']
        stage_map = {0: 1, 1: 2, 2: 3} if task == 'staging' else {i: i for i in range(4)} if task == 'four_class' else None
        metadata = {'backbone': cfg['backbone'], 'backend': cfg['backend'], 'native_model': cfg['resolved_model_name'],
                    'task': task, 'labels': list(TASK_LABELS[task]), 'output_index_to_stage': stage_map,
                    'outputs': 'unnormalized_logits', 'num_classes': cfg['num_classes'], 'feature_dim': cfg['feature_dim'],
                    'head_dropout': cfg['head_dropout'], 'seed': cfg['seed'],
                    'pretrained_loaded': state is not None, 'architecture_check_only': architecture_check,
                    'weights': weights_info, 'state_dict_adaptation': adaptation,
                    'registry_sha256': cfg['registry_sha256'], 'pretraining_data': cfg['pretraining_data'],
                    'preprocessing': preprocessing_metadata(cfg), 'torch_version': torch.__version__,
                    'backend_version': version(cfg['backend']), **native_info}
        return ClassificationModel(native, cfg, metadata)
