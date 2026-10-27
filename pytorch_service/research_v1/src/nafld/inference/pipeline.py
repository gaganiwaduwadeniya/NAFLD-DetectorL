"""Independent task ensembles, exact saved preprocessing, and binary-gated staging."""

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from nafld.models.factory import build_model
from nafld.preprocessing.images import ImagePreprocessor, preprocessing_metadata
from nafld.utils.config import TASK_LABELS
from nafld.utils.io import sha256_file


def validate_export_manifest(manifest):
    task = manifest.get('task')
    if (manifest.get('schema_version') != 1 or manifest.get('trained') is not True
            or task not in TASK_LABELS or manifest.get('labels') != TASK_LABELS[task]
            or manifest.get('final_model') is not False or not manifest.get('models')
            or not manifest.get('model_version') or not manifest.get('dataset_version')):
        raise ValueError('A complete, trained research export with exact task labels is required')
    models = manifest['models']
    if (len(models) != 5 or sorted(m.get('fold', -1) for m in models) != list(range(5))
            or len({m.get('experiment_id') for m in models}) != 5
            or not manifest.get('candidate_id') or not manifest.get('code_version')):
        raise ValueError('Research export must contain five distinct fold models and complete provenance')


def verified_export_path(root, descriptor):
    root = Path(root).resolve()
    path = (root / descriptor['file']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != descriptor['sha256']:
        raise ValueError('Export file is missing, unsafe, or has a SHA256 mismatch')
    return path


class TaskPredictor:
    def __init__(self, export_root, *, device='cpu'):
        self.root = Path(export_root).resolve()
        manifest = json.loads((self.root / 'manifest.json').read_text())
        validate_export_manifest(manifest)
        self.task, self.version = manifest['task'], manifest['model_version']
        self.dataset_version, self.candidate_id = manifest['dataset_version'], manifest['candidate_id']
        self.device = torch.device(device)
        if self.device.type not in {'cpu', 'cuda'} or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Inference device must be available CPU or CUDA')
        registry = verified_export_path(self.root, manifest['model_registry'])
        self.models, self.preprocessors, self.member_ids = [], [], []
        for member in manifest['models']:
            payload = torch.load(verified_export_path(self.root, member), map_location='cpu', weights_only=True)
            cfg, metadata = payload['config'].copy(), payload['model_metadata']
            if (payload.get('schema_version') != 1 or payload.get('checkpoint_role') != 'best'
                    or payload['identity']['experiment_id'] != member['experiment_id']
                    or payload['identity']['dataset_version'] != self.dataset_version
                    or payload['identity']['code_version'] != manifest['code_version']
                    or cfg.get('fold') != member['fold']
                    or payload.get('completed_epoch', 0) < 1
                    or cfg['task'] != self.task or metadata['task'] != self.task
                    or metadata['labels'] != TASK_LABELS[self.task]
                    or not metadata['pretrained_loaded'] or metadata['architecture_check_only']):
                raise ValueError('Export checkpoint lacks matching trained provenance')
            cfg['models_config'] = str(registry)
            if payload['storage'] == 'classifier_only' and cfg['phase'] == 'screening':
                base = manifest.get('base_weights')
                if not base or base['sha256'] != metadata['weights']['sha256']:
                    raise ValueError('Screening export requires its exact native pretrained encoder')
                cfg.update(weights_path=str(verified_export_path(self.root, base)), pretrained=True)
                model = build_model(cfg)
                model.classifier.load_state_dict(payload['model_state'], strict=True)
            elif payload['storage'] == 'full_model' and cfg['phase'] == 'finetune':
                # Build structure without downloads; immediately replace ALL random tensors.
                model = build_model({**cfg, 'pretrained': False}, architecture_check=True)
                model.load_state_dict(payload['model_state'], strict=True)
            else:
                raise ValueError('Export checkpoint storage does not match its training phase')
            for key in ['registry_sha256', 'native_model', 'backend', 'backbone', 'num_classes']:
                if model.metadata[key] != metadata[key]:
                    raise ValueError('Installed model structure/registry differs from the trained checkpoint')
            if model.metadata['backend_version'] != metadata['backend_version'] or str(torch.__version__) != str(metadata['torch_version']):
                raise ValueError('Export requires the recorded torch/backend library versions')
            if preprocessing_metadata(cfg) != metadata['preprocessing']:
                raise ValueError('Preprocessing differs from the trained checkpoint')
            model.metadata = metadata
            model.resolved_config = cfg
            self.models.append(model.to(self.device).eval())
            self.preprocessors.append(ImagePreprocessor(cfg, training=False))
            self.member_ids.append(member['experiment_id'])

    @torch.inference_mode()
    def predict_image(self, image):
        probabilities = []
        for model, preprocessor in zip(self.models, self.preprocessors):
            pixels = torch.from_numpy(preprocessor.process(image).pixels).unsqueeze(0).to(self.device)
            probabilities.append(model(pixels).float().softmax(dim=1)[0].cpu().numpy())
        result = np.stack(probabilities).mean(axis=0)
        _validate_probability(result, len(TASK_LABELS[self.task]))
        return result


def _validate_probability(probability, width):
    p = np.asarray(probability, dtype=float)
    if p.shape != (width,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isclose(p.sum(), 1):
        raise ValueError('Model returned invalid class probabilities')
    return p


class CascadePredictor:
    def __init__(self, binary, staging):
        if binary.task != 'binary' or staging.task != 'staging' or binary.dataset_version != staging.dataset_version:
            raise ValueError('Cascade requires independent binary/staging models with matching data provenance')
        self.binary, self.staging = binary, staging

    def predict_image(self, image):
        binary = _validate_probability(self.binary.predict_image(image), 2)
        result = {'stage': 0, 'binary_probabilities': {'Non-NAFLD': float(binary[0]), 'NAFLD': float(binary[1])},
                  'conditional_stage_probabilities': None, 'joint_stage_probabilities': None,
                  'staging_applied': False, 'threshold': .5,
                  'model_version': {'binary': self.binary.version, 'staging': self.staging.version},
                  'dataset_version': self.binary.dataset_version, 'research_prototype': True,
                  'final_model': False, 'probability_calibration_verified': False}
        if binary[1] >= .5:
            conditional = _validate_probability(self.staging.predict_image(image), 3)
            result.update(stage=int(conditional.argmax())+1, staging_applied=True,
                          conditional_stage_probabilities={str(i+1): float(p) for i, p in enumerate(conditional)},
                          joint_stage_probabilities={'0': float(binary[0]),
                              **{str(i+1): float(binary[1]*p) for i, p in enumerate(conditional)}})
        return result

    def predict(self, path):
        content = Path(path).read_bytes()
        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                result = self.predict_image(image)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError(f'Invalid inference image: {type(exc).__name__}') from exc
        return {**result, 'input_sha256': hashlib.sha256(content).hexdigest()}
