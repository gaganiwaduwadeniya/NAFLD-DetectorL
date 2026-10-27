"""Preserve the complete image field while producing normalized CHW arrays.

No statistics are estimated from this dataset. No anatomy-dependent crop,
histogram equalization, denoising, or contrast modification is applied.
"""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class PreparedImage:
    pixels: np.ndarray
    geometry: dict[str, Any]


def validate_preprocessing(cfg: dict) -> None:
    if type(cfg.get('image_size')) is not int or cfg['image_size'] < 1:
        raise ValueError('image_size must be a positive integer')
    for key in ['mean', 'std']:
        values = cfg.get(key)
        if (not isinstance(values, (list, tuple)) or len(values) != 3
                or not all(type(x) in {int, float} and math.isfinite(x) for x in values)):
            raise ValueError(f'{key} must contain three finite numbers')
    if any(x <= 0 for x in cfg['std']):
        raise ValueError('Normalization standard deviations must be positive')
    preprocessing = cfg.get('preprocessing', {})
    if not isinstance(preprocessing, dict):
        raise ValueError('preprocessing must be a mapping')
    unknown = set(preprocessing) - {'resize_mode', 'interpolation', 'padding_value'}
    if unknown:
        raise ValueError(f'Unsupported preprocessing options: {sorted(unknown)}')
    if preprocessing.get('resize_mode', 'letterbox') != 'letterbox':
        raise ValueError('resize_mode must be letterbox; automatic cropping is not enabled')
    if preprocessing.get('interpolation', 'bilinear') not in {'bilinear', 'bicubic'}:
        raise ValueError('interpolation must be bilinear or bicubic')
    padding = preprocessing.get('padding_value', 0)
    if type(padding) is not int or not 0 <= padding <= 255:
        raise ValueError('padding_value must be an integer between 0 and 255')
    augmentation = cfg.get('augmentation', {})
    if not isinstance(augmentation, dict):
        raise ValueError('augmentation must be a mapping')
    unknown = set(augmentation) - {'horizontal_flip', 'approved'}
    if unknown:
        raise ValueError(f'Unsupported augmentation options: {sorted(unknown)}')
    if any(type(value) is not bool for value in augmentation.values()):
        raise ValueError('Augmentation options must be booleans')
    if augmentation.get('horizontal_flip', False) and not augmentation.get('approved', False):
        raise ValueError('Clinical augmentation approval is required for horizontal flipping')


def preprocessing_metadata(cfg: dict) -> dict:
    validate_preprocessing(cfg)
    preprocessing = cfg.get('preprocessing', {})
    return {
        'schema_version': 1,
        'pipeline': 'exif_orientation_rgb_letterbox_float32_chw_normalize',
        'image_size': cfg['image_size'],
        'resize_mode': 'letterbox',
        'interpolation': preprocessing.get('interpolation', 'bilinear'),
        'padding_value': preprocessing.get('padding_value', 0),
        'mean': list(cfg['mean']), 'std': list(cfg['std']),
        'normalization_source': 'fixed_config_not_fitted',
        'input_modes': ['RGB', 'L'],
        'output_dtype': 'float32', 'output_layout': 'CHW', 'input_scale': 'uint8/255',
        'augmentation': {'horizontal_flip': cfg.get('augmentation', {}).get('horizontal_flip', False),
                         'approved': cfg.get('augmentation', {}).get('approved', False),
                         'applies_to': 'training_only'},
        'cropping': None,
    }


class ImagePreprocessor:
    """Pillow/NumPy core shared with future single-image inference.

    Input is explicitly limited to 8-bit RGB/grayscale images. Rejecting other
    modes prevents silent clipping of high-bit-depth medical images. A seeded
    RNG is required only if approved training augmentation is enabled.
    """

    def __init__(self, cfg: dict, *, training: bool = False):
        self.metadata = preprocessing_metadata(cfg)
        self.training = training
        self.size = self.metadata['image_size']
        self.mean = np.asarray(self.metadata['mean'], dtype=np.float32)[:, None, None]
        self.std = np.asarray(self.metadata['std'], dtype=np.float32)[:, None, None]

    def process(self, image: Image.Image, *, rng: random.Random | None = None) -> PreparedImage:
        if image.mode not in {'RGB', 'L'}:
            raise ValueError(f'Unsupported image mode {image.mode}; expected 8-bit RGB or L')
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images require an explicit frame-selection policy')
        original_size = list(image.size)
        image = ImageOps.exif_transpose(image).convert('RGB')
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions must be positive')
        ratio = min(self.size / width, self.size / height)
        resized = [max(1, min(self.size, round(width * ratio))),
                   max(1, min(self.size, round(height * ratio)))]
        resampling = Image.Resampling.BILINEAR if self.metadata['interpolation'] == 'bilinear' else Image.Resampling.BICUBIC
        resized_image = image.resize(tuple(resized), resample=resampling)
        left, top = (self.size - resized[0]) // 2, (self.size - resized[1]) // 2
        padding = self.metadata['padding_value']
        canvas = Image.new('RGB', (self.size, self.size), (padding, padding, padding))
        canvas.paste(resized_image, (left, top))
        flipped = False
        if self.training and self.metadata['augmentation']['horizontal_flip']:
            if rng is None:
                raise ValueError('A seeded RNG is required for training augmentation')
            flipped = rng.random() < .5
            if flipped:
                canvas = ImageOps.mirror(canvas)
        pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.)
        pixels = np.ascontiguousarray((pixels - self.mean) / self.std, dtype=np.float32)
        if not np.isfinite(pixels).all():
            raise ValueError('Normalization produced non-finite pixels')
        geometry = {'original_size': original_size, 'oriented_size': [width, height],
                    'resized_size': resized, 'output_size': [self.size, self.size],
                    'padding_ltrb': [left, top, self.size - resized[0] - left, self.size - resized[1] - top],
                    'horizontal_flip': flipped}
        return PreparedImage(pixels=pixels, geometry=geometry)

    def load(self, path: str | Path, *, expected_sha256: str | None = None,
             rng: random.Random | None = None) -> PreparedImage:
        """Verify and decode the same bytes; fail rather than skipping bad data."""
        path = Path(path)
        payload = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f'SHA256 mismatch for image: {path}')
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                return self.process(image, rng=rng)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError(f'Image decoding failed for {path}: {type(exc).__name__}') from exc

