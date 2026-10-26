"""Deterministic geometry and grayscale review; source pixels never change in place."""
from __future__ import annotations

import numpy as np
from PIL import Image

VERSION = 'gate-a-1.0'


def classify_layout(pixels):
    """Observable UI signatures, never class, patient, or scanner identity.

    A footer amber-text signature distinguishes edge-filling screenshots from
    gray-header/yellow-marker screenshots at the SAME dimensions. Unknown real
    layouts remain blocked; these signatures do not identify a scanner vendor.
    """
    h, w = pixels.shape[:2]
    if pixels.ndim != 3 or (w, h) not in {(1024, 768), (800, 600)}:
        return f'{w}x{h}'
    rgb = pixels.astype(np.int16)
    footer = rgb[int(h * .94):, int(w * .70):]
    amber = ((footer[..., 0] > 110) & (footer[..., 1] > 65) &
             (footer[..., 0] - footer[..., 2] > 60) &
             (footer[..., 1] - footer[..., 2] > 30))
    if int(amber.sum()) >= 8:
        return f'amber_footer_{w}x{h}'
    if (w, h) == (1024, 768):
        top = rgb[56:110, 110:900]
        yellow = (top[..., 0] > 130) & (top[..., 1] > 100) & (top[..., 2] < 140)
        if int(yellow.sum()) >= 8:
            return 'header_banner_1024x768'
    return f'unknown_{w}x{h}'


def checked_box(box, width, height):
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
        raise ValueError('Crop coordinates must be four integer source pixels.')
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError('Crop coordinates are outside source bounds or empty.')
    return left, top, right, bottom


def crop_pixels(pixels, box):
    left, top, right, bottom = checked_box(box, pixels.shape[1], pixels.shape[0])
    return pixels[top:bottom, left:right].copy()


def mask_pixels(pixels, rectangles):
    result = pixels.copy()
    for box in rectangles:
        left, top, right, bottom = checked_box(box, pixels.shape[1], pixels.shape[0])
        result[top:bottom, left:right] = 0
    return result


def require_preview_dtype(pixels):
    if pixels.dtype != np.uint8 or (pixels.ndim != 2 and not (pixels.ndim == 3 and pixels.shape[2] == 3)):
        raise ValueError('Gate A rendering supports uint8 L/RGB only; format-specific review required.')


def propose_crop(pixels, black_threshold=8, margin_px=8):
    """Bounding box of nonblack display content, NOT an anatomical segmentation.

    All nonblack components are retained, including overlays. Detection failure
    returns a flagged full-field fallback. No largest-component tissue removal.
    """
    require_preview_dtype(pixels)
    if not 0 <= black_threshold <= 255 or margin_px < 0:
        raise ValueError('Invalid black threshold or margin.')
    active = pixels.max(axis=2) > black_threshold if pixels.ndim == 3 else pixels > black_threshold
    yy, xx = np.nonzero(active)
    h, w = pixels.shape[:2]
    if not len(xx):
        return [0, 0, w, h], ['empty_detection_full_field_fallback']
    box = [max(0, int(xx.min()) - margin_px), max(0, int(yy.min()) - margin_px),
           min(w, int(xx.max()) + margin_px + 1), min(h, int(yy.max()) + margin_px + 1)]
    warnings = []
    if (box[2] - box[0]) * (box[3] - box[1]) < w * h * .5:
        warnings.append('large_area_removed_needs_visual_review')
    return box, warnings


def letterbox(pixels, size=224):
    """Bilinear antialiased Pillow resize; preserve aspect, center on zero padding.

    Round resized dimensions half-up; odd extra padding goes right/bottom.
    Preserve source RGB/L values and orientation. No auto-contrast, grayscale,
    intensity remapping, EXIF transpose, or model tensor generation at Gate A.
    """
    require_preview_dtype(pixels)
    if not isinstance(size, int) or size < 1:
        raise ValueError('Output size must be positive.')
    h, w = pixels.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(w * scale + .5)), max(1, int(h * scale + .5))
    resized = np.asarray(Image.fromarray(pixels).resize((nw, nh), Image.Resampling.BILINEAR))
    shape = (size, size) + pixels.shape[2:]
    result = np.zeros(shape, dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    result[y:y + nh, x:x + nw] = resized
    return result, {'resized_size': [nw, nh], 'padding_ltrb': [x, y, size - nw - x, size - nh - y],
                    'padding_fraction': 1 - nw * nh / (size * size), 'interpolation': 'Pillow BILINEAR antialiased',
                    'rounding': 'half_up', 'padding_value': 0, 'orientation': 'unchanged'}


def grayscale_preserve(pixels):
    """Fixed BT.601 luminance in source range, with integer half-up rounding.

    uint8 RGB -> floor((299*R + 587*G + 114*B + 500)/1000).
    uint8 L is copied unchanged. No intensity normalization or enhancement.
    Color Doppler/non-B-mode review is the caller's responsibility before this
    function is used; the Gate B renderer explicitly requires that evidence.
    """
    require_preview_dtype(pixels)
    if pixels.ndim == 2:
        return pixels.copy()
    rgb = pixels.astype(np.uint32)
    return ((299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2] + 500) // 1000).astype(np.uint8)


def grayscale_preview_tensor(gray, size=224):
    """Review-only tensor contract: uint8 L -> padded float32 RGB, range 0–255.

    The tensor is not materialized for a whole cohort or sent to any model.
    This helper is proposed until Gate B/C approval and pilot validation.
    """
    if gray.ndim != 2:
        raise ValueError('A 2D grayscale image is required.')
    padded, metadata = letterbox(gray, size)
    return np.repeat(padded[..., None], 3, axis=2).astype(np.float32), metadata


def validate_intensity_config(config):
    if config.get('candidates') != ['I0_preserve']:
        raise ValueError('This bounded comparison implements only I0_preserve.')
    if config.get('optional_intensity_operation') != 'identity':
        raise ValueError('Only the proposed identity intensity operation is supported.')
    if config.get('grayscale_method') != 'BT601_integer_299_587_114_half_up' or config.get('output_size') != 224:
        raise ValueError('Unsupported grayscale or output contract; refresh the proposal explicitly.')
    if config.get('statistic_threshold', 8) != 8:
        raise ValueError('This review uses the documented fixed statistic threshold 8.')


def preprocess_decoded(pixels, crop_config, intensity_config):
    """One shared approved operation order for review and bounded pilot use.

    Approval, modality, source provenance, and cohort limits are checked by the
    caller. Unknown layouts fail rather than silently choose an unapproved crop.
    The source array is never changed in place; no review annotations are drawn.
    """
    require_preview_dtype(pixels)
    validate_intensity_config(intensity_config)
    if crop_config.get('output_size', 224) != 224:
        raise ValueError('Unapproved geometry output size.')
    layout_key = classify_layout(pixels)
    layout = crop_config.get('layouts', {}).get(layout_key)
    if not layout or layout.get('privacy_reviewed') is not True or not layout.get('template_box'):
        raise ValueError('Unknown or unapproved layout; needs_review with full-field review required.')
    if layout.get('selector') != {'width': pixels.shape[1], 'height': pixels.shape[0]}:
        raise ValueError('Unapproved source dimensions/layout selector.')
    if not layout.get('mask_rectangles'):
        raise ValueError('Explicit approved privacy masks are required.')
    deidentified = mask_pixels(pixels, layout['mask_rectangles'])
    cropped = crop_pixels(deidentified, layout['template_box'])
    gray = grayscale_preserve(cropped)
    tensor, resize = grayscale_preview_tensor(gray, 224)
    return {'deidentified': deidentified, 'cropped': cropped, 'gray': gray, 'tensor': tensor,
            'resize': resize, 'layout': layout_key, 'box_ltrb': list(layout['template_box']),
            'privacy_mask_rectangles': [list(box) for box in layout['mask_rectangles']]}
