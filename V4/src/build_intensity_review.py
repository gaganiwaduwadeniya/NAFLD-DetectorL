"""Bounded Gate B grayscale review after actual, hash-matched crop approval."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .audit_dataset import sha256, write_csv, write_json
from .build_review import configuration_hash, load_checked_source
from .preprocess import classify_layout, crop_pixels, preprocess_decoded, validate_intensity_config


def require_crop_approval(config, approval):
    valid = (approval.get('gate') == 'A' and approval.get('decision') == 'approved'
             and approval.get('candidate_id') == 'C2_layout_specific_crop'
             and approval.get('config_sha256') == configuration_hash(config)
             and approval.get('approved_by') == 'user' and bool(approval.get('user_instruction', '').strip())
             and bool(approval.get('recorded_at_utc'))
             and {'crop_geometry', 'prepare_gate_b_intensity_previews'}.issubset(approval.get('permitted_scope', [])))
    if not valid:
        raise ValueError('Matching user crop approval and Gate B preview scope are required.')


def source_luminance(pixels):
    if pixels.ndim == 2:
        return pixels.astype(np.float64)
    return (299 * pixels[..., 0].astype(np.float64) + 587 * pixels[..., 1].astype(np.float64)
            + 114 * pixels[..., 2].astype(np.float64)) / 1000


def intensity_statistics(rgb, gray):
    """Statistics use unpadded nonblack content, not the whole canvas or anatomy.

    The same source-derived mask is used for both arrays. Its threshold is a
    reporting proxy only and never changes saved pixels or chooses a gain.
    """
    active = rgb.max(axis=2) > 8 if rgb.ndim == 3 else rgb > 8
    count = int(active.sum())
    stats = {'status': 'ok' if count else 'needs_review', 'statistic_pixels': count,
             'statistic_fraction_of_crop': count / active.size,
             'statistic_region': 'source max-channel >8 in approved unpadded crop; not anatomical segmentation',
             'optional_transform': 'identity', 'source_luminance_quantiles': None, 'grayscale_quantiles': None,
             'grayscale_mean': None, 'rounding_max_abs_error': None,
             'zero_fraction_unpadded_crop': float((gray == 0).mean())}
    hist = {'source_luminance': [0] * 256, 'grayscale': [0] * 256}
    if count:
        before, after = source_luminance(rgb)[active], gray[active]
        quantiles = [0, .01, .05, .5, .95, .99, 1]
        keys = ['min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max']
        stats.update(source_luminance_quantiles=dict(zip(keys, map(float, np.quantile(before, quantiles)))),
                     grayscale_quantiles=dict(zip(keys, map(float, np.quantile(after, quantiles)))),
                     grayscale_mean=float(after.mean()), source_luminance_mean=float(before.mean()),
                     rounding_max_abs_error=float(np.abs(before - after).max()),
                     zero_fraction_valid_before=float((before <= 0).mean()), zero_fraction_valid_after=float((after == 0).mean()),
                     saturated_fraction_valid_before=float((before >= 255).mean()), saturated_fraction_valid_after=float((after == 255).mean()),
                     optional_operation_new_zero_pixels=0, optional_operation_new_saturated_pixels=0)
        hist['source_luminance'] = np.histogram(before, bins=np.arange(257) - .5)[0].tolist()
        hist['grayscale'] = np.bincount(after, minlength=256).tolist()
    return stats, hist, active


def histogram_plot(histogram, destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3), dpi=130)
    x = np.arange(256)
    ax.plot(x, histogram['source_luminance'], label='Source RGB luminance before rounding', color='#3465a4', linewidth=1)
    ax.plot(x, histogram['grayscale'], label='I0 grayscale', color='#c85820', linewidth=1, linestyle='--')
    ax.set(xlim=(0, 255), xlabel='Pixel intensity (fixed 0–255)', ylabel='Unpadded content pixels')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def comparison_sheet(panels, title, footer):
    width, row_height = 1200, 275
    canvas = Image.new('RGB', (width, 45 + len(panels) * row_height + 45), '#15191f')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill='white')
    for row_index, row in enumerate(panels):
        y = 45 + row_index * row_height
        cell_width = width // len(row)
        for column, (label, pixels) in enumerate(row):
            tile = Image.fromarray(pixels).convert('RGB')
            tile.thumbnail((cell_width - 14, row_height - 35), Image.Resampling.BILINEAR)
            x = column * cell_width
            draw.text((x + 7, y), label, fill='#d2dbe5')
            canvas.paste(tile, (x + (cell_width - tile.width) // 2, y + 25 + (row_height - 35 - tile.height) // 2))
    draw.text((12, canvas.height - 31), footer[:175], fill='#f4ce8e')
    return canvas


def write_intensity_overviews(output, records):
    output = Path(output)
    for page, start in enumerate(range(0, len(records), 6), 1):
        group = records[start:start + 6]
        canvas = Image.new('RGB', (1140, 45 + len(group) * 285), '#15191f')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), 'Approved RGB crop | I0 grayscale, no optional enhancement | 224 padded; fixed 0-255 display', fill='white')
        for index, record in enumerate(group):
            y = 45 + index * 285
            draw.text((12, y), f"{record['patient_id']} / {record['image_id']} / Gate B PENDING", fill='white')
            for column, name in enumerate(['approved_rgb_crop.png', 'I0_preserve_grayscale.png', 'I0_preserve_padded224.png']):
                with Image.open(output / record['image_id'] / name) as source:
                    tile = source.convert('RGB')
                tile.thumbnail((370, 255), Image.Resampling.BILINEAR)
                canvas.paste(tile, (column * 380 + (380 - tile.width) // 2, y + 24 + (255 - tile.height) // 2))
        canvas.save(output / f'intensity_overview_{page}.png')


def build_intensity_review(root, rows, crop_config, approval, intensity_config, gate_a_gallery, modality_reviews, output):
    root, gate_a_gallery, output = Path(root).resolve(), Path(gate_a_gallery).resolve(), Path(output).resolve()
    require_crop_approval(crop_config, approval)
    validate_intensity_config(intensity_config)
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Review output must be separate from source tree.')
    if output.exists():
        raise ValueError('Existing review output is preserved; use a new versioned directory.')
    if not 1 <= len(rows) <= 40 or len({r['patient_id'] for r in rows}) > 28:
        raise ValueError('Gate B is limited to 40 review frames / 28 development patients.')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated review image IDs.')
    if not {r['image_id'] for r in rows}.issubset(approval.get('reviewed_image_ids', [])):
        raise ValueError('Only previously crop-reviewed image IDs are permitted.')
    candidates_path = gate_a_gallery / 'crop_candidates.jsonl'
    if sha256(candidates_path) != approval.get('gate_a_candidates_sha256'):
        raise ValueError('Crop approval snapshot changed; preserve matching geometry evidence.')
    candidates = {r['image_id']: r for r in map(json.loads, candidates_path.read_text().splitlines()) if r['candidate_id'] == 'C2_layout_specific_crop'}
    prepared = []
    for row in rows:
        if row.get('patient_id') is None or row.get('grade') is None:
            raise ValueError('Unresolved mapping blocks intensity review.')
        modality = modality_reviews.get(row['image_id'], {})
        if (modality.get('grayscale_preview_allowed') is not True or modality.get('source_sha256') != row['source_sha256']
                or modality.get('classification') != 'bmode_appearing_no_doppler_seen'):
            raise ValueError('Local modality review is required before converting possible Doppler/color images.')
        source, pixels = load_checked_source(root, row)
        layout = crop_config.get('layouts', {}).get(classify_layout(pixels))
        if not layout or layout.get('privacy_reviewed') is not True or layout.get('selector') != {'width': pixels.shape[1], 'height': pixels.shape[0]}:
            raise ValueError('Source layout or privacy status no longer matches approved geometry.')
        accepted = candidates.get(row['image_id'], {})
        if (accepted.get('box_ltrb') != layout.get('template_box') or accepted.get('privacy_mask_rectangles') != layout.get('mask_rectangles')
                or accepted.get('config_sha256') != configuration_hash(crop_config) or accepted.get('source_sha256') != row['source_sha256']):
            raise ValueError('Recorded crop differs from the approved geometry.')
        processed = preprocess_decoded(pixels, crop_config, intensity_config)
        deidentified, cropped = processed['deidentified'], processed['cropped']
        crop_path = (gate_a_gallery / accepted['crop_path']).resolve()
        if not crop_path.is_relative_to(gate_a_gallery):
            raise ValueError('Approved crop path escapes its gallery.')
        with Image.open(crop_path) as previous:
            if not np.array_equal(cropped, np.asarray(previous)):
                raise ValueError('Crop pixels changed since Gate A; regenerate and reapprove.')
        prepared.append((row, source, cropped, deidentified, layout, accepted, processed))
    output.mkdir(parents=True)
    records, cards = [], []
    intensity_hash = configuration_hash(intensity_config)
    for row, source, cropped, original, layout, accepted, processed in prepared:
        directory = output / row['image_id']
        directory.mkdir()
        gray, tensor, resize = processed['gray'], processed['tensor'], processed['resize']
        stats, histogram, valid_mask = intensity_statistics(cropped, gray)
        h, w = gray.shape
        side = max(1, int(min(w, h) * .25))
        left, top = int(w * .20), int(h * .35)
        zoom_box = [left, top, min(w, left + side), min(h, top + side)]
        rgb_zoom = crop_pixels(cropped, zoom_box)
        gray_zoom = crop_pixels(gray, zoom_box)
        images = {'deidentified_original.png': original, 'approved_rgb_crop.png': cropped,
                  'I0_preserve_grayscale.png': gray, 'I0_preserve_padded224.png': tensor.astype(np.uint8),
                  'source_texture_patch.png': rgb_zoom, 'I0_texture_patch.png': gray_zoom,
                  'statistics_mask.png': valid_mask.astype(np.uint8) * 255}
        for name, pixels in images.items():
            Image.fromarray(pixels).save(directory / name)
        # Nearest neighbor enlarges the same fixed patch without smoothing its texture.
        for name, pixels in [('source_texture_zoom.png', rgb_zoom), ('I0_texture_zoom.png', gray_zoom)]:
            Image.fromarray(pixels).resize((480, 480), Image.Resampling.NEAREST).save(directory / name)
        histogram_plot(histogram, directory / 'histogram.png')
        write_json(directory / 'statistics.json', stats)
        write_json(directory / 'histogram.json', histogram)
        panels = [[('De-identified original', original), ('Approved C2 RGB crop', cropped), ('I0 grayscale; no enhancement', gray), ('224 x 224; aspect preserved', tensor.astype(np.uint8))],
                  [('Source RGB texture patch', rgb_zoom), ('Same patch: I0 grayscale', gray_zoom), ('I0 final 224 x 224', tensor.astype(np.uint8))]]
        title = f"{row['patient_id']} / {row['image_id']} / Gate B PENDING / I0_preserve"
        comparison_sheet(panels, title, 'Approved crop unchanged. Fixed 0-255 display. Patch is a technical display sample, not an organ annotation.').save(directory / 'comparison.png')
        warnings = list(accepted['warnings']) + modality_reviews[row['image_id']].get('review_warnings', [])
        record = {'image_id': row['image_id'], 'patient_id': row['patient_id'], 'candidate_id': 'I0_preserve',
                  'source_sha256': row['source_sha256'], 'approved_crop_sha256': sha256(gate_a_gallery / accepted['crop_path']),
                  'approved_geometry_sha256': configuration_hash(crop_config), 'intensity_config_sha256': intensity_hash,
                  'intensity_config_version': intensity_config['version'], 'box_ltrb': layout['template_box'],
                  'privacy_mask_rectangles': layout['mask_rectangles'], 'crop_size': [w, h], 'output_size': [224, 224],
                  'intensity_operation': 'identity_after_fixed_grayscale', 'grayscale_formula': '(299R+587G+114B+500)//1000',
                  'resize': resize, 'texture_patch_box_relative_to_crop': zoom_box, 'statistics': stats, 'warnings': warnings,
                  'status': 'pending', 'model_input_eligible': False,
                  'tensor_contract': {'shape': [224, 224, 3], 'dtype': 'float32', 'range': [0, 255], 'channels_identical': True},
                  'preview_png_dtype': 'uint8', 'all_source_geometry_retained': True}
        records.append(record)
        image_id = row['image_id']
        figures = ''.join(f'<figure><a href="{image_id}/{name}"><img loading="lazy" src="{image_id}/{name}" alt="{label}"></a><figcaption>{label}</figcaption></figure>' for name, label in [
            ('deidentified_original.png', 'De-identified original'), ('approved_rgb_crop.png', 'Approved C2 crop, source RGB'),
            ('I0_preserve_grayscale.png', 'I0 grayscale, no enhancement'), ('I0_preserve_padded224.png', '224 × 224 padded grayscale')])
        cards.append(f'<article id="{image_id}"><h2>{html.escape(title)}</h2><p>Crop {w} × {h}; box {layout["template_box"]}; no geometry change. <a href="{image_id}/comparison.png">Comparison sheet</a></p><div class="panels">{figures}</div><div class="panels"><figure><img src="{image_id}/source_texture_zoom.png" alt="Source texture"><figcaption>Source RGB patch, enlarged with nearest neighbor</figcaption></figure><figure><img src="{image_id}/I0_texture_zoom.png" alt="Grayscale texture"><figcaption>Identical patch, I0 grayscale</figcaption></figure><figure><a href="{image_id}/histogram.png"><img src="{image_id}/histogram.png" alt="Intensity histogram"></a><figcaption>Same source-derived unpadded content mask; fixed intensity axis</figcaption></figure></div><p>Patch coordinates in crop: {zoom_box}. Foreground proxy fraction {stats["statistic_fraction_of_crop"]:.1%}. <a href="{image_id}/statistics.json">Quantiles and clipping</a> · <a href="{image_id}/statistics_mask.png">Statistics mask</a></p><p class="warning">{html.escape("; ".join(warnings))}</p></article>')
        if sha256(source) != row['source_sha256']:
            raise ValueError('Source changed during review; outputs invalid.')
    css = 'body{font:16px/1.5 system-ui;background:#10151b;color:#edf2f7;margin:24px auto;max-width:1500px;padding:0 16px}a{color:#9ed2ff}article{border-top:2px solid #68788b;margin-top:30px;padding-top:12px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:250px;object-fit:contain;background:black}.warning{color:#f5cd8b}@media(max-width:850px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:480px){.panels{grid-template-columns:1fr}}'
    header = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gate B — intensity review</title><style>{css}</style><h1>Gate B: I0_preserve — awaiting approval</h1><p>The user approved C2 crop proposal 3. This gallery proposes fixed grayscale conversion with no optional gain, gamma, equalization, or denoising. Crop boxes and privacy masks are identical to the approved geometry. Disease grades are hidden. All images retain fixed 0–255 display intensity; click for full-size pixels.</p><p>Technical texture patches and content masks are for review statistics only, not anatomy annotations or training inputs. No full-dataset processing, transfer, job, or training is authorized. P006/P015 duplicate-label conflicts and other flagged cases remain unresolved.</p>'
    (output / 'index.html').write_text(header + ''.join(cards) + '</html>', encoding='utf-8')
    write_json(output / 'intensity.proposed.snapshot.json', intensity_config)
    write_json(output / 'approved_geometry.snapshot.json', crop_config)
    write_json(output / 'viewed_patients.json', sorted({r['patient_id'] for r in rows}))
    (output / 'intensity_candidates.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    write_csv(output / 'review_decisions.csv', [{'image_id': r['image_id'], 'patient_id': r['patient_id'], 'candidate_id': 'I0_preserve',
              'status': 'pending', 'reviewer_notes': '', 'geometry_sha256': r['approved_geometry_sha256'],
              'intensity_config_sha256': intensity_hash} for r in records])
    write_intensity_overviews(output, records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'annotations', 'audit', 'subset', 'crop-config', 'approval', 'intensity-config', 'gate-a-gallery', 'modality-reviews', 'output']:
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    audited, rows, approval = read(args.audit), read(args.subset), read(args.approval)
    if sha256(args.annotations) != approval.get('source_csv_sha256'):
        parser.error('Annotations changed since crop approval; refresh the audit.')
    if sha256(args.audit.parent / 'image_manifest.csv') != approval.get('image_manifest_sha256'):
        parser.error('Image manifest changed since crop approval.')
    indexed = {r['image_id']: r for r in audited['images']}
    reserved_file = args.audit.parent / 'reserved_patients.json'
    reserved = read(reserved_file) if reserved_file.exists() else []
    if audited['summary']['mapping_status'] == 'blocked' or any(indexed.get(r['image_id']) != r or r['patient_id'] in reserved for r in rows):
        parser.error('Subset must match audited records and exclude reserved patients.')
    if sha256(args.subset) != approval.get('review_subset_sha256'):
        parser.error('Review subset changed since crop approval.')
    records = build_intensity_review(args.root, rows, read(args.crop_config), approval, read(args.intensity_config),
                                     args.gate_a_gallery, read(args.modality_reviews), args.output)
    print(f'{len(records)} I0_preserve preview rows. STOP at Gate B: intensity approval is pending.')
    print(args.output / 'index.html')


if __name__ == '__main__':
    main()
