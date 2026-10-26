"""Render a bounded, de-identified Gate A gallery. No approved pipeline is created."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .audit_dataset import pixel_digest, sha256, write_csv, write_json
from .preprocess import VERSION, checked_box, classify_layout, crop_pixels, letterbox, mask_pixels, propose_crop, require_preview_dtype


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_checked_source(root, row):
    path = (root / row['relative_image_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Image path escapes source tree.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['source_sha256']:
        raise ValueError('Source hash changed since audit; re-audit before review.')
    with Image.open(io.BytesIO(raw)) as image:
        if image.getexif().get(274, 1) != 1 or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Orientation/cine requires format-specific review.')
        if image.mode not in {'L', 'RGB'}:
            raise ValueError('Source image mode requires explicit format/conversion review; only uint8 L/RGB is supported.')
        pixels = np.asarray(image).copy()
    require_preview_dtype(pixels)
    if pixel_digest(pixels) != row['decoded_pixel_sha256']:
        raise ValueError('Decoded source pixels differ from audit.')
    return path, pixels


def review_sheet(original, candidates, title):
    """Annotations sit outside image cells. Full-size diagnostic panels saved separately."""
    cell_w, cell_h = 300, 235
    sheet = Image.new('RGB', (cell_w * 4, 55 + len(candidates) * (cell_h + 65)), '#15191f')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill='white')
    headings = ['De-identified original', 'Proposed source rectangle / masks', 'Actual crop (unchanged intensity)', '224 x 224, aspect-preserving padding']
    for index, (record, overlay, cropped, padded) in enumerate(candidates):
        y = 55 + index * (cell_h + 65)
        for col, pixels in enumerate([original, overlay, cropped, padded]):
            tile = Image.fromarray(pixels).convert('RGB')
            tile.thumbnail((cell_w - 12, cell_h - 24), Image.Resampling.BILINEAR)
            x = col * cell_w
            draw.text((x + 6, y), headings[col], fill='#c9d3dd')
            sheet.paste(tile, (x + (cell_w - tile.width) // 2, y + 24 + (cell_h - 24 - tile.height) // 2))
        draw.text((12, y + cell_h + 4), f"{record['candidate_id']} | box {record['box_ltrb']} | crop {record['crop_size']} | retained {record['retained_area_fraction']:.1%}", fill='white')
        draw.text((12, y + cell_h + 23), 'Pending human crop review; ' + '; '.join(record['warnings'])[:155], fill='#f4ce8e')
    return sheet


def write_overviews(output, rows):
    for page, start in enumerate(range(0, len(rows), 6), 1):
        group = rows[start:start + 6]
        canvas = Image.new('RGB', (1140, len(group) * 285 + 40), '#15191f')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'Actual images: de-identified original | C2 display proposal | 224 padded preview', fill='white')
        for i, row in enumerate(group):
            y = 40 + i * 285
            draw.text((12, y), f"{row['patient_id']} / {row['image_id']} / PENDING", fill='white')
            for col, name in enumerate(['deidentified_original.png', 'C2_layout_specific_crop_crop.png', 'C2_layout_specific_crop_padded.png']):
                path = output / row['image_id'] / name
                if not path.exists():
                    continue
                with Image.open(path) as source:
                    tile = source.convert('RGB')
                tile.thumbnail((370, 255), Image.Resampling.BILINEAR)
                canvas.paste(tile, (col * 380 + (380 - tile.width) // 2, y + 24 + (255 - tile.height) // 2))
        canvas.save(output / f'overview_{page}.png')


def build_review(root, rows, config, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Preview output must be separate from the source tree.')
    if output.exists():
        raise ValueError('Use a new preview directory; previous reviews are preserved.')
    if len(rows) > 40 or len({r['patient_id'] for r in rows}) > 28:
        raise ValueError('Gate A renders at most 40 images / 28 development patients, not a full-dataset derivative.')
    if not rows:
        raise ValueError('No review images selected.')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated review image IDs.')
    if any(r['patient_id'] is None or r.get('grade') is None for r in rows):
        raise ValueError('Unresolved image mapping blocks gallery generation.')
    # Validate every source/layout before any preview is written.
    prepared = []
    for row in rows:
        source, pixels = load_checked_source(root, row)
        layout_key = classify_layout(pixels)
        layout = config.get('layouts', {}).get(layout_key)
        if not layout or layout.get('privacy_reviewed') is not True:
            raise ValueError(f'Unreviewed privacy layout {layout_key}; no gallery generated.')
        selector = layout.get('selector', {})
        if selector != {'width': pixels.shape[1], 'height': pixels.shape[0]}:
            raise ValueError('Layout selector must match observable dimensions exactly.')
        rectangles = layout.get('mask_rectangles', [])
        if not rectangles:
            raise ValueError('Explicit privacy masks are required for screenshot review.')
        for box in rectangles:
            checked_box(box, pixels.shape[1], pixels.shape[0])
        if layout.get('template_box'):
            checked_box(layout['template_box'], pixels.shape[1], pixels.shape[0])
        prepared.append((row, source, pixels, layout))

    config_sha = configuration_hash(config)
    records, cards = [], []
    output.mkdir(parents=True)
    for row, source, pixels, layout in prepared:
        directory = output / row['image_id']
        directory.mkdir()
        deidentified = mask_pixels(pixels, layout['mask_rectangles'])
        Image.fromarray(deidentified).save(directory / 'deidentified_original.png')
        w, h = pixels.shape[1], pixels.shape[0]
        box, warnings = propose_crop(deidentified, config.get('black_threshold', 8), config.get('margin_px', 8))
        proposals = [('C0_full_field', [0, 0, w, h], ['full_screen_ui_retained']),
                     ('C1_conservative_field_crop', box, warnings + ['nonblack_detector_retains_external_ui'])]
        if layout.get('template_box'):
            proposals.append(('C2_layout_specific_crop', layout['template_box'], ['display_geometry_template_not_anatomy']))
        sheet_candidates, variant_html = [], []
        for candidate_id, box, warnings in proposals:
            warnings = warnings + layout.get('warnings', []) + row.get('quality_flags', [])
            cropped = crop_pixels(deidentified, box)
            padded, resize = letterbox(cropped, config.get('output_size', 224))
            if resize['padding_fraction'] > .30:
                warnings.append('more_than_30_percent_padding')
            overlay = Image.fromarray(deidentified).convert('RGB')
            draw = ImageDraw.Draw(overlay)
            l, t, r, b = box
            draw.rectangle((l, t, r - 1, b - 1), outline='#46ec90', width=3)
            for ml, mt, mr, mb in layout['mask_rectangles']:
                draw.rectangle((ml, mt, mr - 1, mb - 1), outline='#c286ff', width=2)
            paths = {}
            for kind, image in [('overlay', np.asarray(overlay)), ('crop', cropped), ('padded', padded)]:
                rel = f"{row['image_id']}/{candidate_id}_{kind}.png"
                Image.fromarray(image).save(output / rel)
                paths[f'{kind}_path'] = rel
            record = dict(image_id=row['image_id'], patient_id=row['patient_id'], candidate_id=candidate_id,
                          source_size=[w, h], box_ltrb=box, coordinate_convention='left/top inclusive; right/bottom exclusive',
                          crop_size=[r - l, b - t], output_size=[224, 224],
                          retained_area_fraction=(r - l) * (b - t) / (w * h),
                          privacy_mask_rectangles=layout['mask_rectangles'], privacy_status='header_masked_local_visual_review_required',
                          algorithm_version=VERSION, config_version=config['version'], config_sha256=config_sha,
                          source_sha256=row['source_sha256'], decoded_pixel_sha256=row['decoded_pixel_sha256'],
                          warnings=warnings, status='pending', model_input_eligible=False,
                          intensity='source_uint8_RGB_or_L_unchanged_no_enhancement', resize=resize, **paths)
            records.append(record)
            sheet_candidates.append((record, np.asarray(overlay), cropped, padded))
            panels = [(f"{row['image_id']}/deidentified_original.png", 'De-identified original'),
                      (paths['overlay_path'], 'Rectangle / masks'), (paths['crop_path'], 'Actual crop'), (paths['padded_path'], '224 × 224 padded')]
            figures = ''.join(f'<figure><a href="{html.escape(p)}"><img loading="lazy" src="{html.escape(p)}" alt="{label}"></a><figcaption>{label}</figcaption></figure>' for p, label in panels)
            variant_html.append(f'<section class="candidate"><h3>{candidate_id}</h3><p>Source {w} × {h}; crop {r-l} × {b-t}; box {box}; retained {record["retained_area_fraction"]:.1%}; padding {resize["padding_fraction"]:.1%}.</p><div class="panels">{figures}</div><p class="warning">{html.escape("; ".join(warnings))}</p><label>Local review note <select data-image="{row["image_id"]}" data-patient="{row["patient_id"]}" data-candidate="{candidate_id}"><option>pending</option><option>approved</option><option>needs_adjustment</option><option>rejected</option></select><input aria-label="Reviewer notes" placeholder="Crop / text / tissue concerns"></label></section>')
        title = f"{row['patient_id']} | {row['image_id']} | source {w} x {h} | Gate A PENDING"
        review_sheet(deidentified, sheet_candidates, title).save(directory / 'comparison.png')
        cards.append(f'<article id="{row["image_id"]}"><h2>{html.escape(title)}</h2><p>Green: proposed crop. Violet: privacy masks. Click any panel for its full pixel size. <a href="{row["image_id"]}/comparison.png">Comparison sheet</a></p>{"".join(variant_html)}</article>')
        if sha256(source) != row['source_sha256']:
            raise ValueError('Source changed during preview generation; outputs invalid.')
    header = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NAFLD V4 — Gate A crop review</title>
<style>body{font:16px/1.5 system-ui;background:#10151b;color:#edf2f7;margin:24px auto;max-width:1500px;padding:0 16px}a{color:#9ed2ff}article{border-top:2px solid #667587;margin-top:32px}section{background:#1b232e;padding:16px;margin:16px 0;border-radius:8px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:250px;object-fit:contain;background:black}figcaption{font-size:14px}.warning{color:#f5cd8b}select,input,button{font:inherit;padding:6px;margin:4px;max-width:100%}input{width:45%}nav{position:sticky;top:0;background:#18222e;padding:12px;z-index:2}@media(max-width:850px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:480px){.panels{grid-template-columns:1fr}}</style>
<h1>Gate A: crop review — pending approval</h1><p>Local de-identified preview derivatives only. No contrast change, grayscale conversion, training, remote transfer, or full-dataset processing. Labels stay outside pixels and the gallery hides disease grades. Source anatomy and orientation are unchanged except the displayed masks and candidate rectangle. Check near-field and deep tissue, diaphragm/kidney where present, both views, residual text, and padding.</p><p>Preview settings are proposals. Local row decisions do not approve Gate A or authorize later stages. Keep this gallery local until its de-identification has been reviewed.</p><nav><button id="export">Export row decisions CSV</button> <span id="counts"></span></nav>'''
    script = '''<script>
const configHash=CONFIG_HASH;
document.getElementById('counts').textContent=document.querySelectorAll('select').length+' candidate rows; Gate A remains pending';
document.getElementById('export').addEventListener('click',()=>{
const quote=s=>'"'+String(s).replaceAll('"','""')+'"';
const rows=[['image_id','patient_id','candidate_id','status','reviewer_notes','config_sha256']];
document.querySelectorAll('select').forEach(s=>rows.push([s.dataset.image,s.dataset.patient,s.dataset.candidate,s.value,s.nextElementSibling.value,configHash]));
const blob=new Blob([rows.map(r=>r.map(quote).join(',')).join('\\r\\n')],{type:'text/csv'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review_decisions.csv';a.click();URL.revokeObjectURL(a.href);
});</script></html>'''.replace('CONFIG_HASH', json.dumps(config_sha))
    (output / 'index.html').write_text(header + ''.join(cards) + script, encoding='utf-8')
    (output / 'crop_candidates.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records), encoding='utf-8')
    write_json(output / 'preprocessing.proposed.snapshot.json', config)
    write_json(output / 'viewed_patients.json', sorted({r['patient_id'] for r in rows}))
    write_csv(output / 'review_decisions.csv', [dict(image_id=r['image_id'], patient_id=r['patient_id'], candidate_id=r['candidate_id'],
              status='pending', reviewer_notes='', config_sha256=config_sha) for r in records])
    write_overviews(output, rows)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--subset', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audited = json.loads(args.audit.read_text())
    if audited['summary']['mapping_status'] == 'blocked':
        parser.error('Resolve image mapping/decode issues before gallery generation.')
    rows = json.loads(args.subset.read_text())
    indexed = {r['image_id']: r for r in audited['images']}
    reserved_path = args.audit.parent / 'reserved_patients.json'
    reserved = json.loads(reserved_path.read_text()) if reserved_path.exists() else []
    if any(r['patient_id'] in reserved or indexed.get(r['image_id']) != r for r in rows):
        parser.error('Subset must match audited records and exclude reserved patients.')
    config = json.loads(args.config.read_text())
    records = build_review(args.root, rows, config, args.output)
    print(f'{len(rows)} images / {len(set(r["patient_id"] for r in rows))} patients; {len(records)} pending crop candidates.')
    print(f'Local gallery: {args.output / "index.html"}')
    print('STOP: Gate A requires the user\'s crop decision. No subsequent stages are implemented here.')


if __name__ == '__main__':
    main()
