"""Gate C local RGB contact sheets; rendering never grants a QC pass.

Native crops remain available for inspecting small overlays. These are local
review materials, not privacy-cleared exports or grayscale model inputs.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .audit_dataset import sha256, write_json
from .build_review import load_checked_source
from .prepare_dataset import build_quality_queue, require_preparation_approval
from .preprocess import classify_layout, crop_pixels, mask_pixels


def render_quality_sheets(context, approval, output):
    require_preparation_approval(context, approval)
    output = Path(output).resolve()
    root = context['root']
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Review output must be separate from source images.')
    if output.exists():
        raise ValueError('Existing review outputs are preserved; use a new directory.')
    output.mkdir(parents=True)
    (output / 'crops').mkdir()
    queue = build_quality_queue(context)
    candidates = [context['indexed'][row['image_id']] for row in context['plan']
                  if row['proposed_action'] == 'candidate_for_preprocessing']
    candidates.sort(key=lambda row: (row['patient_id'], row['relative_image_path']))
    result = {'schema_version': 1, 'bindings': context['bindings'],
              'candidate_frames': len(candidates), 'grayscale_applied': False,
              'review_status': 'Pending visual inspection; no QC pass inferred.',
              'native_crops': 'Lossless RGB approved masks/crop only; inspect for residual text before passing.',
              'pages': [], 'render_failures': []}
    links = []
    # 30 frames per sheet, each 320 wide; full source-resolution crop linked below.
    for page_index, start in enumerate(range(0, len(candidates), 30), 1):
        group = candidates[start:start + 30]
        sheet = Image.new('RGB', (1600, 36 + math.ceil(len(group) / 5) * 276), '#15191f')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'Local Gate C RGB review / page {page_index:03d} / native crops linked in index / NOT privacy cleared', fill='white')
        page = {'page': page_index, 'sheet_path': f'sheet_{page_index:03d}.png', 'images': []}
        for index, row in enumerate(group):
            x, y = (index % 5) * 320, 36 + (index // 5) * 276
            draw.text((x + 4, y), f"{start+index+1:04d} {row['patient_id']} {row['image_id'][-8:]}", fill='white')
            try:
                _, pixels = load_checked_source(root, row)
                if classify_layout(pixels) != row['layout']:
                    raise ValueError('Source layout differs from audited layout.')
                layout = context['crop']['layouts'][row['layout']]
                cropped = crop_pixels(mask_pixels(pixels, layout['mask_rectangles']), layout['template_box'])
                crop_path = f"crops/{row['image_id']}.png"
                Image.fromarray(cropped).save(output / crop_path)
                tile = Image.fromarray(cropped).convert('RGB')
                tile.thumbnail((316, 250), Image.Resampling.BILINEAR)
                sheet.paste(tile, (x + (320 - tile.width)//2, y + 20 + (250-tile.height)//2))
                page['images'].append({'image_id': row['image_id'], 'ordinal': start+index+1,
                                       'crop_path': crop_path, 'crop_sha256': sha256(output / crop_path),
                                       'source_sha256': row['source_sha256']})
                links.append(f'<a href="{crop_path}">{start+index+1:04d} {html.escape(row["patient_id"])} {row["image_id"]}</a>')
            except (OSError, ValueError, KeyError) as error:
                draw.text((x + 8, y + 50), 'RENDER FAILURE - HOLD', fill='#ffb080')
                result['render_failures'].append({'image_id': row['image_id'], 'reason': str(error)})
        sheet.save(output / page['sheet_path'])
        page['sheet_sha256'] = sha256(output / page['sheet_path'])
        result['pages'].append(page)
        print(f'Rendered local review page {page_index}', flush=True)
    write_json(output / 'quality_reviews.pending.json', queue)
    write_json(output / 'sheet_manifest.json', result)
    page_links = ''.join(f'<p><a href="{p["sheet_path"]}">Sheet {p["page"]:03d}</a></p>' for p in result['pages'])
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Gate C local technical review</title>'
                                     '<h1>Local technical review</h1><p>RGB masks/crop only. Pending review; no clinical grading or export clearance.</p>'
                                     + page_links + '<h2>Native crops</h2>' + '<br>'.join(links), encoding='utf-8')
    return result
