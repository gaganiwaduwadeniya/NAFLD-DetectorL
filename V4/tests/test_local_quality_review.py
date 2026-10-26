"""Gate C sheets preserve color and never infer a quality pass."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from V4.src.local_quality_review import render_quality_sheets
from V4.src.prepare_dataset import load_preparation
from V4.tests.test_prepare_dataset import gate_c, preparation_fixture


def test_sheet_render_requires_gate_c_before_creating_output(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    output = tmp_path / 'sheets'
    with pytest.raises(PermissionError, match='Gate C'):
        render_quality_sheets(context, None, output)
    assert not output.exists()


def test_sheets_cover_only_candidates_preserve_rgb_and_leave_all_qc_pending(tmp_path):
    context = load_preparation(preparation_fixture(tmp_path))
    before = copy.deepcopy(context['plan'])
    output = tmp_path / 'sheets'
    result = render_quality_sheets(context, gate_c(context), output)
    assert result['candidate_frames'] == 3
    assert sum(len(page['images']) for page in result['pages']) == 3
    assert result['grayscale_applied'] is False
    queue = json.loads((output / 'quality_reviews.pending.json').read_text())
    assert len(queue['reviews']) == 4
    assert all(row['privacy_status'] == 'pending' and not row['reviewed_by'] for row in queue['reviews'])
    for page in result['pages']:
        for row in page['images']:
            original = context['indexed'][row['image_id']]
            with Image.open(context['root'] / original['relative_image_path']) as source:
                expected = np.asarray(source)[10:60]
            with Image.open(output / row['crop_path']) as crop:
                assert crop.mode == 'RGB'
                np.testing.assert_array_equal(np.asarray(crop), expected)
    assert context['plan'] == before
    with pytest.raises(ValueError, match='Existing'):
        render_quality_sheets(context, gate_c(context), output)
