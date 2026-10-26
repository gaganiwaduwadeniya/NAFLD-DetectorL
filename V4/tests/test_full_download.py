"""New download boundary: safe extraction, hashes and actual embedded figures."""
import json
import zipfile

import nbformat
import pytest
from PIL import Image

from V4.src.full_training import export_archive
from V4.kaggle.monitor_full_training import extract_verified, embed_results


def test_download_archive_roundtrip_hashes_and_embedded_results_notebook(tmp_path):
    source=tmp_path/'run';source.mkdir()
    (source/'run_report.json').write_text(json.dumps(dict(status='incomplete_budget')))
    (source/'best.keras').write_bytes(b'saved model fixture')
    Image.new('RGB',(8,8),'blue').save(source/'actual.png')
    archive=export_archive(source)
    destination=tmp_path/'download';destination.mkdir()
    root=extract_verified(archive,destination)
    assert (root/'best.keras').read_bytes()==b'saved model fixture'
    embed_results(root,destination)
    notebook=nbformat.read(destination/'downloaded_results.ipynb',as_version=4)
    nbformat.validate(notebook)
    assert sum('image/png' in o.get('data',{}) for c in notebook.cells if c.cell_type=='code' for o in c.outputs)==1
    assert notebook.metadata.nafld_v4.locally_executed is False


def test_download_rejects_archive_traversal_before_writing_any_member(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('full_cv_results/ok.txt','should not be written')
        z.writestr('full_cv_results/../../outside.txt','bad')
    destination=tmp_path/'download'
    with pytest.raises(ValueError,match='Unsafe'):
        extract_verified(archive,destination)
    assert not destination.exists() and not (tmp_path/'outside.txt').exists()
