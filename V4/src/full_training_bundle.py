"""Build the local full-CV allowlist and runnable notebook; never upload or fit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from .prepare_dataset import load_preparation
from .training_bundle import load_prepared_cohort, _safe_file
from .training_plan import validate_splits
from .full_training import FullFrameStore, full_config, save_json, digest, config_hash

DATASET='bimalawijekoon/nafld-v4-cv-inputs-20261020'
NOTEBOOK='bimalawijekoon/nafld-v4-five-fold-20261020'
MODULES=['training_data.py','training_runtime.py','training_visuals.py',
         'full_training.py','full_training_visuals.py','full_gradcam.py']


def notebook_source(manifest_hash):
    bootstrap='''from pathlib import Path, PurePosixPath
import hashlib, json, os, stat, sys, zipfile
sys.dont_write_bytecode = True
expected_manifest = MANIFEST_HASH
def sha(path):
    with Path(path).open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()
matches = [p.parent for p in Path('/kaggle/input').rglob('training_manifest.json')
           if sha(p) == expected_manifest]
if not matches:
    archives = list(Path('/kaggle/input').rglob('inputs_bundle.zip'))
    assert len(archives) == 1, 'Require the reviewed private full-CV input archive.'
    destination = Path('/kaggle/working/verified_full_input')
    assert not destination.exists(), 'Preserve an existing extraction.'
    with zipfile.ZipFile(archives[0]) as archive:
        manifest_bytes = archive.read('full_inputs/training_manifest.json')
        assert hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest
        packed_manifest = json.loads(manifest_bytes)
        allow = {'training_manifest.json', 'gate_e_full_approval.json'}
        allow.update(r['file'] for r in packed_manifest['frames'])
        allow.update(a['file'] for a in packed_manifest['assets'])
        assert len(archive.infolist()) == len(allow)
        assert sum(m.file_size for m in archive.infolist()) < 250_000_000
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            assert len(name.parts) == 2 and name.parts[0] == 'full_inputs' and name.parts[1] in allow
            assert not stat.S_ISLNK(item.external_attr >> 16)
        archive.extractall(destination)
    matches = [destination/'full_inputs']
assert len(matches) == 1, 'Require one exact manifest mount.'
root = matches[0]
manifest = json.loads((root/'training_manifest.json').read_text())
for item in manifest['assets']:
    assert Path(item['file']).name == item['file'] and sha(root/item['file']) == item['sha256']
sys.path.insert(0, str(root))
from full_training import FullFrameStore, require_full_approval, run_experiment, export_archive
from full_training_visuals import plot_inputs, plot_fold, plot_summary, write_gallery
config = json.loads((root/'experiment_config.json').read_text())
approval = json.loads((root/'gate_e_full_approval.json').read_text())
require_full_approval(approval, config, expected_manifest)
store = FullFrameStore(root, expected_manifest)
print(f'Verified {len(store.rows)} prepared image hashes; {len(store.groups)} opaque patient groups.')
print(json.dumps(config, indent=2))
'''.replace('MANIFEST_HASH',repr(manifest_hash))
    training='''from IPython.display import FileLink, Markdown, display
output = Path('/kaggle/working/full_cv_results')
assert not output.exists(), 'Require a fresh versioned run.'
try:
    result = run_experiment(root, expected_manifest, config, approval, output, on_fold=plot_fold)
    plot_summary(store, output, result)
finally:
    if output.exists():
        # Input figures were generated before fitting; include them in the export.
        import shutil
        shutil.copytree('/kaggle/working/full_cv_input_plots', output/'input_plots', dirs_exist_ok=True)
        write_gallery(output)
        archive = export_archive(output)
        display(FileLink(str(archive)))
        print(f'Exported {archive.name}: {archive.stat().st_size:,} bytes')
assert result['status'] == 'completed', 'Partial run: retain artifacts; do not report full-CV completion.'
display(Markdown('**Actual patient out-of-fold results — internal validation**'))
print(json.dumps({k: result[k] for k in ['patient_metrics', 'frame_metrics',
    'majority_baseline_patient_metrics', 'patient_bootstrap_95']}, indent=2))
'''
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,id=f'cell-{len(cells)}',metadata={},source=source.splitlines(keepends=True))
        if kind=='code': cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown', '# V4 five-fold EfficientNetB0 experiment\n\n'
        'Actual full-experiment approval is required before fitting. Five independent ImageNet initializations; '
        '109 patients / 1,640 approved prepared frames. Classes: 0 Normal, 1 Mild, 2 Moderate, 3 Severe. '
        'Private, GPU on, Internet/TPU off, two-hour hard job cap.\n\n'
        'All plots below are generated from actual inputs or observed outputs and saved as PNG/SVG. '
        'The local monitor downloads the artifact archive and logs after completion. '
        'All groups had technical-QC exposure and the pilot supplied development feedback; results are internal CV.')
    add('markdown','## 1. Exact input, runtime and permission checks\n\nNo TensorFlow import or model fitting occurs before these checks pass.')
    add('code',bootstrap)
    add('markdown','## 2. Actual inputs, splits, weights and recipe\n\nMetadata class counts include outer folds; examples shown here use fold-0 train/inner validation only. No extra image transforms.')
    add('code',"plot_inputs(store, Path('/kaggle/working/full_cv_input_plots'), config)")
    add('markdown','## 3. Train five independent folds, then score their excluded patients\n\n'
        'Every fold displays its actual training curves, inner/outer confusion matrices, class metrics, '
        'probabilities, ROC/PR, patient error examples and fixed Grad-CAM panel. '
        'Inner validation alone selects checkpoints. Outer scoring begins after the selected checkpoint is locked. '
        'The combined out-of-fold summary and patient uncertainty plot follow the fifth fold. '
        'Heatmaps are descriptive and do not establish organ localization or clinical validity.')
    add('code',training)
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)


def build_bundle(workspace, output):
    workspace,output=Path(workspace).resolve(),Path(output).resolve()
    if output.exists(): raise ValueError('Preserve existing staging; select a new bundle version.')
    context=load_preparation(workspace)
    prepared=workspace/'artifacts/prepared_v1'
    rows=load_prepared_cohort(context,prepared)
    split_path=workspace/'manifests/gate_e_splits_v1/patient_splits.json'
    splits=json.loads(split_path.read_text()); validate_splits(rows,splits)
    if len(rows)!=1640 or len({r['patient_id'] for r in rows})!=109:
        raise ValueError('Full prepared cohort differs from the reviewed scope.')
    weights=workspace/'artifacts/pretrained/efficientnetb0_notop.h5'
    if digest(weights)!='0683edcf6151811fb3ddd6f098ac48df644d74592c4919a5acdee1c37554add2':
        raise ValueError('Official pretrained weights hash differs.')
    payload=output/'payload'; dataset=output/'dataset'; kernel=output/'kernel'
    for p in (payload,dataset,kernel): p.mkdir(parents=True)
    patients=sorted({r['patient_id'] for r in rows},key=lambda p:hashlib.sha256(('full-cv-group|'+p).encode()).hexdigest())
    groups={p:f'G{i:04d}' for i,p in enumerate(patients,1)}
    frames=[]; mapping=[]
    for i,r in enumerate(sorted(rows,key=lambda r:r['image_id']),1):
        source=_safe_file(prepared,r['image_path']); frame_id=f'F{i:06d}'
        if digest(source)!=r['image_file_sha256']: raise ValueError('Prepared PNG changed.')
        target=payload/(frame_id+'.png'); shutil.copyfile(source,target)
        if digest(target)!=r['image_file_sha256']: raise ValueError('Staged PNG differs from approved bytes.')
        frames.append(dict(frame_id=frame_id,group_id=groups[r['patient_id']],grade=r['grade'],file=target.name,sha256=r['image_file_sha256']))
        mapping.append(dict(frame_id=frame_id,group_id=groups[r['patient_id']],patient_id=r['patient_id'],image_id=r['image_id']))
        if i%400==0: print(f'Staged {i}/1640 approved PNGs with matching hashes.',flush=True)
    folds=[dict(fold=f['fold'],**{role:[groups[p] for p in f[role]] for role in ('train','validation','test')}) for f in splits['folds']]
    config=full_config(digest(weights)); save_json(payload/'experiment_config.json',config)
    shutil.copyfile(weights,payload/weights.name)
    for name in MODULES: shutil.copyfile(Path(__file__).with_name(name),payload/name)
    assets=[dict(file=n,sha256=digest(payload/n)) for n in [weights.name,'experiment_config.json',*MODULES]]
    bindings={k:v for k,v in context['bindings'].items() if k.endswith('sha256')}
    bindings.update(prepared_manifest_sha256=digest(prepared/'output_manifest.json'),patient_splits_sha256=digest(split_path))
    manifest=dict(schema_version=2,purpose='five_fold_internal_cv',frames=frames,folds=folds,assets=assets,
                  source_bindings=bindings,pixel_policy='Unchanged approved C2/I0 grayscale PNG → replicate channels → float32 0–255.',
                  evaluation_scope='Internal CV; all patient groups exposed to technical QC; fold-0 pilot feedback disclosed.')
    save_json(payload/'training_manifest.json',manifest); manifest_hash=digest(payload/'training_manifest.json')
    save_json(payload/'gate_e_full_approval.json',dict(gate='E',decision='pending_user_approval',
        note='Local tools go is not a full-cohort transfer/training approval. Bind actual approval to this exact manifest/config.'))
    save_json(output/'local_mapping.DO_NOT_UPLOAD.json',mapping)
    save_json(dataset/'dataset-metadata.json',dict(id=DATASET,title='NAFLD V4 CV Inputs 20261020',licenses=[dict(name='other')],
        description='Restricted private research. Rights remain with original authors. No public redistribution license. De-identified approved prepared inputs only.'))
    save_json(kernel/'five_fold_training.ipynb',notebook_source(manifest_hash))
    save_json(kernel/'kernel-metadata.json',dict(id=NOTEBOOK,title='NAFLD V4 Five Fold 20261020',
        code_file='five_fold_training.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=False,dataset_sources=[DATASET],
        kernel_sources=[],competition_sources=[],model_sources=[]))
    store=FullFrameStore(payload,manifest_hash)
    with zipfile.ZipFile(dataset/'inputs_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.iterdir()): z.write(p,'full_inputs/'+p.name)
    with zipfile.ZipFile(dataset/'inputs_bundle.zip') as z:
        if len(z.infolist())!=len(list(payload.iterdir())): raise ValueError('Archive member count differs.')
        for p in payload.iterdir():
            if hashlib.sha256(z.read('full_inputs/'+p.name)).hexdigest()!=digest(p):
                raise ValueError('Expanded archive bytes differ from reviewed payload.')
    logical=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(payload.iterdir())]
    transfer=[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=digest(p))
              for d in (dataset,kernel) for p in sorted(d.iterdir())]
    save_json(output/'logical_payload_inventory.json',logical)
    save_json(output/'transfer_inventory.json',transfer)
    proposal=dict(schema_version=1,gate='E',decision='pending_user_approval',dataset=DATASET,notebook=NOTEBOOK,
        frames=len(frames),patient_groups=len(groups),held_frames_excluded=29,
        manifest_sha256=manifest_hash,experiment_config_sha256=config_hash(config),
        patient_splits_sha256=digest(split_path),maximum_remote_runs=1,timeout_seconds=7200,
        fold_training_wall_seconds=1200,total_training_wall_seconds=6000,
        epoch_caps=dict(head=10,fine_tune=30,early_stopping_patience=5),
        is_private=True,internet_enabled=False,single_visible_gpu=True,
        full_cohort_private_upload_approved=False,training_approved=False,
        actual_approval_file='payload/gate_e_full_approval.json',
        approval_completion_policy='Only the actual full approval JSON and containing transport ZIP are completed in a separate execution copy after real approval; all other logical bytes remain unchanged.',
        transfer_inventory=transfer,total_transfer_bytes=sum(r['bytes'] for r in transfer),
        logical_payload_files=len(logical),logical_payload_bytes=sum(r['bytes'] for r in logical),
        automatic_download_script='V4/kaggle/monitor_full_training.py',
        exclusions=['raw scans','source annotation CSV','original patient/image IDs','local mapping','held frames','V3 models','application','credentials'],
        acceptance=['Five completed independently initialized folds','Checkpoint selection uses inner validation only',
                    '109 patient and 1640 frame OOF predictions, each in exactly its excluded outer fold',
                    'Actual plots, histories, checkpoint hashes, metrics and logs exported and downloaded',
                    'No automatic retry of an accepted or ambiguous full job'])
    save_json(output/'transfer_proposal.json',proposal)
    save_json(output/'local_validation.json',dict(status='passed',all_staged_png_hashes_match=1640,
        archive_members_verified=len(logical),patients=len(store.groups),frozen_splits_unchanged=True,
        transfer_paths=[r['path'] for r in transfer],training_executed=False,upload_performed=False,
        pixel_equivalence_basis='Existing Gate C and pilot loader proof reused; no repeat tensor reconstruction suite.'))
    return proposal


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build_bundle(args.workspace,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='transfer_inventory'},indent=2))
