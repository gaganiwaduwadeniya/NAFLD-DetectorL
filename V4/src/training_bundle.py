"""Create local Gate E split/review artifacts; no credentials, upload or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .audit_dataset import sha256, write_csv, write_json
from .prepare_dataset import load_preparation, require_preparation_approval, _quality_reasons
from .training_plan import make_splits, validate_splits
from .training_data import FrameStore, patient_weights


DATASET_SLUG = 'nafld-v4-training-inputs-20261019'
NOTEBOOK_SLUG = 'nafld-v4-e0-gpu-pilot-20261019'


def load_prepared_cohort(context, prepared):
    prepared = Path(prepared).resolve()
    require_preparation_approval(context, json.loads((context['workspace']/'configs/preparation_approval.json').read_text()))
    status = json.loads((prepared/'run_status.json').read_text())
    summary = json.loads((prepared/'preparation_summary.json').read_text())
    if (status.get('status') != 'complete' or status.get('output_manifest_sha256') != sha256(prepared/'output_manifest.json')
            or summary.get('bindings') != context['bindings'] or summary.get('mode') != 'approved_local_preparation'):
        raise ValueError('A complete unchanged Gate C derivative is required.')
    records = json.loads((prepared/'output_manifest.json').read_text())
    rows = [r for r in records if r['status'] == 'prepared']
    if len(rows) != summary['prepared_frames'] or len({r['patient_id'] for r in rows}) != summary['prepared_patients']:
        raise ValueError('Prepared cohort counts disagree with Gate C.')
    candidates = {r['image_id'] for r in context['plan'] if r['proposed_action'] == 'candidate_for_preprocessing'}
    for row in rows:
        source = context['indexed'].get(row['image_id'])
        if (not source or row['image_id'] not in candidates or row['bindings'] != context['bindings']
                or row['patient_id'] != source['patient_id'] or row['grade'] != source['grade']
                or row['source_sha256'] != source['source_sha256']
                or _quality_reasons(context, source, row.get('quality_review'))):
            raise ValueError('A held, misassigned or uncleared frame entered the candidate cohort.')
    return rows


def _safe_file(prepared, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
        raise ValueError('Unsafe prepared file path.')
    path = prepared/relative
    if path.is_symlink() or not path.resolve().is_relative_to(prepared) or not path.is_file():
        raise ValueError('Prepared path must be a regular file inside the derivative.')
    return path


def stage_training_bundle(rows, plan, prepared, weights, output, bindings=None, owner='bimalawijekoon'):
    prepared, weights, output = Path(prepared).resolve(), Path(weights).resolve(), Path(output).resolve()
    validate_splits(rows, plan)
    if (any(r.get('status') != 'prepared' for r in rows) or output.exists()
            or output.is_relative_to(prepared) or prepared.is_relative_to(output)):
        raise ValueError('Only prepared rows and a new separate output directory are permitted.')
    if hashlib.md5(weights.read_bytes()).hexdigest() != '50bc09e76180e00e4465e1a485ddc09d':
        raise ValueError('Require the official checksum-verified ImageNet B0 no-top weights.')
    fold = plan['folds'][0]
    roles = {p: role for role in ['train', 'validation'] for p in fold[role]}
    selected = [r for r in sorted(rows, key=lambda r: r['image_id']) if r['patient_id'] in roles]
    patients = sorted(roles, key=lambda p: hashlib.sha256(('gate-e-group|'+p).encode()).hexdigest())
    group_ids = {p: f'G{i:04d}' for i,p in enumerate(patients, 1)}
    sources = [(r, _safe_file(prepared, r['image_path']), _safe_file(prepared, r['tensor_path'])) for r in selected]
    dataset, kernel = output/'dataset', output/'kernel'; dataset.mkdir(parents=True); kernel.mkdir()
    frames, local_mapping = [], []
    for index, (row, image_path, tensor_path) in enumerate(sources, 1):
        if sha256(image_path) != row['image_file_sha256'] or sha256(tensor_path) != row['tensor_file_sha256']:
            raise ValueError('Prepared image/tensor file hash changed.')
        tensor = np.load(tensor_path, allow_pickle=False)
        with Image.open(image_path) as image:
            if image.mode != 'L' or image.size != (224,224) or image.getexif() or image.info:
                raise ValueError('Expected a plain lossless prepared grayscale PNG.')
            pixels = np.repeat(np.asarray(image)[...,None], 3, axis=-1).astype(np.float32)
        if tensor.dtype != np.float32 or tensor.shape != (224,224,3) or not np.array_equal(pixels, tensor):
            raise ValueError('PNG loader does not reproduce the approved canonical tensor exactly.')
        frame_id = f'F{index:06d}'; name = frame_id+'.png'; shutil.copyfile(image_path, dataset/name)
        frames.append({'frame_id':frame_id,'group_id':group_ids[row['patient_id']], 'grade':row['grade'],
                       'role':roles[row['patient_id']],'file':name,'sha256':sha256(dataset/name)})
        local_mapping.append({'frame_id':frame_id,'group_id':group_ids[row['patient_id']],
                              'image_id':row['image_id'],'patient_id':row['patient_id']})
        if index % 200 == 0: print(f'Verified and staged {index}/{len(sources)} lossless pilot frames.', flush=True)
    shutil.copyfile(weights, dataset/'efficientnetb0_notop.h5')
    config = {'schema_version':1,'purpose':'fold_0_technical_pilot','fold':0,'seed':42,
              'architecture':'ImageNet_EfficientNetB0_GAP_Dropout_Dense4_softmax',
              'input_shape':[224,224,3],'input_range':[0,255],'batch_size':16,'dropout':.35,
              'head_epochs':5,'fine_tune_epochs':5,'head_learning_rate':.001,'fine_tune_learning_rate':.00001,
              'fine_tune_blocks':['block6','block7','top_'],'batchnorm':'frozen_inference_mode',
              'augmentation':'none','precision':'float32','patient_and_class_weighting':'training_patient_counts_only',
              'outer_test_evaluation':False,'training_wall_seconds':1500,'job_timeout_seconds':1800,
              'pretrained_weights_file':'efficientnetb0_notop.h5','pretrained_weights_sha256':sha256(weights)}
    write_json(dataset/'experiment_config.json', config)
    for name in ['training_data.py','training_runtime.py']: shutil.copyfile(Path(__file__).with_name(name), dataset/name)
    assets = [{'file':name,'sha256':sha256(dataset/name)} for name in
              ['efficientnetb0_notop.h5','experiment_config.json','training_data.py','training_runtime.py']]
    manifest = {'schema_version':1,'fold':0,'allowed_roles':['train','validation'],'frames':frames,'assets':assets,
                'source_bindings':bindings or {},'pixel_policy':'Lossless approved 224x224 uint8 grayscale PNG -> repeat channels -> float32 0..255.',
                'outer_test_images_included':False,'training_authorization_required':True}
    write_json(dataset/'training_manifest.json', manifest); manifest_hash = sha256(dataset/'training_manifest.json')
    write_json(dataset/'dataset-metadata.json', {'id':f'{owner}/{DATASET_SLUG}','title':'NAFLD V4 Training Inputs 20261019',
               'licenses':[{'name':'other'}],'description':'Restricted private research. Rights remain with original authors. No public redistribution license is granted. Fold 0 training and inner validation only; no outer test images.'})
    write_json(dataset/'gate_e_approval.json', {'gate':'E','decision':'pending_user_approval',
               'note':'Must be replaced by the actual scoped user approval before any future upload or training.'})
    code = f'''from pathlib import Path
import hashlib, json, types, sys
expected_manifest = {manifest_hash!r}
matches = [p.parent for p in Path('/kaggle/input').rglob('training_manifest.json')
           if hashlib.sha256(p.read_bytes()).hexdigest() == expected_manifest]
assert len(matches) == 1, 'Require the exact reviewed pilot input dataset.'
root = matches[0]
manifest = json.loads((root/'training_manifest.json').read_text())
for item in manifest['assets']:
    assert Path(item['file']).name == item['file']
    assert hashlib.sha256((root/item['file']).read_bytes()).hexdigest() == item['sha256']
module = types.ModuleType('training_data'); module.__file__ = str(root/'training_data.py')
exec(compile((root/'training_data.py').read_text(), module.__file__, 'exec'), module.__dict__)
sys.modules['training_data'] = module
namespace = {{'__name__':'training_runtime', '__file__':str(root/'training_runtime.py')}}
exec(compile((root/'training_runtime.py').read_text(), str(root/'training_runtime.py'), 'exec'), namespace)
config = json.loads((root/'experiment_config.json').read_text())
approval = json.loads((root/'gate_e_approval.json').read_text())
result = namespace['run_training'](root, expected_manifest, config, approval, Path('/kaggle/working/gpu_pilot_result'))
print(json.dumps(result, indent=2))
assert result['status'] == 'completed'
'''
    write_json(kernel/'gpu_pilot.ipynb', {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
               'cells':[{'cell_type':'markdown','id':'scope','metadata':{},'source':['# Gate E pilot: private GPU, offline, 30-minute job cap. Actual Gate E approval required.\n']},
                        {'cell_type':'code','id':'pilot','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]})
    write_json(kernel/'kernel-metadata.json', {'id':f'{owner}/{NOTEBOOK_SLUG}','title':'NAFLD V4 E0 GPU Pilot 20261019',
               'code_file':'gpu_pilot.ipynb','language':'python','kernel_type':'notebook','is_private':True,
               'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':[f'{owner}/{DATASET_SLUG}'],
               'kernel_sources':[],'competition_sources':[],'model_sources':[]})
    write_json(output/'local_mapping.DO_NOT_UPLOAD.json', local_mapping)
    inventory = [{'path':p.relative_to(output).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)}
                 for directory in [dataset,kernel] for p in sorted(directory.iterdir())]
    review = {'schema_version':1,'gate':'E','decision':'pending_user_approval','dataset':f'{owner}/{DATASET_SLUG}',
              'notebook':f'{owner}/{NOTEBOOK_SLUG}','fold':0,'frames':len(frames),'patient_groups':len(roles),
              'frames_by_role':dict(Counter(r['role'] for r in frames)),
              'patients_by_role':{role:len(fold[role]) for role in ['train','validation']},
              'outer_test_images_included':False,'manifest_sha256':manifest_hash,
              'maximum_remote_runs':1,'timeout_seconds':1800,'single_visible_gpu':True,'internet_enabled':False,
              'transfer_inventory':inventory,'total_proposed_bytes':sum(i['bytes'] for i in inventory),
              'remote_transfer_approved':False,'training_approved':False,
              'approval_file_policy':'Only dataset/gate_e_approval.json changes after actual user approval; it must bind this manifest and experiment config. No sample/code/weights change is authorized by that administrative completion.'}
    write_json(output/'transfer_proposal.json', review)
    store = FrameStore(dataset, manifest_hash)
    weights_, classes = patient_weights(store.partition('train'))
    write_json(output/'local_validation.json', {'all_exported_frames_pixel_exact':len(frames),
               'outer_test_images_absent':True,'training_weights_mean':float(weights_.mean()),
               'class_weights_from_train_patients':classes,'training_executed':False})
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True); parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--splits',type=Path,required=True); parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    args = parser.parse_args(); context = load_preparation(args.workspace)
    rows = load_prepared_cohort(context,args.prepared); plan = make_splits(rows)
    if args.splits.exists(): raise ValueError('Preserve the frozen splits; choose a new version.')
    args.splits.mkdir(parents=True)
    write_json(args.splits/'patient_splits.json',plan)
    write_csv(args.splits/'patient_roles.csv', [{'fold':f['fold'],'role':role,'patient_id':pid}
              for f in plan['folds'] for role in ['train','validation','test'] for pid in f[role]])
    review = stage_training_bundle(rows,plan,args.prepared,args.weights,args.bundle,
                                  context['bindings'] | {'prepared_manifest_sha256':sha256(args.prepared/'output_manifest.json'),
                                                         'patient_splits_sha256':sha256(args.splits/'patient_splits.json')})
    print(json.dumps({k:v for k,v in review.items() if k!='transfer_inventory'},indent=2))


if __name__ == '__main__': main()
