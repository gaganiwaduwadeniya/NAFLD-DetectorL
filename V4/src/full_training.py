"""Guarded five-fold experiment. No TensorFlow import or fitting on import.

The pilot-only guard and runtime are preserved. Reuse its verified image loader,
model constructor and stage configuration with a separate full-experiment guard.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import zipfile

import numpy as np

try:
    from .training_data import FrameStore, digest, patient_weights, patient_predictions, classification_metrics
    from .training_runtime import build_model, configure_stage, load_checkpoint, config_hash
except ImportError:
    from training_data import FrameStore, digest, patient_weights, patient_predictions, classification_metrics
    from training_runtime import build_model, configure_stage, load_checkpoint, config_hash


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def full_config(weights_sha256):
    return dict(schema_version=1, purpose='five_fold_internal_cv', seed=42, folds=list(range(5)),
        architecture='ImageNet_EfficientNetB0_GAP_Dropout_Dense4_softmax',
        input_shape=[224,224,3], input_range=[0,255], batch_size=16, dropout=.35,
        head_epochs=10, fine_tune_epochs=30, early_stopping_patience=5,
        head_learning_rate=.001, fine_tune_learning_rate=.00001,
        fine_tune_blocks=['block6','block7','top_'], batchnorm='frozen_inference_mode',
        augmentation='none', precision='float32', patient_and_class_weighting='training_patient_counts_only',
        outer_test_evaluation=True, fold_training_wall_seconds=1200,
        total_training_wall_seconds=6000, job_timeout_seconds=7200,
        pretrained_weights_file='efficientnetb0_notop.h5', pretrained_weights_sha256=weights_sha256,
        bootstrap_draws=2000, bootstrap_seed=42, bootstrap_unit='patient_stratified_by_grade',
        expected_patients=109, expected_frames=1640,
        gradcam='first_patient_per_true_grade_plus_first_additional_error_per_fold',
        clinical_claim=False)


def require_full_approval(approval, config, manifest_sha256):
    if not (isinstance(approval, dict) and approval.get('gate') == 'E'
            and approval.get('decision') == 'approved' and approval.get('approved_by') == 'user'
            and approval.get('user_instruction') and approval.get('recorded_at_utc')
            and approval.get('permitted_scope') == ['one_private_five_fold_experiment']
            and approval.get('manifest_sha256') == manifest_sha256
            and approval.get('experiment_config_sha256') == config_hash(config)
            and approval.get('maximum_remote_runs') == 1
            and approval.get('timeout_seconds') == 7200
            and approval.get('full_cohort_private_upload_approved') is True):
        raise PermissionError('Matching actual user approval of the full five-fold experiment is required.')
    if config != full_config(config.get('pretrained_weights_sha256')):
        raise ValueError('Full experiment differs from the reviewed configuration.')


class FullFrameStore(FrameStore):
    """Canonical full-cohort rows; inherited load/batches preserve pixel semantics."""
    def __init__(self, root, manifest_sha256):
        self.root = Path(root).resolve()
        path = self.root/'training_manifest.json'
        if digest(path) != manifest_sha256:
            raise ValueError('Full training manifest hash mismatch.')
        self.manifest = json.loads(path.read_text())
        if (self.manifest.get('schema_version') != 2
                or self.manifest.get('purpose') != 'five_fold_internal_cv'):
            raise ValueError('Require the full-experiment manifest, not a pilot manifest.')
        self.rows = self.manifest['frames']; self._indexed = {}; self.groups = {}
        for row in self.rows:
            key, group = row['frame_id'], row['group_id']
            if (not re.fullmatch(r'F\d{6}', key) or key in self._indexed
                    or not re.fullmatch(r'G\d{4}', group) or row['file'] != key+'.png'
                    or type(row['grade']) is not int or row['grade'] not in range(4)
                    or ('role' in row) or (group in self.groups and self.groups[group] != row['grade'])):
                raise ValueError('Invalid full-cohort frame/group/grade.')
            self._indexed[key] = dict(row); self.groups[group] = row['grade']
            p = self.root/row['file']
            if not p.is_file() or p.is_symlink() or digest(p) != row['sha256']:
                raise ValueError('Prepared PNG hash/path mismatch.')
        if not self.rows or set(self.groups.values()) != set(range(4)):
            raise ValueError('Require all four classes.')
        self.folds = self.manifest['folds']
        if [f.get('fold') for f in self.folds] != list(range(5)):
            raise ValueError('Require five fixed outer folds in order.')
        outer = []
        for f in self.folds:
            partitions = [f[role] for role in ('train','validation','test')]
            sets = list(map(set, partitions))
            if (any(len(a) != len(b) for a,b in zip(partitions,sets))
                    or sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]
                    or set.union(*sets) != set(self.groups)
                    or any({self.groups[g] for g in s} != set(range(4)) for s in sets)):
                raise ValueError('Patient partition overlap, missing group, or missing class.')
            outer.extend(f['test'])
        if Counter(outer) != Counter({g:1 for g in self.groups}):
            raise ValueError('Every patient must occur in exactly one outer fold.')
        assets = self.manifest.get('assets', [])
        names = [a['file'] for a in assets]
        if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError('Unsafe/repeated full-experiment asset.')
        allowed = {'training_manifest.json','dataset-metadata.json','gate_e_full_approval.json'}
        allowed.update(names); allowed.update(r['file'] for r in self.rows)
        for p in self.root.iterdir():
            if p.name not in allowed or not p.is_file() or p.is_symlink():
                raise ValueError('Unexpected private full-experiment input.')
        for asset in assets:
            if digest(self.root/asset['file']) != asset['sha256']:
                raise ValueError('Full-experiment runtime/config/weight asset mismatch.')

    def partition(self, fold, role):
        if type(fold) is not int or fold not in range(5) or role not in ('train','validation','test'):
            raise ValueError('Unknown fold/role.')
        groups = set(self.folds[fold][role])
        return [r for r in self.rows if r['group_id'] in groups]


def frame_predictions(rows, probabilities, fold):
    # Reuse normalization checks, without aggregating frame outputs here.
    patient_predictions(rows, probabilities)
    return [dict(frame_id=r['frame_id'], group_id=r['group_id'], grade=r['grade'],
                 probabilities=p.tolist(), prediction=int(np.argmax(p)), fold=fold)
            for r,p in zip(rows, np.asarray(probabilities))]


def predict_rows(model, store, rows):
    return np.concatenate([np.asarray(model(x, training=False))
                           for x,_,_ in store.batches(rows,16)])


def predict_outer_fold(model, store, fold, directory):
    directory = Path(directory); lock = directory/'selection_lock.json'
    if not lock.is_file():
        raise ValueError('Outer inference requires a completed inner-selection lock.')
    selected = json.loads(lock.read_text()); name = selected.get('checkpoint','')
    if (selected.get('fold') != fold or selected.get('selected_using') != 'inner_validation_only'
            or not name or Path(name).name != name or not (directory/name).is_file()
            or digest(directory/name) != selected.get('checkpoint_sha256')):
        raise ValueError('Selected checkpoint lock/hash differs before outer inference.')
    rows = store.partition(fold,'test')
    probabilities = predict_rows(model,store,rows)
    patients = [dict(p,fold=fold) for p in patient_predictions(rows,probabilities)]
    return patients, frame_predictions(rows,probabilities,fold)


def validate_oof(store, patients, frames):
    expected = {r['frame_id']:r for r in store.rows}
    if (len(patients) != len(store.groups) or len({p['group_id'] for p in patients}) != len(patients)
            or {p['group_id'] for p in patients} != set(store.groups)
            or len(frames) != len(expected) or len({p['frame_id'] for p in frames}) != len(frames)
            or {p['frame_id'] for p in frames} != set(expected)):
        raise ValueError('OOF coverage must be complete, with no duplicate patient/frame.')
    for p in patients:
        if (type(p.get('fold')) is not int or p['fold'] not in range(5)
                or p['group_id'] not in store.folds[p['fold']]['test']
                or p['grade'] != store.groups[p['group_id']]):
            raise ValueError('OOF patient fold/label mismatch.')
    for p in frames:
        r = expected[p['frame_id']]
        if (p['group_id'] != r['group_id'] or p['grade'] != r['grade']
                or type(p.get('fold')) is not int or p['fold'] not in range(5)
                or p['group_id'] not in store.folds[p['fold']]['test']
                or p['prediction'] != int(np.argmax(p['probabilities']))):
            raise ValueError('OOF frame fold/label/prediction mismatch.')
    for fold in range(5):
        selected = sorted([p for p in frames if p['fold']==fold],key=lambda p:p['frame_id'])
        rows = [expected[p['frame_id']] for p in selected]
        recomputed = patient_predictions(rows,[p['probabilities'] for p in selected])
        actual = {p['group_id']:p for p in patients if p['fold']==fold}
        for p in recomputed:
            q = actual.get(p['group_id'],{})
            if (p['grade'] != q.get('grade') or p['frames'] != q.get('frames')
                    or p['prediction'] != q.get('prediction')
                    or not np.allclose(p['probabilities'],q.get('probabilities',[]),atol=1e-7,rtol=0)):
                raise ValueError('OOF patient probabilities do not aggregate the saved frames.')
    fm = classification_metrics(frames); fm['frames'] = fm.pop('patients')
    return dict(patient_metrics=classification_metrics(patients), frame_metrics=fm)


def majority_predictions(train_rows, patients):
    groups = {r['group_id']:r['grade'] for r in train_rows}
    counts = Counter(groups.values())
    majority = min(counts,key=lambda y:(-counts[y],y))
    return [dict(p,prediction=majority,probabilities=np.eye(4)[majority].tolist()) for p in patients]


def bootstrap_intervals(patients, draws=2000, seed=42):
    rng = np.random.default_rng(seed)
    strata = [[p for p in patients if p['grade']==g] for g in range(4)]
    if any(not s for s in strata) or draws < 1:
        raise ValueError('Patient bootstrap requires four nonempty class strata.')
    results = []
    for _ in range(draws):
        sample = [s[int(i)] for s in strata for i in rng.integers(len(s),size=len(s))]
        m = classification_metrics(sample)
        results.append([m['balanced_accuracy'],m['macro_f1']])
    q = np.quantile(results,[.025,.975],axis=0)
    return dict(balanced_accuracy=q[:,0].tolist(), macro_f1=q[:,1].tolist(),
                draws=draws, seed=seed, method='patient_resampling_stratified_by_true_grade',
                scope='Conditional on observed OOF predictions; excludes retraining and external variability.')


def export_archive(output):
    output = Path(output).resolve()
    inventory = [dict(file=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=digest(p))
                 for p in sorted(output.rglob('*')) if p.is_file() and not p.is_symlink()
                 and '.keras' not in p.relative_to(output).parts and p.name != 'output_inventory.json']
    save_json(output/'output_inventory.json', inventory)
    archive = output.parent/'full_cv_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for item in inventory + [dict(file='output_inventory.json')]:
            p = output/item['file']
            z.write(p,'full_cv_results/'+item['file'])
    return archive


class Tee:
    def __init__(self,*handles): self.handles=handles
    def write(self,text):
        for h in self.handles: h.write(text); h.flush()
        return len(text)
    def flush(self):
        for h in self.handles: h.flush()


def run_experiment(root, manifest_sha256, config, approval, output, on_fold=None):
    # Reject pending/pilot/stale approval before TF import or creating output.
    require_full_approval(approval,config,manifest_sha256)
    store = FullFrameStore(root,manifest_sha256)
    if len(store.rows)!=config['expected_frames'] or len(store.groups)!=config['expected_patients']:
        raise ValueError('Full-cohort count differs from the reviewed scope.')
    root, output = Path(root).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Require a fresh output directory outside the dataset.')
    output.mkdir(parents=True)
    # Claim the output directory BEFORE Keras initializes its cache.
    os.environ['KERAS_HOME'] = str(output/'.keras')
    save_json(output/'experiment_config.json',config)
    save_json(output/'input_manifest.json',store.manifest)
    save_json(output/'actual_approval.json',approval)
    report = dict(status='initializing',training_executed=False,outer_test_evaluated=False,
        manifest_sha256=manifest_sha256,experiment_config_sha256=config_hash(config),
        started_at_utc=datetime.now(timezone.utc).isoformat(),folds=[],
        scope='Internal CV; technical-QC exposure and fold-0 pilot feedback disclosed. Not external validation.')
    save_json(output/'run_report.json',report)
    with (output/'run.log').open('w') as log, redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
        try:
            import tensorflow as tf
            gpus=tf.config.list_physical_devices('GPU')
            if not gpus: raise RuntimeError('GPU required; no silent CPU training.')
            report['gpu_inventory']=[tf.config.experimental.get_device_details(g).get('device_name',g.name) for g in gpus]
            tf.config.set_visible_devices(gpus[0],'GPU'); tf.config.experimental.set_memory_growth(gpus[0],True)
            tf.config.experimental.enable_op_determinism()
            tf.keras.mixed_precision.set_global_policy('float32')
            report.update(visible_gpu_count=1, software=dict(python=platform.python_version(),
                tensorflow=tf.__version__,keras=tf.keras.__version__,numpy=np.__version__))
            start=time.monotonic(); patients=[]; frames=[]; baseline=[]
            for fold in range(5):
                tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(config['seed']+fold)
                directory=output/f'fold_{fold}'; directory.mkdir()
                train=store.partition(fold,'train'); inner=store.partition(fold,'validation')
                _,classes=patient_weights(train)
                fold_report=dict(fold=fold,status='training',stages=[],class_weights=classes,
                                 selected_using='inner_validation_only',outer_test_evaluated=False)
                report['folds'].append(fold_report); report['status']='training'; save_json(output/'run_report.json',report)
                fold_start=time.monotonic()
                deadline=min(fold_start+config['fold_training_wall_seconds'],start+config['total_training_wall_seconds'])
                expired=[False]
                class Budget(tf.keras.callbacks.Callback):
                    def on_train_batch_end(self,batch,logs=None):
                        if time.monotonic()>=deadline:
                            expired[0]=True; self.model.stop_training=True
                signature=(tf.TensorSpec((None,224,224,3),tf.float32),tf.TensorSpec((None,),tf.int64),tf.TensorSpec((None,),tf.float32))
                epoch=[0]
                def make_dataset(rows,shuffle):
                    def generate():
                        number=epoch[0]
                        if shuffle: epoch[0]+=1
                        yield from store.batches(rows,16,shuffle,config['seed']+fold*1000+number,classes)
                    options=tf.data.Options(); options.experimental_deterministic=True
                    return tf.data.Dataset.from_generator(generate,output_signature=signature).with_options(options).prefetch(1)
                train_ds=make_dataset(train,True); val_ds=make_dataset(inner,False)
                model,base=build_model(str(root/config['pretrained_weights_file']),config['dropout'])
                for stage,epochs,rate in [('head',config['head_epochs'],config['head_learning_rate']),
                                          ('fine_tune',config['fine_tune_epochs'],config['fine_tune_learning_rate'])]:
                    if time.monotonic()>=deadline: expired[0]=True; break
                    configure_stage(model,base,stage,rate)
                    checkpoint=directory/f'best_{stage}.keras'
                    callbacks=[Budget(),tf.keras.callbacks.TerminateOnNaN(),
                        tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['early_stopping_patience'],restore_best_weights=True),
                        tf.keras.callbacks.CSVLogger(str(directory/f'{stage}_history.csv'))]
                    report['training_executed']=True; save_json(output/'run_report.json',report)
                    history=model.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=callbacks,verbose=2).history
                    if (not history.get('val_loss') or not all(np.isfinite(v).all() for v in history.values())
                            or not checkpoint.is_file()):
                        raise RuntimeError('Non-finite history or no validated checkpoint; stop experiment.')
                    fold_report['stages'].append(dict(stage=stage,epochs_completed=len(history['val_loss']),
                        best_validation_loss=float(min(history['val_loss'])),checkpoint=checkpoint.name,
                        checkpoint_sha256=digest(checkpoint),history=history))
                    save_json(directory/'fold_report.json',fold_report); save_json(output/'run_report.json',report)
                    if expired[0]: break
                fold_report['training_elapsed_seconds']=time.monotonic()-fold_start
                if expired[0] or len(fold_report['stages'])!=2:
                    fold_report['status']='incomplete_budget'; report['status']='incomplete_budget'
                    save_json(directory/'fold_report.json',fold_report); break
                best=min(fold_report['stages'],key=lambda s:s['best_validation_loss'])
                selection=dict(fold=fold,checkpoint=best['checkpoint'],checkpoint_sha256=best['checkpoint_sha256'],
                    selected_using='inner_validation_only',selection_metric='patient_class_weighted_frame_loss',
                    selected_at_utc=datetime.now(timezone.utc).isoformat())
                save_json(directory/'selection_lock.json',selection)
                selected=load_checkpoint(directory/best['checkpoint'],best['checkpoint_sha256'])
                inner_patients=patient_predictions(inner,predict_rows(selected,store,inner))
                save_json(directory/'inner_validation_patient_predictions.json',inner_patients)
                outer_patients,outer_frames=predict_outer_fold(selected,store,fold,directory)
                save_json(directory/'outer_patient_predictions.json',outer_patients)
                save_json(directory/'outer_frame_predictions.json',outer_frames)
                fold_report.update(status='completed',outer_test_evaluated=True,selected_checkpoint=best['checkpoint'],
                    selected_checkpoint_sha256=best['checkpoint_sha256'],
                    inner_validation_metrics=classification_metrics(inner_patients),
                    outer_patient_metrics=classification_metrics(outer_patients))
                patients.extend(outer_patients); frames.extend(outer_frames)
                baseline.extend(majority_predictions(train,outer_patients))
                report['outer_test_evaluated']=True
                save_json(directory/'fold_report.json',fold_report); save_json(output/'run_report.json',report)
                if on_fold: on_fold(store,fold,fold_report,selected,outer_patients,directory)
                del selected,model,base,train_ds,val_ds
            if len(report['folds'])==5 and all(f['status']=='completed' for f in report['folds']):
                metrics=validate_oof(store,patients,frames)
                save_json(output/'oof_patient_predictions.json',sorted(patients,key=lambda p:p['group_id']))
                save_json(output/'oof_frame_predictions.json',sorted(frames,key=lambda p:p['frame_id']))
                save_json(output/'majority_baseline_patient_predictions.json',baseline)
                report.update(status='completed',**metrics,
                    majority_baseline_patient_metrics=classification_metrics(baseline),
                    patient_bootstrap_95=bootstrap_intervals(patients,config['bootstrap_draws'],config['bootstrap_seed']))
            report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
            save_json(output/'run_report.json',report)
            return report
        except Exception as exc:
            report.update(status='failed',exception_type=type(exc).__name__,error=str(exc)[:1500])
            save_json(output/'run_report.json',report)
            raise
