"""Real CPU model construction and serialization only. Never calls fit."""
from pathlib import Path
import hashlib
import urllib.request
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow', reason='Use isolated .venv-training-check-py313 for framework checks.')
from V4.src.training_runtime import build_model, configure_stage, load_checkpoint


def test_model_head_upper_blocks_bn_and_optimizer_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tf.keras.Model, 'fit', lambda *a, **kw: pytest.fail('Fitting is forbidden in local checks.'))
    model, base = build_model()
    pixels = np.full((1,224,224,3), 37, np.float32)
    assert model.output_shape == (None,4)
    assert np.allclose(model(pixels, training=False).numpy().sum(axis=1), 1)
    configure_stage(model, base, 'head', .001)
    assert not base.trainable_variables and len(model.trainable_variables) == 2
    configure_stage(model, base, 'fine_tune', .00001)
    assert base.trainable_variables
    assert all(not layer.trainable for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
    assert base.get_layer('top_conv').trainable and not base.get_layer('block5a_expand_conv').trainable
    model.optimizer.build(model.trainable_variables)
    model.optimizer.iterations.assign(7)  # Synthetic serialization state, no optimization step.
    before = model(pixels, training=False).numpy()
    path = tmp_path/'checkpoint.keras'; model.save(path)
    restored = load_checkpoint(path, hashlib.sha256(path.read_bytes()).hexdigest())
    np.testing.assert_allclose(restored(pixels, training=False).numpy(), before, atol=1e-6)
    assert int(restored.optimizer.iterations.numpy()) == 7
    assert len(restored.optimizer.variables) == len(model.optimizer.variables)
    with pytest.raises(ValueError, match='hash'):
        load_checkpoint(path, '0'*64)


def test_offline_pretrained_branch_keeps_imagenet_normalization(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'artifacts/pretrained/efficientnetb0_notop.h5'
    if not path.is_file(): pytest.skip('Official public pretrained file is not staged locally.')
    retrieve = urllib.request.urlretrieve
    def local_only(url, *args, **kwargs):
        assert not url.startswith(('http:', 'https:')), 'Offline model construction must not fetch the network.'
        return retrieve(url, *args, **kwargs)
    monkeypatch.setattr(urllib.request, 'urlretrieve', local_only)
    monkeypatch.setattr(tf.keras.Model, 'fit', lambda *a, **kw: pytest.fail('No fitting in framework checks.'))
    _, base = build_model(str(path))
    native = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
    assert sum(isinstance(layer, tf.keras.layers.Rescaling) for layer in base.layers) == 2
    pixels = np.full((1,224,224,3), 83, np.float32)
    np.testing.assert_allclose(base(pixels, training=False).numpy(), native(pixels, training=False).numpy(), atol=1e-6)


def test_fresh_process_keras_cache_does_not_break_runtime_startup(tmp_path):
    """Real Keras initialization with simulated discovery; stop before building a model."""
    script = textwrap.dedent('''
        import builtins, hashlib, json, sys
        from pathlib import Path
        from types import SimpleNamespace
        from V4.src import training_runtime as runtime
        root=Path(sys.argv[1]); output=root/'result'; data=root/'inputs'; data.mkdir()
        weights=data/'fixture.h5'; weights.write_bytes(b'synthetic startup fixture, never model weights')
        config=json.loads(Path('V4/configs/training_pilot.proposed.json').read_text())
        config.update(pretrained_weights_file='fixture.h5', pretrained_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
        approval={'gate':'E','decision':'approved','approved_by':'user',
                  'user_instruction':'SYNTHETIC TEST FIXTURE ONLY; no fitting permitted',
                  'recorded_at_utc':'synthetic-fixture','permitted_scope':['one_private_gpu_pilot'],
                  'manifest_sha256':'synthetic-manifest','experiment_config_sha256':runtime.config_hash(config),
                  'maximum_remote_runs':1,'timeout_seconds':1800}
        class FixtureStore:
            manifest={'assets':[]}
            def __init__(self,*args):pass
            def partition(self,role):return [{'group_id':role,'grade':0}]
        runtime.FrameStore=FixtureStore
        original_import=builtins.__import__
        initialized=[]
        def forbidden_fit(*args,**kwargs):raise AssertionError('Fitting forbidden in startup regression')
        def import_with_simulated_discovery(name,*args,**kwargs):
            module=original_import(name,*args,**kwargs)
            if (name=='tensorflow' and args and isinstance(args[0],dict)
                    and args[0].get('__name__')=='V4.src.training_runtime' and not initialized):
                initialized.append(True)
                module.config.list_physical_devices=lambda kind: [SimpleNamespace(name='SIMULATED_DISCOVERY_ONLY')] if kind=='GPU' else []
                module.config.experimental.get_device_details=lambda gpu: {'device_name':'SIMULATED_DISCOVERY_ONLY'}
                module.config.set_visible_devices=lambda *args:None
                module.config.experimental.set_memory_growth=lambda *args:None
                # Real Keras import creates its configured cache directory.
                module.keras.Model.fit=forbidden_fit
            return module
        builtins.__import__=import_with_simulated_discovery
        class StopBeforeModel(Exception):pass
        def stop_before_model(*args,**kwargs):raise StopBeforeModel('Controlled stop before model construction')
        runtime.build_model=stop_before_model
        try:
            runtime.run_training(data,'synthetic-manifest',config,approval,output)
        except StopBeforeModel:pass
        else:raise AssertionError('Expected controlled pre-model stop')
        report=json.loads((output/'run_report.json').read_text())
        assert report['exception_type']=='StopBeforeModel'
        assert report['training_executed'] is False
        assert (output/'.keras/keras.json').is_file()
        print(json.dumps({'startup_reached_model_boundary':True,'real_keras_initialization':True,
                          'gpu_discovery_simulated':True,'model_constructed':False,'fit_executed':False}))
    ''')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='2',
                       TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2], env=environment,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    outcome = json.loads(result.stdout.splitlines()[-1])
    assert outcome['startup_reached_model_boundary'] and outcome['fit_executed'] is False
