"""Portable Gate E GPU pilot. Importing this module never fits or imports TF."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .training_data import FrameStore, digest, patient_weights, patient_predictions, classification_metrics
except ImportError:  # Verified standalone copy in the future private Kaggle dataset.
    from training_data import FrameStore, digest, patient_weights, patient_predictions, classification_metrics


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_training_approval(approval, config, manifest_sha256):
    if not (isinstance(approval, dict) and approval.get('gate') == 'E'
            and approval.get('decision') == 'approved' and approval.get('approved_by') == 'user'
            and approval.get('user_instruction') and approval.get('recorded_at_utc')
            and approval.get('permitted_scope') == ['one_private_gpu_pilot']
            and approval.get('manifest_sha256') == manifest_sha256
            and approval.get('experiment_config_sha256') == config_hash(config)
            and approval.get('maximum_remote_runs') == 1
            and approval.get('timeout_seconds') == 1800):
        raise PermissionError('Gate E: matching actual user approval of this GPU pilot is required.')
    if (config.get('purpose') != 'fold_0_technical_pilot' or config.get('fold') != 0
            or config.get('outer_test_evaluation') is not False
            or config.get('augmentation') != 'none'
            or config.get('input_shape') != [224, 224, 3]
            or config.get('input_range') != [0, 255]
            or config.get('batch_size') != 16 or config.get('dropout') != .35
            or config.get('head_epochs') != 5 or config.get('fine_tune_epochs') != 5
            or config.get('head_learning_rate') != .001 or config.get('fine_tune_learning_rate') != .00001
            or config.get('training_wall_seconds') != 1500):
        raise ValueError('Experiment differs from the reviewed baseline GPU pilot.')


def build_model(weights_path=None, dropout=.35):
    """Model construction only; None weights are reserved for synthetic checks."""
    import tensorflow as tf
    if weights_path is not None:
        # Stage the verified public asset locally, then use the official ImageNet
        # constructor branch. A custom weights path skips an ImageNet-specific
        # rescaling layer in Keras EfficientNet and would change its input contract.
        tf.keras.utils.get_file('efficientnetb0_notop.h5',
                                origin=Path(weights_path).resolve().as_uri(),
                                file_hash='50bc09e76180e00e4465e1a485ddc09d',
                                hash_algorithm='md5', cache_subdir='models')
    base = tf.keras.applications.EfficientNetB0(include_top=False,
                                               weights='imagenet' if weights_path is not None else None,
                                               input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input((224, 224, 3), dtype='float32', name='approved_pixels_0_255')
    # EfficientNet contains its own rescaling. Never divide the approved input by 255 here.
    features = base(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(dropout)(features)
    outputs = tf.keras.layers.Dense(4, activation='softmax', dtype='float32')(features)
    return tf.keras.Model(inputs, outputs, name='nafld_v4_four_class'), base


def configure_stage(model, base, stage, learning_rate):
    import tensorflow as tf
    if stage not in ['head', 'fine_tune']:
        raise ValueError('Unknown training stage.')
    base.trainable = stage == 'fine_tune'
    for layer in base.layers:
        layer.trainable = (stage == 'fine_tune' and layer.name.startswith(('block6', 'block7', 'top_'))
                           and not isinstance(layer, tf.keras.layers.BatchNormalization))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='weighted_frame_accuracy')])


def load_checkpoint(path, expected_sha256):
    import tensorflow as tf
    if digest(path) != expected_sha256:
        raise ValueError('Checkpoint hash mismatch.')
    model = tf.keras.models.load_model(path)
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 4):
        raise ValueError('Checkpoint has the wrong input/output contract.')
    return model


def run_training(root, manifest_sha256, config, approval, output):
    # This guard precedes TF import, model construction, output creation and fitting.
    require_training_approval(approval, config, manifest_sha256)
    root, output = Path(root).resolve(), Path(output).resolve()
    store = FrameStore(root, manifest_sha256)
    for item in store.manifest.get('assets', []):
        file = root / item['file']
        if Path(item['file']).name != item['file'] or file.is_symlink() or digest(file) != item['sha256']:
            raise ValueError('Runtime, config or pretrained asset differs from reviewed hashes.')
    weights_path = root / config['pretrained_weights_file']
    if digest(weights_path) != config['pretrained_weights_sha256']:
        raise ValueError('Pretrained weights mismatch; no automatic network download is allowed.')
    if output.exists() or output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Use a new output directory outside the dataset.')
    # Keras initialization creates KERAS_HOME and its parents. Claim the fresh
    # output first so that cache initialization cannot race a later mkdir.
    output.mkdir(parents=True)
    os.environ['KERAS_HOME'] = str(output/'.keras')
    import numpy as np
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('A GPU is required; do not silently run training on CPU.')
    inventory = [tf.config.experimental.get_device_details(gpu).get('device_name', gpu.name) for gpu in gpus]
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.utils.set_random_seed(config['seed'])
    tf.config.experimental.enable_op_determinism()
    tf.keras.mixed_precision.set_global_policy('float32')
    train, validation = store.partition('train'), store.partition('validation')
    _, class_weights = patient_weights(train)
    report = {'status': 'preflight_passed', 'training_executed': False, 'outer_test_evaluated': False,
              'manifest_sha256': manifest_sha256, 'experiment_config_sha256': config_hash(config),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_inventory': inventory,
              'visible_gpu_count': 1, 'software': {'python': platform.python_version(),
                                                'tensorflow': tf.__version__, 'keras': tf.keras.__version__,
                                                'numpy': np.__version__},
              'class_weights_from_training_patients_only': class_weights,
              'selection_metric': 'inner_validation_patient_and_class_weighted_frame_loss',
              'scope': 'Technical pilot; no outer test scores or final performance claim.', 'stages': []}
    save = lambda: (output/'run_report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    save(); start = time.monotonic()

    class Budget(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if time.monotonic()-start >= config['training_wall_seconds']:
                self.model.stop_training = True

    signature = (tf.TensorSpec((None,224,224,3), tf.float32), tf.TensorSpec((None,), tf.int64),
                 tf.TensorSpec((None,), tf.float32))
    epochs_started = [0]

    def generator(rows, shuffle):
        def generate():
            epoch = epochs_started[0]
            if shuffle: epochs_started[0] += 1
            yield from store.batches(rows, config['batch_size'], shuffle=shuffle,
                                     seed=config['seed']+epoch, class_weights=class_weights)
        return generate

    options = tf.data.Options(); options.experimental_deterministic = True
    train_ds = tf.data.Dataset.from_generator(generator(train, True), output_signature=signature).with_options(options).prefetch(1)
    val_ds = tf.data.Dataset.from_generator(generator(validation, False), output_signature=signature).with_options(options).prefetch(1)
    try:
        model, base = build_model(str(weights_path), config['dropout'])
        for stage, epochs, rate in [('head', config['head_epochs'], config['head_learning_rate']),
                                    ('fine_tune', config['fine_tune_epochs'], config['fine_tune_learning_rate'])]:
            if time.monotonic()-start >= config['training_wall_seconds']:
                break
            configure_stage(model, base, stage, rate)
            checkpoint = output/f'best_{stage}.keras'
            callbacks = [Budget(), tf.keras.callbacks.TerminateOnNaN(),
                         tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True),
                         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
                         tf.keras.callbacks.CSVLogger(str(output/f'{stage}_history.csv'))]
            report['training_executed'] = True; report['status'] = 'training'; save()
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
            values = history.history.get('val_loss', [])
            if not values or not np.isfinite(values).all() or not checkpoint.is_file():
                raise RuntimeError('No finite validated checkpoint was produced; stop pilot.')
            report['stages'].append({'stage': stage, 'epochs_completed': len(values),
                                     'best_validation_loss': float(min(values)), 'checkpoint': checkpoint.name,
                                     'checkpoint_sha256': digest(checkpoint), 'history': history.history})
            # Restored best weights remain on the same graph for the fine-tuning phase.
            save()
        if not report['stages']:
            raise RuntimeError('No training stage finished within the budget.')
        best = min(report['stages'], key=lambda s: s['best_validation_loss'])
        selected = load_checkpoint(output/best['checkpoint'], best['checkpoint_sha256'])
        probabilities = np.concatenate([selected(x, training=False).numpy()
                                        for x, _, _ in store.batches(validation, config['batch_size'],
                                                                    class_weights=class_weights)])
        patients = patient_predictions(validation, probabilities)
        (output/'inner_validation_patient_predictions.json').write_text(json.dumps(patients, indent=2)+'\n')
        report.update(status='completed', selected_checkpoint=best['checkpoint'],
                      inner_validation_metrics=classification_metrics(patients),
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-start,
                      budget_stop=time.monotonic()-start >= config['training_wall_seconds'])
        save(); return report
    except Exception as exc:
        report.update(status='failed', exception_type=type(exc).__name__, error=str(exc)[:1500])
        save(); raise
