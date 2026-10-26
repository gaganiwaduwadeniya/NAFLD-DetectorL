"""Read-only nested EfficientNet Grad-CAM using pre-softmax target scores.

Algorithm reference: https://keras.io/examples/vision/grad_cam/
No image preprocessing, activation mutation, model fitting or optimizer update.
"""
import numpy as np


def gradcam(model, pixels, target):
    import tensorflow as tf
    base=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model))
    extractor=tf.keras.Model(base.inputs,[base.get_layer('top_conv').output,base.output])
    x=tf.convert_to_tensor(np.asarray(pixels,dtype=np.float32)[None])
    with tf.GradientTape() as tape:
        tape.watch(x)
        conv,value=extractor(x,training=False)
        after=False
        for layer in model.layers:
            if layer is base: after=True; continue
            if not after: continue
            if layer is model.layers[-1]:
                logits=tf.linalg.matmul(value,layer.kernel)+layer.bias
            else:
                value=layer(value,training=False) if isinstance(layer,tf.keras.layers.Dropout) else layer(value)
        score=logits[:,int(target)]
    gradient=tape.gradient(score,conv)
    if gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient))):
        raise RuntimeError('Grad-CAM has no finite connected gradient.')
    pooled=tf.reduce_mean(gradient,axis=(0,1,2))
    heat=tf.nn.relu(tf.reduce_sum(conv[0]*pooled,axis=-1))
    heat=tf.math.divide_no_nan(heat,tf.reduce_max(heat)).numpy()
    if not np.isfinite(heat).all(): raise RuntimeError('Non-finite Grad-CAM heatmap.')
    return heat,tf.nn.softmax(logits).numpy()[0],float(tf.linalg.global_norm([gradient]).numpy())
