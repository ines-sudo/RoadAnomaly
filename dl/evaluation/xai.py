"""Region-targeted Seg-Grad-CAM for the DeepLab MSP anomaly score.

Reporting only. Attribution is not a causal explanation or a selection metric.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import keras


def seg_gradcam(model: Any, image: Any, region: Any, layer_name: str = "decoder_2") -> Any:
    """Explain the mean MSP score in a fixed nonempty region for one image.

    Region must be chosen before this call. It is held constant during differentiation.
    The global pooling branch has no detailed spatial interpretation.
    """
    image = tf.cast(image, tf.float32)
    region = tf.cast(region, tf.float32)
    tf.debugging.assert_equal(tf.shape(image)[0], 1)
    tf.debugging.assert_equal(tf.shape(region), tf.shape(image)[1:3])
    tf.debugging.assert_positive(tf.reduce_sum(region))
    inspector = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        features, logits = inspector(image, training=False)
        score = 1 - tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1)
        target = tf.reduce_sum(score[0] * region) / tf.reduce_sum(region)
    gradients = tape.gradient(target, features)
    if gradients is None:
        raise ValueError("Layer has no differentiable path to the target")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1, keepdims=True))
    heatmap = tf.image.resize(heatmap, tf.shape(image)[1:3], method="bilinear")[0, :, :, 0]
    return tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
