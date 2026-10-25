"""Model smoke tests, with no benchmark training."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from dl.models.deeplabv3plus import build, export, predict


def test_gradient_and_export(tmp_path):
    model = build(
        {
            "height": 32,
            "width": 48,
            "classes": 3,
            "weights": None,
            "decoder_channels": 8,
            "freeze_encoder": True,
        }
    )
    x = tf.ones((1, 32, 48, 3)) * 100
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(
                tf.zeros((1, 32, 48), tf.int32), logits, from_logits=True
            )
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    assert np.isfinite(float(loss))
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    keras.optimizers.Adam().apply_gradients(zip(gradients, model.trainable_variables, strict=True))
    scores = predict(model, x).numpy()
    assert scores.shape == (1, 32, 48)
    assert scores.dtype == np.float32
    assert scores.min() >= 0 and scores.max() <= 1
    export(model, str(tmp_path / "saved"))
    loaded = tf.saved_model.load(str(tmp_path / "saved"))
    np.testing.assert_allclose(
        loaded.signatures["serving_default"](images=x)["score_map"], scores, atol=1e-5
    )
    model.save(tmp_path / "model.keras")
    restored = keras.models.load_model(tmp_path / "model.keras")
    np.testing.assert_allclose(predict(restored, x), scores, atol=1e-5)


def test_invalid_shape():
    with pytest.raises(ValueError):
        build({"height": 33, "width": 48, "classes": 3})


def test_attribution_shapes_and_empty_region():
    from dl.evaluation.xai import seg_gradcam

    model = build({"height": 32, "width": 48, "classes": 3, "weights": None, "decoder_channels": 8})
    image = tf.ones((1, 32, 48, 3))
    for layer in ("decoder_2", "aspp_rate_6", "aspp_rate_12", "aspp_rate_18"):
        heatmap = seg_gradcam(model, image, tf.ones((32, 48)), layer)
        assert tuple(heatmap.shape) == (32, 48)
        assert bool(tf.reduce_all(tf.math.is_finite(heatmap)))
    with pytest.raises(tf.errors.InvalidArgumentError):
        seg_gradcam(model, image, tf.zeros((32, 48)))
