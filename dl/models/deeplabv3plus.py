"""Keras DeepLabV3+ with a ResNet50 encoder and explicit ASPP branches.

Architecture reference: https://keras.io/examples/vision/deeplabv3_plus/
Input: float32 RGB in [0,255]. Output: normal-road semantic logits.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _conv(x: Any, filters: int, name: str, kernel: int = 3, rate: int = 1) -> Any:
    x = layers.Conv2D(
        filters, kernel, padding="same", dilation_rate=rate, use_bias=False, name=name + "_conv"
    )(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.Activation("relu", name=name)(x)


def build(config: dict[str, Any]) -> Any:
    """Build logits. weights=None is for smoke only; imagenet for transfer learning."""
    height, width = config.get("height", 288), config.get("width", 512)
    classes = config["classes"]
    if min(height, width) < 32 or height % 16 or width % 16 or classes < 2:
        raise ValueError("H/W must be multiples of 16, >=32; classes must be >=2.")
    inputs = keras.Input((height, width, 3), name="rgb_0_255")
    # ResNet preprocess converts RGB to BGR and subtracts ImageNet means.
    x = keras.applications.resnet50.preprocess_input(inputs)
    backbone = keras.applications.ResNet50(
        include_top=False, weights=config.get("weights", "imagenet"), input_tensor=x
    )
    backbone.trainable = not config.get("freeze_encoder", True)
    high = backbone.get_layer("conv4_block6_2_relu").output
    low = backbone.get_layer("conv2_block3_2_relu").output
    channels = config.get("decoder_channels", 128)
    branches = [_conv(high, channels, "aspp_1x1", kernel=1)]
    for rate in (6, 12, 18):
        branches.append(_conv(high, channels, f"aspp_rate_{rate}", rate=rate))
    pooled = layers.GlobalAveragePooling2D(keepdims=True, name="aspp_pool")(high)
    # No BatchNorm on the 1x1 pooled branch: works with batch size one.
    pooled = layers.Conv2D(channels, 1, activation="relu", name="aspp_global")(pooled)
    pooled = layers.Resizing(height // 16, width // 16, interpolation="bilinear")(pooled)
    x = layers.Concatenate(name="aspp_concat")([*branches, pooled])
    x = _conv(x, channels, "aspp_projection", kernel=1)
    x = layers.Resizing(height // 4, width // 4, interpolation="bilinear")(x)
    low = _conv(low, 48, "low_projection", kernel=1)
    x = layers.Concatenate(name="decoder_concat")([x, low])
    x = _conv(x, channels, "decoder_1")
    x = _conv(x, channels, "decoder_2")
    x = layers.Conv2D(classes, 1, name="semantic_logits_small")(x)
    logits = layers.Resizing(height, width, interpolation="bilinear", name="logits")(x)
    return keras.Model(inputs, logits, name="deeplabv3plus")


def predict(model: Any, batch: Any) -> Any:
    """MSP anomaly scores (N,H,W), float32. Not a calibrated probability."""
    logits = model(tf.cast(batch, tf.float32), training=False)
    return tf.cast(1 - tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1), tf.float32)


def export(model: Any, saved_model_dir: str) -> None:
    """Export score-map signature; registry manifests are a separate team concern."""
    height, width = model.input_shape[1:3]
    signature = tf.TensorSpec((None, height, width, 3), tf.float32, name="images")

    @tf.function(input_signature=[signature])
    def serve(images: Any) -> dict[str, Any]:
        return {"score_map": predict(model, images)}

    tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": serve})
