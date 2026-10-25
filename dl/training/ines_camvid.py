"""Learning experiment only: CamVid training, normal validation, no benchmark fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from dl.models.deeplabv3plus import build


def load_manifest(root: Path, path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    assignment = sorted({(row["scene"], row["split"]) for row in manifest["frames"]})
    digest = hashlib.sha256(json.dumps(assignment).encode()).hexdigest()
    if digest != manifest["split_hash"]:
        raise ValueError("Split hash mismatch")
    scenes: dict[str, str] = {}
    for row in manifest["frames"]:
        if row["scene"] in scenes and scenes[row["scene"]] != row["split"]:
            raise ValueError("Scene leakage")
        scenes[row["scene"]] = row["split"]
        for key in ("image", "mask"):
            target = (root / row[key]).resolve()
            if not target.is_relative_to((root / "artifacts/camvid").resolve()):
                raise ValueError("Only the external CamVid corpus is allowed for training")
            if hashlib.sha256(target.read_bytes()).hexdigest() != row[key + "_sha256"]:
                raise ValueError(f"Content hash mismatch: {target}")
    return manifest


def dataset(
    root: Path, manifest: dict[str, Any], split: str, height: int, width: int, batch: int
) -> Any:
    rows = [row for row in manifest["frames"] if row["split"] == split]
    if not rows:
        raise ValueError(f"Empty split {split}")

    def generate() -> Any:
        for row in rows:
            with Image.open(root / row["image"]) as image:
                x = np.asarray(
                    image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
            with Image.open(root / row["mask"]) as image:
                y = np.asarray(
                    image.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32
                )
            valid = (y != manifest["ignore_label"]).astype(np.float32)
            yield x, np.where(valid, y, 0).astype(np.int32), valid

    signature = (
        tf.TensorSpec((height, width, 3), tf.float32),
        tf.TensorSpec((height, width), tf.int32),
        tf.TensorSpec((height, width), tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generate, output_signature=signature)
    if split == "train":
        ds = ds.shuffle(len(rows), seed=42)
    return ds.batch(batch).prefetch(1)


def run(root: Path, epochs: int = 3) -> None:
    path = root / "configs/datasets/camvid.json"
    if not path.exists():
        raise FileNotFoundError("Run python -m dl.data.prepare_camvid first. Never train on data/.")
    manifest = load_manifest(root, path)
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    config = json.loads((root / "configs/ines-camvid-learning.json").read_text())
    if config["classes"] != len(manifest["classes"]):
        raise ValueError("Class map mismatch")
    model = build(config)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight"
        ),
    )
    output = root / "artifacts/ines-deeplabv3plus"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history = model.fit(
        dataset(root, manifest, "train", 288, 512, 2),
        validation_data=dataset(root, manifest, "val", 288, 512, 2),
        epochs=epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output / "best.keras"), monitor="val_loss", save_best_only=True
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        ],
    )
    record = {
        "profile": "learning",
        "config": config,
        "seed": 42,
        "epochs_requested": epochs,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_hash": manifest["split_hash"],
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "seconds": time.time() - started,
        "history": history.history,
        "benchmark_used_for_training": False,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2))
    records = root / "docs/experiments/learning"
    records.mkdir(parents=True, exist_ok=True)
    (records / "ines-camvid.md").write_text(
        "# CamVid DeepLabV3+ learning run\n\nProfile: learning\n\n"
        + "Not official evidence. See artifacts/ines-deeplabv3plus/run.json.\n\n"
        + "```json\n"
        + json.dumps(record, indent=2)
        + "\n```\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(Path.cwd(), args.epochs)
