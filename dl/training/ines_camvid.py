"""Learning experiment only: CamVid training, no benchmark fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np   
import tensorflow as tf
from PIL import Image
import keras

from dl.models.deeplabv3plus import build


def load_manifest(root: Path, path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assignment = sorted(
        {(row["scene"], row["split"]) for row in manifest["frames"]}
    )
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
            allowed = (root / "artifacts/camvid").resolve()

            if not target.is_relative_to(allowed):
                raise ValueError(
                    "Only the external CamVid corpus is allowed for training"
                )

            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != row[key + "_sha256"]:
                raise ValueError(f"Content hash mismatch: {target}")

    return manifest


def dataset(
    root: Path,
    manifest: dict[str, Any],
    split: str,
    height: int,
    width: int,
    batch: int,
    seed: int = 42,
) -> Any:
    rows = [row for row in manifest["frames"] if row["split"] == split]

    if not rows:
        raise ValueError(f"Empty split {split}")
    if min(height, width, batch) < 1:
        raise ValueError("Height, width and batch must be positive")

    # Shuffle lightweight indices, rather than keeping decoded images in RAM.
    rng = np.random.default_rng(seed)
    classes = len(manifest["classes"])

    def generate() -> Any:
        indices = (
            rng.permutation(len(rows))
            if split == "train"
            else range(len(rows))
        )

        for index in indices:
            row = rows[index]

            with Image.open(root / row["image"]) as image:
                x = np.asarray(
                    image.convert("RGB").resize(
                        (width, height),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )

            with Image.open(root / row["mask"]) as image:
                y = np.asarray(
                    image.resize(
                        (width, height),
                        Image.Resampling.NEAREST,
                    ),
                    dtype=np.int32,
                )

            if y.shape != (height, width):
                raise ValueError(f"Expected an indexed mask: {row['mask']}")

            valid = y != manifest["ignore_label"]
            invalid = valid & ((y < 0) | (y >= classes))
            if np.any(invalid):
                raise ValueError(f"Unexpected class in mask: {row['mask']}")

            labels = np.where(valid, y, 0).astype(np.int32)
            weights = valid.astype(np.float32)
            yield x, labels, weights

    signature = (
        tf.TensorSpec((height, width, 3), tf.float32),
        tf.TensorSpec((height, width), tf.int32),
        tf.TensorSpec((height, width), tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        generate,
        output_signature=signature,
    )
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(rows)))

    # Keep the final batch even when it contains fewer examples.
    return ds.batch(batch, drop_remainder=False).prefetch(1)


def run(root: Path, epochs: int = 3) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")

    root = root.resolve()
    path = root / "configs/datasets/camvid.json"

    if not path.exists():
        raise FileNotFoundError(
            "Run python -m dl.data.prepare_camvid first. Never train on data/."
        )

    manifest = load_manifest(root, path)
    config = json.loads(
        (root / "configs/ines-camvid-learning.json").read_text(
            encoding="utf-8"
        )
    )

    if config["classes"] != len(manifest["classes"]):
        raise ValueError("Class map mismatch")

    seed = int(config.get("seed", 42))
    height = int(config["height"])
    width = int(config["width"])
    batch_size = 2
    learning_rate = 1e-3

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    train_ds = dataset(
        root, manifest, "train", height, width, batch_size, seed
    )
    val_ds = dataset(
        root, manifest, "val", height, width, batch_size, seed
    )

    train_batches = int(tf.data.experimental.cardinality(train_ds).numpy())
    val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())

    print(f"Training batches per epoch: {train_batches}")
    print(f"Validation batches per epoch: {val_batches}")

    # Every invocation starts a fresh experiment.
    model = build(config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="mean_with_sample_weight",
        ),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    output = root / "artifacts/ines-deeplabv3plus" / run_id
    output.mkdir(parents=True, exist_ok=False)

    print(f"Output directory: {output}")
    started = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output / "best.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.CSVLogger(str(output / "history.csv")),
            keras.callbacks.TerminateOnNaN(),
        ],
    )

    record = {
        "profile": "learning",
        "run_id": run_id,
        "config": config,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "initialization": "fresh model from config; not resumed",
        "epochs_requested": epochs,
        "epochs_completed": len(history.epoch),
        "train_batches": train_batches,
        "validation_batches": val_batches,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_hash": manifest["split_hash"],
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "seconds": time.perf_counter() - started,
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "benchmark_used_for_training": False,
    }

    record_text = json.dumps(record, indent=2)
    (output / "run.json").write_text(record_text, encoding="utf-8")

    records = root / "docs/experiments/learning"
    records.mkdir(parents=True, exist_ok=True)
    relative_output = output.relative_to(root).as_posix()

    (records / f"ines-camvid-{run_id}.md").write_text(
        "# CamVid DeepLabV3+ learning run\n\n"
        "Profile: learning. Not official benchmark evidence.\n\n"
        f"Artifacts: `{relative_output}`\n\n"
        "```json\n" + record_text + "\n```\n",
        encoding="utf-8",
    )

    print(f"Completed epochs: {len(history.epoch)}")
    print(f"Best checkpoint: {output / 'best.keras'}")
    print(f"Run record: {output / 'run.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("epochs must be positive")

    run(Path.cwd(), args.epochs)
