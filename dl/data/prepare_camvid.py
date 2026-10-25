"""Acquire the authors' CamVid tutorial release and create a sequence-safe manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REVISION = "bb68b64e72cb004b3ed927fee0f5624dd8097498"
URL = f"https://codeload.github.com/alexgkendall/SegNet-Tutorial/zip/{REVISION}"
CLASSES = [
    "sky",
    "building",
    "pole",
    "road",
    "pavement",
    "tree",
    "sign_symbol",
    "fence",
    "car",
    "pedestrian",
    "bicyclist",
]


def prepare(root: Path, archive: Path | None = None) -> Path:
    destination = root / "artifacts/camvid"
    destination.mkdir(parents=True, exist_ok=True)
    archive = archive or root / "artifacts/camvid-source.zip"
    if not archive.exists():
        print("Downloading pinned CamVid tutorial archive (about 180 MB)...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    rows = []
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        prefix = f"SegNet-Tutorial-{REVISION}/CamVid/"
        for name in sorted(names):
            relative = name.removeprefix(prefix)
            parts = relative.split("/")
            if not name.startswith(prefix) or len(parts) != 2:
                continue
            original_split, filename = parts
            if original_split not in {"train", "val", "test"} or not filename.endswith(".png"):
                continue
            mask_name = prefix + original_split + "annot/" + filename
            if mask_name not in names:
                raise ValueError(f"Missing annotation: {mask_name}")
            sequence = filename.split("_")[0]
            # Whole video sequences, never random adjacent frames across partitions.
            split = {"0001TP": "val", "0006R0": "test", "0016E5": "train", "Seq05VD": "train"}.get(
                sequence
            )
            if split is None:
                raise ValueError(f"Unknown sequence {sequence}; review before assigning a split")
            image_bytes, mask_bytes = source.read(name), source.read(mask_name)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_size = image.size
            with Image.open(io.BytesIO(mask_bytes)) as mask:
                if mask.size != image_size:
                    raise ValueError("Image/mask shape mismatch")
                values = np.unique(np.asarray(mask))
                if not set(values.tolist()) <= set(range(12)):
                    raise ValueError(f"Unexpected CamVid labels: {values}")
            target = destination / sequence
            target.mkdir(exist_ok=True)
            image_path, mask_path = target / filename, target / (filename[:-4] + "_mask.png")
            image_path.write_bytes(image_bytes)
            mask_path.write_bytes(mask_bytes)
            rows.append(
                {
                    "image": str(image_path.relative_to(root)),
                    "mask": str(mask_path.relative_to(root)),
                    "scene": sequence,
                    "split": split,
                    "original_split": original_split,
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                }
            )
    if not rows or {row["split"] for row in rows} != {"train", "val", "test"}:
        raise ValueError("Incomplete dataset; cannot create a valid training manifest")
    assignment = sorted({(row["scene"], row["split"]) for row in rows})
    manifest = {
        "source": "CamVid SegNet tutorial",
        "revision": REVISION,
        "url": URL,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "licence": "See original CamVid terms and SegNet-Tutorial licence; research use only",
        "licence_url": "https://github.com/alexgkendall/SegNet-Tutorial",
        "preprocessing_version": 1,
        "classes": CLASSES,
        "ignore_label": 11,
        "split_hash": hashlib.sha256(json.dumps(assignment).encode()).hexdigest(),
        "excluded": [],
        "normality_note": "Urban semantic corpus; not certified anomaly-free. Review scenes before official use.",
        "frames": rows,
    }
    output = root / "configs/datasets/camvid.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print({split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")})
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare(Path.cwd(), args.archive)
