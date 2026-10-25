# Delivery validation

Commands run in the supplied ZIP checkout:

- `make lint type test BIN=/root/.local/bin`: lint/type pass; original audit suite fails because the ZIP has no Git metadata (10 failed, 23 passed, 13 errors before the additional XAI test). No original tests changed.
- `python -m ruff check .`: All checks passed.
- `python -m ruff format --check .`: all files formatted.
- `python -m mypy dl backend tests`: Success, no issues in 14 source files.
- `python -m pytest tests/test_ines_deeplab.py tests/dataset/test_dataset_integrity.py tests/architecture tests/notebooks -q`: 21 passed. Keras/NumPy deprecation warnings remain.

Real CamVid batch loading verified. Random-weight gradient update, SavedModel roundtrip, Keras roundtrip and four Seg-Grad-CAM target layers checked. No performance measurements or full training run.

CamVid source revision bb68b64e72cb004b3ed927fee0f5624dd8097498. 476 train / 124 validation / 101 test, grouped by source video. Content digests and source URL in configs/datasets/camvid.json.
