# Ines: DeepLabV3+ learning implementation

Implement the existing model contract using TensorFlow/Keras and an external CamVid corpus.
RoadAnomaly remains evaluation-only. No published performance or registry selection is made.
The authors' SegNet tutorial is pinned to bb68b64e72cb004b3ed927fee0f5624dd8097498.
Train/validation/test are reassigned by full source video: 0016E5 and Seq05VD / 0001TP / 0006R0.
This is a learning candidate, not a claim that CamVid is certified anomaly-free.
Validate labels, hashes and disjoint scene assignments before any fit.
11 semantic classes; void label 11 contributes zero sample weight.
Deliver model, score-map export, acquisition, guarded training, tests and client notebook.
Official experiment integration, XAI fidelity and threshold selection remain future work.
