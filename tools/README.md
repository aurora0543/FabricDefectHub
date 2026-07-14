# tools

Data conversion, export, and visualization scripts (standalone, not package code):

- `convert_annotations.py` — COCO detection annotations -> `Sample` JSON.
- `export_model.py` — calls `ModelAdapter.export`; can also build an ONNX model into a TensorRT engine.
- `visualize_predictions.py` — draws a `Prediction` (boxes/masks/anomaly heatmap) on an image.

Run any of the three scripts with `python tools/<script>.py --help` to see its arguments.
