# frontend

FabricDefectHub uses Gradio to provide a deployable model workspace, rather
than building a separate mock frontend disconnected from the backend. Pages
consume the unified `Sample`, `Prediction`, and `ExperimentResult` contracts
directly.

## Launch

```bash
pip install -e ".[ui]"
fdh-ui
```

Hugging Face Spaces can use the repository root's `app.py` directly. After
setting `ZJU_LEAPER_ROOT`, the Single Image Detection page randomly samples
a set of local images from a chosen ZJU-Leaper split; use the left/right
buttons to browse, and run real inference according to the selected
model/weights.

## Inference Sessions

The single-image page doesn't recreate the model on every inference click.
First pick an artifact from the local model dropdown, then click
**Load model** — this action, via the backend `InferenceSessionManager`,
keeps the model resident on an auto-selected CUDA, Apple MPS, or CPU device.
The page shows model parameter/buffer memory usage, current process RSS,
and CUDA/MPS allocated memory (where the platform supports it).
**Unload model** releases the active model and accelerator cache.

The UI only calls the backend's `load`, `predict`, and `unload` interfaces,
so the same session mechanism can be reused by Gradio, the CLI, a server
API, or another platform's UI; the UI never holds a framework model object
directly.

## Current Pages

- **Single Image Detection**: dataset random sampling, image browsing,
  model status, checkpoint/pretrained selection, and bbox/mask/anomaly-map
  result display.
- **Benchmark**: a standalone placeholder for now, to be implemented later
  by reusing the backend's saved leaderboard and `ExperimentResult`
  artifacts.
