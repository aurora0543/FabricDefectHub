# frontend

FabricDefectHub's front end **is** the Gradio app in
`src/fabric_defect_hub/web/` — there is no separate SPA, and that is a
decision rather than an omission: a second front end would either duplicate
the `Sample` / `Prediction` / `ExperimentResult` contracts or drift from
them. The pages consume those contracts directly, through the same
`ModelCapabilities` / `core.availability` / `core.checkpoint` seams the CLI
and SDK use (the rules are pinned by `tests/test_web_layering.py`).

This directory previously also held an empty `src/pages/*` skeleton for that
never-built SPA. It has been removed — six empty directories asserting a plan
that the paragraph above explicitly rejects.

## Launch

```bash
pip install -r requirements.txt   # lean UI/inference set
fdh-ui                            # or: python app.py
```

`app.py` at the repository root is the Hugging Face Spaces entry point and
launches the same app.

## What you see on a fresh clone

Every model shows **🟠 Checkpoint missing**. That is expected: checkpoints
are produced by `fdh train`, not shipped in git (`/artifacts/` is
gitignored). Run `fdh doctor` to see which backends are trainable on this
machine, then `fdh train <model>` — the result is published to
`artifacts/models/published/` and the page picks it up on refresh. See
`docs/cloud_training_runbook.md` for the full path.

Datasets work the same way: stage them under `data/<Dataset>` (usually a
symlink onto external storage) and the sampler finds them.
