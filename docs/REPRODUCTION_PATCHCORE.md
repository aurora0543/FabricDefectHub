# Reproduction Card — PatchCore on MVTec-AD

The reproduce-then-iterate protocol: before trusting the `patchcore`
recipe on fabric data, reproduce the method's published number on the dataset
it was published against. Only once this matches do we apply the recipe to
fabric benchmarks and claim any improvement on top.

## Source (anchored)

| Field | Value |
| :--- | :--- |
| Paper | Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022 |
| Official code | `amazon-science/patchcore-inspection` |
| Reproduction impl | `anomalib` 2.5.0 `Patchcore` (we do **not** re-implement — see `models/anomalib/presets.py`) |
| Settings | WideResNet-50 backbone, features from `layer2`+`layer3`, coreset sampling ratio `0.1`, `num_neighbors=9` |

The settings are supplied by the `patchcore` recipe
(`src/fabric_defect_hub/recipes/patchcore_recipe.py`), whose hyperparameter
keys/values were reconciled to anomalib's real `Patchcore` constructor
vocabulary. Running the reproduction therefore also exercises the recipe
wiring (`ModelSpec.recipe` → `AnomalibConfig.resolved_model_kwargs`).

## Target numbers (to match)

MVTec-AD, mean over all 15 categories (anomalib-reproduced, single model):

| Metric | Target |
| :--- | :--- |
| Image AUROC | ≈ 0.991 |
| Pixel AUROC | ≈ 0.981 |
| Pixel PRO | ≈ 0.935 |

Paper abstract reports image AUROC "up to 99.6%" with ensembling; the single
model above is the ≈99.1% operating point. Per-category numbers vary (e.g.
`bottle` typically ≈ 1.000 image AUROC); the config ships with `bottle` first,
then sweep `category` across all 15 for the mean.

Sources: anomalib PatchCore model card; Roth et al., CVPR 2022.

## How to run

```bash
MVTEC_AD_ROOT=/path/to/mvtec_ad fdh run configs/models/patchcore_mvtec_repro.yaml
```

Requires the anomalib extra (`pip install -e ".[anomalib]"`) and the MVTec-AD
download. GPU recommended but not required (PatchCore is a single forward pass
+ coreset subsampling, no gradient training).

## Result log (fill in after each run)

| Date | Category | Image AUROC | Pixel AUROC | PRO | Matches target? | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| _pending_ | bottle | | | | | first reproduction run |
| _pending_ | (mean of 15) | | | | | full-benchmark sweep |

Only after the mean row matches the target within a small tolerance is the
recipe considered *reproduction-validated* for PatchCore, and only then does
applying it to fabric datasets (`raw-fabric`, `zju-leaper`, `tianchi`) and
reporting a delta become a defensible claim rather than academic theater.
