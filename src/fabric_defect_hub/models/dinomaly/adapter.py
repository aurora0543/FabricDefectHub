"""`ModelAdapter` implementation wrapping the vendored `components/dinomaly`
checkout (see `components/README.md` and `vendor.py`).

Dinomaly ships as research scripts, not a library: no train/predict CLI, no
checkpoint saving (commented out in upstream), and no combined
"model produces a score" call -- the anomaly map is computed by a free
function (`utils.cal_anomaly_maps`) applied to the encoder/decoder feature
lists the model returns. This adapter re-implements upstream's reference
training loop (`dinomaly_mvtec_sep.py::train()`) and scoring path
(`utils.evaluation_batch`) against this project's `Sample`/`Prediction`/
`Artifact` contracts, loading only the vendored `nn.Module` and utility
functions -- never anomalib-style, because there is no anomalib-style API
to call.

Checkpoints are plain `state_dict`s (`torch.save`/`torch.load`, no custom
pickle globals), so unlike the anomalib adapter's Lightning checkpoints,
there's no "untrusted checkpoint" trust gate needed here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter
from fabric_defect_hub.models.dinomaly import presets
from fabric_defect_hub.models.dinomaly.vendor import ensure_on_path


@register_model("dinomaly")
class DinomalyAdapter(ModelAdapter):
    """Wraps upstream's `ViTill` (DINOv2 encoder + bottleneck + decoder).

    `name` selects the encoder preset (see `presets.ENCODER_PRESETS`),
    e.g. 'dinov2reg_vit_base_14' (default), '..._small_14', '..._large_14'.
    """

    backend = "dinomaly"

    def __init__(self, name: str = presets.DEFAULT_ENCODER_NAME, **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder_name = presets.resolve_encoder_name(name)
        self._model = None
        self._loaded_path: str | None = None

    # ------------------------------------------------------------------ #
    # Model construction, shared by train() and _load_artifact()
    # ------------------------------------------------------------------ #
    def _resolve_device(self, requested: str | None = None):
        import torch

        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self, encoder_name: str, target_layers: list[int]):
        """Build an untrained `ViTill` exactly like upstream's `train()`:
        a DINOv2 encoder (downloads/caches upstream's public weights on
        first use), a single-layer MLP bottleneck, and an 8-block
        linear-attention decoder.
        """

        ensure_on_path()
        import torch.nn as nn
        from functools import partial

        from dinov1.utils import trunc_normal_
        from models import vit_encoder
        from models.uad import ViTill
        from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp

        preset = presets.encoder_preset(encoder_name)
        embed_dim, num_heads = preset["embed_dim"], preset["num_heads"]

        encoder = vit_encoder.load(encoder_name)

        bottleneck = nn.ModuleList(
            [bMlp(embed_dim, embed_dim * presets.BOTTLENECK_HIDDEN_RATIO, embed_dim,
                  drop=presets.BOTTLENECK_DROPOUT)]
        )
        decoder = nn.ModuleList(
            [
                VitBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.0,
                    attn=LinearAttention2,
                )
                for _ in range(presets.DECODER_DEPTH)
            ]
        )

        model = ViTill(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder,
            target_layers=target_layers, mask_neighbor_size=0,
            fuse_layer_encoder=presets.FUSE_LAYER_ENCODER,
            fuse_layer_decoder=presets.FUSE_LAYER_DECODER,
        )

        trainable = nn.ModuleList([bottleneck, decoder])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        return model, trainable

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def train(self, config: dict[str, Any]) -> Artifact:
        """Re-implements `dinomaly_mvtec_sep.py::train()`'s loop against this
        project's data. Two ways to point this at data, same as the
        anomalib adapter:

        - `config['train_samples']` + `config['test_samples']`: raw
          `Sample` lists (`train_samples` all-normal, e.g.
          `ZJULeaperDataset(..., use_defect=False)`); staged via the same
          MVTec-style symlink layout the anomalib adapter uses (Dinomaly's
          own `ImageFolder` + `MVTecDataset` loaders expect the identical
          `train/good`, `test/good`, `test/defect`, `ground_truth/defect`
          layout).
        - `config['data_root']`: an existing MVTec-style folder, used as-is.

        Other keys (all optional, default from `presets.DEFAULT_TRAIN_KWARGS`):
        `total_iters`, `batch_size`, `image_size`, `crop_size`, `lr`,
        `final_lr`, `warmup_iters`, `weight_decay`, `hm_percent_final`,
        `hm_percent_warmup_iters`, `hm_factor`, `grad_clip_max_norm`,
        `device`, `num_workers`, `work_dir` (checkpoint destination;
        defaults to a fresh temp dir -- pass this, or
        `register_trained_model()` afterwards, to keep the checkpoint
        past process exit).
        """

        ensure_on_path()
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        from dataset import MVTecDataset, get_data_transforms
        from optimizers import StableAdamW
        from utils import WarmCosineScheduler, global_cosine_hm_percent

        kwargs = {**presets.default_train_kwargs(), **{k: v for k, v in config.items() if k in presets.DEFAULT_TRAIN_KWARGS}}
        device = self._resolve_device(config.get("device"))
        num_workers = config.get("num_workers", 4)

        preset = presets.encoder_preset(self.encoder_name)
        target_layers = config.get("target_layers", preset["target_layers"])

        data_transform, gt_transform = get_data_transforms(kwargs["image_size"], kwargs["crop_size"])

        def _run(data_root: str) -> Artifact:
            train_data = ImageFolder(root=str(Path(data_root) / "train"), transform=data_transform)
            test_data = MVTecDataset(root=data_root, transform=data_transform, gt_transform=gt_transform, phase="test")
            if len(train_data) == 0:
                raise ValueError(f"no training samples found under {data_root}/train")
            # Clamp batch_size to the dataset size and only drop the last
            # partial batch when a full batch is still guaranteed elsewhere
            # (len > batch_size). Upstream's own scripts always run at a
            # scale where batch_size < dataset size, so drop_last=True is
            # safe there; but with drop_last=True unconditionally, a
            # low-shot/smoke-test train split smaller than batch_size
            # produces zero batches per epoch -- and since the loop below is
            # `while it < total_iters: for batch in loader: ...`, an empty
            # loader means `it` never advances and training hangs forever.
            effective_batch_size = min(kwargs["batch_size"], len(train_data))
            train_loader = DataLoader(train_data, batch_size=effective_batch_size, shuffle=True,
                                       num_workers=num_workers,
                                       drop_last=len(train_data) > effective_batch_size)

            model, trainable = self._build_model(self.encoder_name, target_layers)
            model = model.to(device)

            optimizer = StableAdamW([{"params": trainable.parameters()}], lr=kwargs["lr"],
                                     betas=(0.9, 0.999), weight_decay=kwargs["weight_decay"],
                                     amsgrad=True, eps=1e-8)
            lr_scheduler = WarmCosineScheduler(optimizer, base_value=kwargs["lr"],
                                                final_value=kwargs["final_lr"],
                                                total_iters=kwargs["total_iters"],
                                                warmup_iters=kwargs["warmup_iters"])

            it = 0
            total_iters = kwargs["total_iters"]
            p_final = kwargs["hm_percent_final"]
            hm_warmup = kwargs["hm_percent_warmup_iters"]
            model.train()
            while it < total_iters:
                for img, _label in train_loader:
                    img = img.to(device)
                    en, de = model(img)
                    p = min(p_final * it / hm_warmup, p_final)
                    loss = global_cosine_hm_percent(en, de, p=p, factor=kwargs["hm_factor"])

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=kwargs["grad_clip_max_norm"])
                    optimizer.step()
                    lr_scheduler.step()

                    it += 1
                    if it >= total_iters:
                        break

            work_dir = Path(config.get("work_dir") or tempfile.mkdtemp(prefix="fdh_dinomaly_"))
            work_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = work_dir / f"dinomaly_{self.encoder_name}.pth"
            torch.save(model.state_dict(), ckpt_path)

            return Artifact(
                path=str(ckpt_path),
                backend=self.backend,
                metadata={
                    "model_class": "ViTill",
                    "encoder_name": self.encoder_name,
                    "target_layers": target_layers,
                    "image_size": kwargs["image_size"],
                    "crop_size": kwargs["crop_size"],
                    "trusted": True,
                },
            )

        train_samples = config.get("train_samples")
        test_samples = config.get("test_samples")
        if train_samples is not None and test_samples is not None:
            self._validate_test_masks(test_samples)
            with anomalib_folder_staging_dir(train_samples, test_samples) as layout:
                return _run(str(layout.root))
        return _run(config["data_root"])

    @staticmethod
    def _validate_test_masks(test_samples: list[Sample]) -> None:
        """Require the one-to-one anomaly masks expected by Dinomaly's loader."""

        missing = [
            sample.id
            for sample in test_samples
            if sample.annotations.is_anomalous
            and (
                not sample.annotations.anomaly_mask
                or not Path(sample.annotations.anomaly_mask).is_file()
            )
        ]
        if missing:
            preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                "Dinomaly requires a readable pixel mask for every defective test sample; "
                f"missing masks for {len(missing)} sample(s): {preview}{suffix}. "
                "Use a dataset/task selection that attaches anomaly_mask paths."
            )

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(
        self, samples: list[Sample], artifact: Artifact, output_dir: str | None = None
    ) -> list[Prediction]:
        """Loads the checkpoint, runs upstream's forward pass, and scores it
        with upstream's own `cal_anomaly_maps` -- the same math
        `evaluation_batch` uses to produce the paper's reported metrics.
        Image-level score = max of the (Gaussian-smoothed) anomaly map, i.e.
        `max_ratio=0` in upstream's `evaluation_batch`. Pass `output_dir` to
        also persist each sample's anomaly map as a `.npy` file (needed for
        `evaluation.anomaly.AnomalyEvaluator`'s pixel-level metrics).
        """

        ensure_on_path()
        import numpy as np
        import torch
        from PIL import Image

        from dataset import get_data_transforms
        from utils import cal_anomaly_maps, get_gaussian_kernel

        device = self._resolve_device(None)
        model = self._load_artifact(artifact, device)

        crop_size = artifact.metadata.get("crop_size", presets.DEFAULT_TRAIN_KWARGS["crop_size"])
        image_size = artifact.metadata.get("image_size", presets.DEFAULT_TRAIN_KWARGS["image_size"])
        data_transform, _ = get_data_transforms(image_size, crop_size)
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

        maps_dir = None
        if output_dir is not None:
            maps_dir = Path(output_dir)
            maps_dir.mkdir(parents=True, exist_ok=True)

        predictions = []
        with torch.no_grad():
            for sample in samples:
                img = Image.open(sample.image_path).convert("RGB")
                img = data_transform(img).unsqueeze(0).to(device)

                en, de = model(img)
                anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
                anomaly_map = gaussian_kernel(anomaly_map)

                score = float(anomaly_map.flatten(1).max(dim=1)[0].item())

                anomaly_map_path = None
                if maps_dir is not None:
                    arr = anomaly_map[0, 0].detach().cpu().numpy()
                    map_path = maps_dir / f"{sample.id}.npy"
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(map_path, arr)
                    anomaly_map_path = str(map_path)

                predictions.append(
                    Prediction(
                        sample_id=sample.id,
                        anomaly_score=score,
                        anomaly_map=anomaly_map_path,
                    )
                )
        return predictions

    def export(self, artifact: Artifact, target: str) -> ExportedArtifact:
        raise NotImplementedError(
            "Dinomaly export is not implemented: ViTill's forward pass has "
            "data-dependent Python control flow (per-layer no_grad toggling, "
            "list-based feature fusion) that hasn't been verified to trace "
            "cleanly through torch.onnx.export. Add and verify this "
            "explicitly before relying on it."
        )

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        import shutil

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing checkpoint: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        filename = model_name or f"dinomaly_{artifact.metadata.get('encoder_name', self.encoder_name)}.pth"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(self, artifact_or_path: Artifact | str) -> Artifact:
        """Load a previously registered/trained checkpoint back into this
        adapter. If given a bare path, architecture defaults to this
        adapter's `encoder_name`/preset `target_layers` -- pass an
        `Artifact` (with `metadata['encoder_name']`/`['target_layers']`)
        if the checkpoint was trained with different ones.
        """

        if isinstance(artifact_or_path, Artifact):
            artifact = artifact_or_path
        else:
            path = artifact_or_path
            if not Path(path).exists():
                raise FileNotFoundError(f"cannot load missing checkpoint: {path}")
            preset = presets.encoder_preset(self.encoder_name)
            artifact = Artifact(
                path=str(path), backend=self.backend,
                metadata={
                    "model_class": "ViTill",
                    "encoder_name": self.encoder_name,
                    "target_layers": preset["target_layers"],
                    "trusted": True,
                },
            )
        if not Path(artifact.path).exists():
            raise FileNotFoundError(f"cannot load missing checkpoint: {artifact.path}")
        self._load_artifact(artifact, self._resolve_device(None))
        return artifact

    def unload(self) -> None:
        self._model = None
        self._loaded_path = None

    def _load_artifact(self, artifact: Artifact, device):
        if self._model is None or self._loaded_path != artifact.path:
            import torch

            encoder_name = artifact.metadata.get("encoder_name", self.encoder_name)
            target_layers = artifact.metadata.get("target_layers", presets.encoder_preset(encoder_name)["target_layers"])
            model, _trainable = self._build_model(encoder_name, target_layers)
            state_dict = torch.load(artifact.path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            self._model = model
            self._loaded_path = artifact.path
        return self._model
