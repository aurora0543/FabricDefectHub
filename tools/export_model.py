#!/usr/bin/env python3
"""Export a registered model artifact, optionally building a TensorRT engine."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.loader import load_model
from fabric_defect_hub.models.base import Artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True, help="model variant/name")
    parser.add_argument("--artifact", required=True, help="trained checkpoint path")
    parser.add_argument("--target", required=True, help="export target, such as onnx or torchscript")
    parser.add_argument("--metadata", help="JSON file containing Artifact metadata")
    parser.add_argument("--export-config", help="JSON object passed to adapters that support config")
    parser.add_argument("--output", help="engine output path when --target tensorrt")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args(argv)

    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    artifact = Artifact(path=args.artifact, backend=args.backend, metadata=metadata)
    model = load_model(args.backend, args.model)

    if args.target == "tensorrt":
        if not args.output:
            parser.error("--output is required when --target tensorrt")
        from fabric_defect_hub.profiling.tensorrt import TensorRTBuildConfig, build_tensorrt_engine

        onnx = model.export(artifact, "onnx")
        exported = build_tensorrt_engine(onnx, args.output, TensorRTBuildConfig(precision=args.precision))
    else:
        export_config = json.loads(args.export_config) if args.export_config else {}
        if "config" in inspect.signature(model.export).parameters:
            exported = model.export(artifact, args.target, config=export_config)
        else:
            if export_config:
                raise ValueError(f"Backend {args.backend!r} does not accept export configuration.")
            exported = model.export(artifact, args.target)
    print(json.dumps(asdict(exported), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
