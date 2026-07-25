"""Append-only provenance records for trained weight artifacts."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


MANIFEST_FILENAME = "weight_manifest.jsonl"


def record_weight(
    *,
    project_root: str | Path,
    backend: str,
    variant: str | None,
    config_path: str | Path,
    resolved_config: dict[str, Any],
    registered_artifact: Any,
    published_path: str | Path | None,
    metrics: dict[str, Any] | None = None,
    run_id: str | None = None,
    model_key: str | None = None,
) -> Path:
    """Persist one immutable record after a successful training run.

    The registered artifact under ``artifacts/models/`` is the archival,
    run-specific weight. A canonical run may additionally be copied into
    ``artifacts/models/published/`` for the frontend. Both paths are kept in
    the record so overwriting the published slot never loses provenance.
    """

    root = Path(project_root)
    models_root = root / "artifacts" / "models"
    records_root = models_root / "records"
    records_root.mkdir(parents=True, exist_ok=True)

    timestamp = _timestamp()
    record_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{_slug(model_key or variant or backend)}_{uuid4().hex[:8]}"
    config_snapshot = records_root / f"{record_id}.config.json"
    _atomic_json_write(config_snapshot, _json_safe(resolved_config))

    registered_path = Path(registered_artifact.path)
    published = Path(published_path) if published_path else None
    parameter_count = registered_artifact.metadata.get("parameter_count")
    trainable_parameter_count = registered_artifact.metadata.get("trainable_parameter_count")
    primary = published or registered_path
    record = {
        "schema_version": 1,
        "record_id": record_id,
        "recorded_at": timestamp,
        "batch_run_id": run_id,
        "model_key": model_key,
        "backend": backend,
        "variant": variant,
        "config_source_path": str(Path(config_path).resolve()),
        "config_snapshot_path": str(config_snapshot),
        "artifact": {
            "primary_path": str(primary),
            "primary_location": "published" if published else "registered",
            "registered_path": str(registered_path),
            "published_path": str(published) if published else None,
            "size_bytes": registered_path.stat().st_size if registered_path.is_file() else None,
        },
        "parameters": {
            "total": int(parameter_count) if parameter_count is not None else None,
            "trainable": int(trainable_parameter_count) if trainable_parameter_count is not None else None,
            "status": "measured_during_training" if parameter_count is not None else "unavailable",
        },
        "metrics": _json_safe(metrics or {}),
        "artifact_metadata": _json_safe(registered_artifact.metadata),
    }
    manifest_path = models_root / MANIFEST_FILENAME
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path


def read_weight_manifest(project_root: str | Path) -> list[dict[str, Any]]:
    """Read valid append-only manifest records in write order."""

    path = Path(project_root) / "artifacts" / "models" / MANIFEST_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "model"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
