"""Non-executing diagnostics for externally trained Anomalib checkpoints.

Lightning checkpoints can contain Python objects.  This module deliberately
inspects their archive metadata without deserialising model state, so an
operator can establish provenance before allowing a checkpoint into the UI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointDiagnostic:
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    unsafe_globals: tuple[str, ...]


def inspect_checkpoint(path: str | Path) -> CheckpointDiagnostic:
    """Inspect a checkpoint without executing its pickle payload."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        return CheckpointDiagnostic(str(checkpoint), False, None, None, ())

    import torch

    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    unsafe = tuple(sorted(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)))
    return CheckpointDiagnostic(
        path=str(checkpoint),
        exists=True,
        size_bytes=checkpoint.stat().st_size,
        sha256=digest.hexdigest(),
        unsafe_globals=unsafe,
    )
