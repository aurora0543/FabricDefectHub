"""What produced a result: git state, vendored-checkout state, and the
training facilities a run actually used.

Two sinks share these helpers so their records can be compared line by line:
`reporting.append_run_log` (the evaluation-side JSONL ledger) and
`weight_registry.record_weight` (the training-side weight manifest).

`vendored_components` closes the gap C3 left open (see
`docs/INTERFACE_TASKS.md`): git submodules already pin each upstream checkout
to a commit, but that hash lived only in the repository — a result row could
not say which upstream code produced it. `describe_training` covers B4: which
optimizer / scheduler / precision a `train()` call actually ran with, read
off the live objects rather than re-declared by hand.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(cwd: str | Path | None = None) -> str:
    """The current commit hash, or "unknown" off-repo / without git."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def vendored_components(cwd: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Pinned commit and dirty-state of every vendored checkout (git
    submodule), keyed by its path (e.g. "components/moeclip").

    `git submodule status` prefixes the hash with " " (clean, at the pinned
    commit), "+" (checked out at a different commit or dirty) or "-" (not
    initialised). A "+" in a result row means the run used upstream code the
    repository does not pin — exactly what a reproduction attempt needs to
    know. Empty dict off-repo / without git / without submodules.
    """

    try:
        output = subprocess.run(
            ["git", "submodule", "status"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            cwd=cwd,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    components: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        marker, rest = line[0], line[1:].split()
        if len(rest) < 2:
            continue
        commit, path = rest[0], rest[1]
        components[path] = {
            "commit": commit,
            "state": {" ": "clean", "+": "modified", "-": "uninitialized"}.get(marker, marker),
        }
    return components


def collect_provenance(cwd: str | Path | None = None) -> dict[str, Any]:
    """The full provenance block both record sinks attach to a row."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(cwd),
        "hostname": platform.node(),
        "vendored_components": vendored_components(cwd),
    }


def describe_training(
    optimizer: Any = None,
    scheduler: Any = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """JSON-safe record of the training facilities a run used, for
    `Artifact.metadata["training"]`.

    `optimizer`/`scheduler` accept the live object (class name and — for the
    optimizer — lr / weight_decay are read off it, so the record cannot drift
    from the code) or a plain string for facilities that exist only as a
    description (Ultralytics' "auto", MambaAD's inline schedule).
    """

    record: dict[str, Any] = {"precision": precision}

    if optimizer is None:
        record["optimizer"] = None
    elif isinstance(optimizer, str):
        record["optimizer"] = optimizer
    else:
        record["optimizer"] = type(optimizer).__name__
        groups = getattr(optimizer, "param_groups", None)
        if groups:
            record["lr"] = groups[0].get("lr")
            if "weight_decay" in groups[0]:
                record["weight_decay"] = groups[0]["weight_decay"]

    if scheduler is None:
        record["scheduler"] = None
    elif isinstance(scheduler, str):
        record["scheduler"] = scheduler
    else:
        record["scheduler"] = type(scheduler).__name__

    return record
