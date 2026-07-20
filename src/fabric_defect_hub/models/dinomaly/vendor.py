"""Bootstraps imports from the vendored `components/dinomaly` checkout.

Dinomaly ships as flat top-level modules (`models.uad`, `dinov1`, `dinov2`,
`beit`, `optimizers`, `utils`, `dataset`) meant to be run from its own repo
root, not as an installable package -- there is no `pip install dinomaly`.
Rather than editing its source to add a package prefix, we add
`components/dinomaly` to `sys.path` and import it by its native module
names. See `components/README.md` for the general vendoring convention and
the resulting name-collision caveat (`utils`, `dataset`, `models` are
generic names that occupy `sys.modules` once imported).
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "components" / "dinomaly"


def ensure_on_path() -> None:
    if not _VENDOR_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dinomaly vendor checkout not found at {_VENDOR_ROOT}. "
            "Expected the unmodified upstream repo under components/dinomaly "
            "(see components/README.md)."
        )
    path_str = str(_VENDOR_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
