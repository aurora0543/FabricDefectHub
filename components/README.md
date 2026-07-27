# components/

Research repos that ship a model but not an installable package — no PyPI
release, no stable API, just scripts and `nn.Module`s. These don't belong
in `anomalib`'s adapter tree because they aren't in `anomalib`'s model zoo
and never will be — they're the author's own code, kept here.

Each one is a **git submodule pointing at our own fork** of the upstream
repo (e.g. `components/dinomaly` -> `aurora0543/Dinomaly`, forked from
[guojiajeremy/Dinomaly](https://github.com/guojiajeremy/Dinomaly)) — not
upstream directly, and not a plain copy. `git submodule status` shows the
exact commit pinned. After cloning this project, run:

```
git submodule update --init --recursive
```

to actually populate `components/*` (a fresh clone leaves these
directories empty until then).

**Why a fork instead of upstream directly:** these repos are written as
one-off scripts, not libraries, and sometimes hardcode paths that assume
they're being run from their own repo root (e.g. Dinomaly's
`models/vit_encoder.py` used to hardcode `"backbones/weights"` as a
*relative* path — every process that imported it dumped ~400MB of
DINOv2 weights whereever that process's cwd happened to be, which was
this project's own root, not the submodule). A fork gives us a legitimate
place to patch exactly that kind of thing (see `aurora0543/Dinomaly`'s
own commit history) without violating "never edit vendored source in
`components/`" — the fork *is* the vendored source at that point, edited
deliberately and diffably against its own upstream.

To add another one:

```
gh repo fork <upstream-owner>/<repo> --clone=false   # creates <you>/<repo>
git submodule add https://github.com/<you>/<repo>.git components/<name>
```

Rules for anything placed here:

- **Never edit files directly under `components/<name>` from this repo's
  working tree, and never commit there without deliberately intending to
  patch the fork.** Any fix belongs in a commit on the fork itself (`cd
  components/<name> && git commit && git push origin <branch>`, then bump
  the pointer in the parent repo — see below) — not an incidental side
  effect of running the model, and not done from the adapter's side.
  `git submodule status` should never show a `+` (dirty/diverged) prefix
  from *uncommitted* changes; a deliberate fork patch is the one case
  where the pinned commit is expected to move.
- **Bumping the pinned commit** (whether picking up an upstream update via
  the fork, or landing a new patch on the fork): `cd components/<name> &&
  git fetch && git checkout <ref>`, then commit the resulting
  `components/<name>` pointer change in the parent repo.
- **One subdirectory per repo**, named after the repo (`components/dinomaly/`).
- **Upstream module names may appear only in the backend's `vendor.py`.**
  Declare the checkout as a `core.vendor.VendoredRepo` there; everywhere
  else, go through `import_vendor()["models.uad"].ViTill`. A bare
  `from utils import ...` in an adapter resolves to whichever vendored repo
  loaded first. `tests/test_vendor_boundary.py` parses every source file
  with `ast` and fails on a violation.
- The corresponding adapter under `src/fabric_defect_hub/models/<name>/`
  is responsible for translating between this project's
  `Sample`/`Prediction`/`Artifact` types and whatever the vendored code
  natively uses.

Not everything that could plausibly live here does. `models/mambaad/`
(see its own module docstring) is a clean-room reimplementation, not a
submodule: the official repo is a plugin that only runs inside a second,
larger framework (`ADer`) it doesn't ship, and its selective-scan core
needs a CUDA-only compiled kernel (`mamba_ssm`) that won't install without
a matching CUDA toolchain. Vendoring it here would mean vendoring ADer
too — a general-purpose framework, not a single model's own code, which
would break "one subdirectory per repo" below — and would gate the
backend to a CUDA host exactly as hard as the CUDA-only kernel already
does on its own. When a target repo isn't runnable on its own (needs a
second framework present to import, needs a compiled extension with no
portable fallback), reimplementing the published architecture directly
against this project's contracts, the way `models/mambaad/` does, is the
better fit than forcing it through this vendoring convention.

Known collision risk: these repos define generic top-level module names
(`utils`, `dataset`, `models`, `optimizers`, ...) rather than a namespaced
package. Once one is imported, it occupies that name in `sys.modules` for
the rest of the process — two vendored repos that both define, say,
`utils.py` would otherwise shadow each other. This is no longer
hypothetical: `components/dinomaly` and `components/moeclip` both ship a
top-level `utils` and `dataset`, and the Benchmark tab runs every model
back to back in one process.

The resolution is `core/vendor.py::VendoredRepo`, used by *every* vendored
checkout — not just the later one. It imports a repo's modules inside a
window where the checkout is `sys.path[0]` and any colliding name is
temporarily evicted from `sys.modules`, then takes its own modules back
out, restores what was there before, removes its `sys.path` entry, and
keeps the imported modules in a private cache. Afterwards the two repos'
`utils` modules are two distinct objects and neither occupies the shared
name.

Dinomaly originally used a plain permanent `sys.path` bootstrap, which
worked only because it happened to load first. That was the last real
hazard before the interface freeze, and it is gone: both checkouts now go
through `VendoredRepo`, and a new one is three declarative lines
(`owned_roots`, `entry_modules`, a not-found hint).

The mechanism relies on the vendored code not importing its own top-level
names lazily from inside a function body — by then the name is out of
`sys.modules`. Verified against both pinned commits (Dinomaly's only such
imports are in `dinov2/run/*`, DINOv2's SLURM job launchers, which nothing
here touches). Check this when bumping a pinned commit.
