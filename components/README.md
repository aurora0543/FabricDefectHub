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
- The corresponding adapter under `src/fabric_defect_hub/models/<name>/`
  is responsible for adding `components/<name>` to `sys.path` before
  importing anything from it, and for translating between this project's
  `Sample`/`Prediction`/`Artifact` types and whatever the vendored code
  natively uses.

Known collision risk: these repos define generic top-level module names
(`utils`, `dataset`, `models`, `optimizers`, ...) rather than a namespaced
package. Once one is imported, it occupies that name in `sys.modules` for
the rest of the process — two vendored repos that both define, say,
`utils.py` cannot be imported in the same process. Not an issue running
one backend at a time (the normal case here), but don't try to `import`
two different `components/*` backends' internals side by side without
checking for name clashes first.
