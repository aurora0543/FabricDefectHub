# components/

Unmodified checkouts of third-party research repos that ship a model but
not an installable package — no PyPI release, no stable API, just scripts
and `nn.Module`s (e.g. [Dinomaly](https://github.com/guojiajeremy/Dinomaly),
[MambaAD](https://github.com/lewandofskee/MambaAD)). These don't belong in
`anomalib`'s adapter tree because they aren't in `anomalib`'s model zoo and
never will be — they're the author's own code, vendored as-is.

Each one is a **git submodule** pointing at the upstream repo, not a
plain copy — `git submodule status` shows the exact commit pinned. After
cloning this project, run:

```
git submodule update --init --recursive
```

to actually populate `components/*` (a fresh clone leaves these
directories empty until then). To add another one:

```
git submodule add <upstream-url> components/<name>
```

Rules for anything placed here:

- **Never edit vendored source, and never commit inside a submodule from
  this repo's working tree.** If a fix is needed, patch it in the adapter
  that wraps it (`src/fabric_defect_hub/models/<name>/`), not here. This
  keeps `components/<name>` a clean checkout of whatever commit it's
  pinned to — `git submodule status` should never show a `+` (dirty/
  diverged) prefix.
- **Bumping the pinned commit** is deliberate: `cd components/<name> &&
  git fetch && git checkout <ref>`, then commit the resulting
  `components/<name>` pointer change in the parent repo — not an
  incidental side effect of editing files inside it.
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
