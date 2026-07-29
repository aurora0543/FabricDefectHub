# Tianchi (Guangdong 2019) split decision — frozen

The Tianchi Guangdong fabric-defect corpus ships no author train/test
split ("train1"/"train2" are release batches, not splits, and
`testA`/`testB` are unlabelled submission sets). This project therefore
defines its own split, implemented in `datasets/tianchi.py` and pinned by
`tests/test_tianchi.py`. As of 2026-07-29 this is the **frozen decision**
— any change invalidates every number ever produced on this dataset and
needs the same treatment as an interface break:

1. **Ratio: 80/20** (`train_ratio=0.8`, the adapter default). Applied
   independently to each part's (`train1-partA`, `train1-partB`,
   `train2`) normal pool and defect pool, so both splits keep every
   part's fabric styles and real bboxed defects.
2. **Assignment: sort by image filename, cut at the ratio.** Fully
   deterministic and seed-independent; `seed` only shuffles selection
   *within* an already-assigned split (`num_samples` subsetting), never
   moves an image across the boundary. No leakage is possible between
   reruns or machines.
3. **`testA`/`testB` are never read.** They carry no ground truth and are
   not guaranteed normal, so they are excluded from both the normal pool
   and any evaluation set (`test_unlabelled_test_directories_are_never_read`).
4. **Roles:** detection (native bboxes, both splits) and anomaly
   normal-pool duty via `fabric-train` (`use_defect=False`). Pixel-level
   anomaly metrics are not available here — box annotations only, by
   design (no mask synthesis from boxes).

Disjointness/coverage are pinned by `test_train_test_split_is_disjoint_
and_deterministic`, the testA/testB exclusion by
`test_unlabelled_test_directories_are_never_read`, and seed-independence
by `_split_pool` cutting the filename-sorted pool before any seeded
selection runs. If a different ratio is ever needed for an experiment,
pass `train_ratio` explicitly in that experiment's config rather than
changing the default.
