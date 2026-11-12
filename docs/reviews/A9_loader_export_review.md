# A9: review of the loader export (`collection/load_cache.py`, PR #3)

Reviewer: Dhyan. Scope: `SharedCache.records` and `load_cache.py export`. Checked: layer keys 0–33, train-only normalization, the unseen lock.
Tests: `tests/test_a9_loader_export.py` (run with `python -m pytest -q tests/test_a9_loader_export.py`). Data: a synthetic 34-layer cache built by the real `build_cache.py`. Nothing in this review was run on the CARC cache.

## Verdict

The loader reads the right data. Every layer key 0–33 returns its own layer for every pooling and both dev splits. Normalization uses train-only statistics, applied per requested layer in any order. The unseen lock holds for `records` and `export`.

Three issues are still open, and A9-1 matters before anyone reports numbers from an export. Each one has a `xfail(strict=True)` test. When a fix lands, its test will XPASS and fail the run. Remove that test's marker at that point.

## Verified (passing tests)

| Property | Test |
|---|---|
| Keys 0..33 read `L00..L33` for mean/first/last in train and seen_val, bit-exact against the source `.layers.pt` (rows taken from inside the split, which checks `row_start`). `layer_names[32]` is `block32_postnorm`, `[33]` is `block32_prenorm`, and the two tensors differ. | `test_every_layer_key_reads_its_own_layer` |
| Requested order is kept (`[33, 0, 16]`). | `test_requested_layer_order_is_kept` |
| Keys outside 0..33, including -1, are refused. | `test_out_of_range_layers_refused` |
| Export metadata: `num_layers=34`, `33=block32_prenorm` is in `layer_indexing`, `FeatureStore.load([33, 0])` re-indexes by key, and unexported or out-of-range keys are refused. | `test_export_layer_keys_and_metadata` |
| `norm_stats` are computed on train rows only (`n_rows` = train rows), per layer, and match the train `.npy` mean and std. | `test_norm_stats_are_train_only_per_layer` |
| `normalize=True` equals `(x - mean[l]) / std[l]` for unsorted layer lists, on train and seen_val. Normalized train data has mean 0. | `test_normalize_uses_the_right_layer_stats_in_any_order` |
| `export --normalize` equals `records(normalize=True)`, and `pooling="mean+trainnorm"`. | `test_normalized_export_matches_records` |
| Unseen lock: `records` raises `PermissionError` (also with `normalize`/`horizon`), and `export --splits unseen_test` fails. The default export holds only train and seen_val, and no exported ID is from tasks 0/3/5. `allow_unseen=True` still works. | `test_unseen_lock` |
| The `episodes.csv` split column equals `splits.json`, and no trajectory is in two splits. | `test_episodes_csv_agrees_with_splits_json` |
| `records(horizon="task_min_step")` cuts every episode to `task_min_step`. | `test_task_min_step_horizon_in_records` |

## Open findings

**A9-1 (high): `export` always writes full-length episodes.** `export()` calls `cache.records(...)` without `horizon`, so it gets the default `"full"`, and the CLI has no `--horizon` option. Failures always run to 520 steps, so any model trained or scored on an export can read the label from episode length. The handoff gives the export command as an alternative to `records(..., horizon="task_min_step")`, which makes this easy to hit. The metadata does not record the horizon either, so nobody can tell afterwards.
Suggested fix: add `--horizon {task_min_step,full}` (default `task_min_step`), pass it through to `records`, and record it in the metadata, e.g. `pooling=f"{pool}{'+trainnorm' if normalize else ''}+h={horizon}"`, the same way normalization is recorded.
Test: `test_export_can_write_task_min_step_horizon`.

**A9-2 (medium): unknown horizons are accepted silently.** `stop = ep.row_end if horizon == "full" else ...` treats any other string as `task_min_step`. A typo like `"task-min-step"` happens to do the right thing, but `"min"` or `"full "` with a trailing space also run without an error.
Suggested fix: `if horizon not in ("full", "task_min_step"): raise ValueError(...)`.
Test: `test_unknown_horizon_refused`.

**A9-3 (low): duplicate layer keys are accepted.** `records(layers=[33, 33])` returns K=2 copies of the same layer, and export would write `layers=(33, 33)`. A fusion model would then train on a duplicated layer without any error.
Suggested fix: refuse duplicates next to the existing range check.
Test: `test_duplicate_layers_refused`.

## Notes (no change needed, worth knowing)

- **Normalization statistics cover the full horizon.** They are computed over full-length train episodes, while results use `task_min_step`. That is fine: it is still train-only and the same for every model. It just means normalized train features at the reporting horizon are not exactly mean 0.
- **The `std` floor is 1e-6, and this does happen.** A dimension that is constant on train but varies on seen_val gets scaled by about 1e6. In the synthetic fixture, layer 16 has dimensions whose values round to the same bf16 number on every train row (std = 0), and normalized seen_val values reach 2,000,000. Before the real sweep, check `min(std)` per layer in `norm_stats/*.npz` on the real cache. A larger floor, e.g. `max(std, 1e-3 * median(std))`, would be safer. `fusion_runner` prints a warning when normalized features exceed |x| > 1e3.
- **Missing layers in `norm_stats`.** If a cache was built with `--layers` (a subset), `norm_stats()` raises a bare `KeyError` for a missing layer instead of a readable message.
- **Pool names and case.** On macOS (case-insensitive file system), `pool="Mean"` silently reads `mean/`. On CARC (Linux) it raises `FileNotFoundError`. Validating `pool` against `manifest["pools_exported"]` would make this the same everywhere.
- **Memory.** `export` holds every record in memory as float32. That is about 1.4 GB per layer for train at full horizon, so a 4-layer export needs about 10 GB of RAM.

`failure_prob/fusion_runner.py` (F3) does not depend on these fixes. It calls `records` directly, validates `horizon` and duplicate layers itself, and checks that every episode has exactly the expected number of steps.
