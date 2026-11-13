"""A9 done-check: the real cache loads through FeatureStore (run once on CARC, CPU node).

    python scripts/a9_verify_export.py --cache $CACHE --out $TMPDIR/a9_fs --layers 33

Exports the given layers for train + seen_val with the loader's CLI (default horizon task_min_step), loads them back
through FeatureStore and checks, per split:
  * one record per episode of that split in episodes.csv, nothing from another split;
  * every episode has exactly min(length, task_min_step) steps (A9-1);
  * features are bit-identical to SharedCache.records(..., horizon="task_min_step");
  * metadata: 34 layers, 33 = block32_prenorm, the horizon is recorded.
Also prints the smallest train std per layer from norm_stats (the 1e-6 floor note in the A9 review).
unseen_test is never exported or read. Memory: about rows x K x 4096 x 4 bytes; one layer is under 1.5 GB.
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "collection"))

from failure_prob.data.feature_store import FeatureStore  # noqa: E402
from load_cache import SharedCache  # noqa: E402

DEV_SPLITS = ("train", "seen_val")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="scratch folder for the export (must not exist)")
    ap.add_argument("--layers", type=int, nargs="+", default=[33])
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--normalize", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"error: {a.out} exists; give a new scratch folder", file=sys.stderr)
        return 2
    r = subprocess.run([sys.executable, str(REPO / "collection" / "load_cache.py"), "export", "--cache", str(a.cache),
                        "--out", str(a.out), "--layers", *map(str, a.layers), "--pool", a.pool,
                        *(["--normalize"] if a.normalize else [])],
                       env={"PYTHONPATH": str(REPO), **__import__("os").environ}, text=True, capture_output=True)
    print(r.stdout, end="")
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return 1
    cache = SharedCache(a.cache)
    eps = cache.episodes.set_index("trajectory_id")
    failures = []
    for split in DEV_SPLITS:
        store = FeatureStore(a.out / split)
        meta = store.metadata()
        got = store.load(list(a.layers))
        ref = cache.records(split, list(a.layers), a.pool, normalize=a.normalize, horizon="task_min_step")
        ids = [g["trajectory_ids"][0] for g in got]
        expected = set(eps.index[eps.split == split])
        if set(ids) != expected or len(ids) != len(expected):
            failures.append(f"{split}: {len(ids)} records, episodes.csv lists {len(expected)}")
        bad_len = [t for g, t in zip(got, ids) if t in eps.index and
                   g["features"].shape[1] != min(eps.at[t, "length"], eps.at[t, "task_min_step"])]
        if bad_len:
            failures.append(f"{split}: {len(bad_len)} episodes not cut to task_min_step, e.g. {bad_len[:3]}")
        if len(got) != len(ref) or not all(torch.equal(g["features"], q["features"]) for g, q in zip(got, ref)):
            failures.append(f"{split}: FeatureStore features differ from SharedCache.records")
        if meta.num_layers != len(cache.layer_names) or "h=task_min_step" not in meta.pooling:
            failures.append(f"{split}: metadata num_layers={meta.num_layers} pooling={meta.pooling!r}")
        if len(cache.layer_names) == 34 and "33=block32_prenorm" not in meta.layer_indexing:
            failures.append(f"{split}: layer_indexing {meta.layer_indexing!r} does not name index 33")
        print(f"{split}: {len(got)} episodes, steps {sum(g['features'].shape[1] for g in got)}, "
              f"features {tuple(got[0]['features'].shape[2:])}, pooling {meta.pooling}")
    z = np.load(a.cache / "norm_stats" / f"{a.pool}.npz")
    pos = {int(l): i for i, l in enumerate(z["layers"])}
    for l in a.layers:
        std = z["std"][pos[l]]
        print(f"norm_stats L{l:02d}: min train std {std.min():.3g}, dims below 1e-3: {int((std < 1e-3).sum())}")
    print("A9 PASS: the cache loads through FeatureStore" if not failures else "A9 FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
