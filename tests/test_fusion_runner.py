"""F3: fusion runner on train/seen_val. Uses a tiny 34-layer cache built by the real build_cache.py (no GPU)."""
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from test_kit_layer34 import build, rollouts34
from test_shared_cache import D, REPO
import load_cache
from failure_prob import fusion_runner as FR
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.protocol import TRAIN_SEEDS


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fusion_cache")
    rollouts34(tmp / "raw")
    r = build(tmp / "raw", tmp / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    return tmp / "c" / "v1"


def cfg(out, **kw):
    base = dict(model="static_weight_fusion", layers=[16, 33], output_dir=str(out), epochs=2, batch_size=4)
    return FR.RunConfig(**(base | kw))


# --------------------------------------------------------------------------- end to end on the cache

def test_cache_run_end_to_end_and_never_reads_unseen(cache, tmp_path, monkeypatch):
    seen = []
    orig = load_cache.SharedCache._array
    monkeypatch.setattr(load_cache.SharedCache, "_array",
                        lambda self, pool, split, layer: (seen.append(split), orig(self, pool, split, layer))[1])
    out = tmp_path / "run"
    summary = FR.run(cfg(out, cache=str(cache)))
    assert set(seen) == {"train", "seen_val"}
    assert summary["seeds"] == list(TRAIN_SEEDS) and not summary["synthetic"]
    assert summary["split_label"] == "single fixed split (SAFE seed-0 unseen tasks)"
    assert len(summary["metrics"]["layer_weights"]["mean"]) == 2
    conf = json.loads((out / "config.json").read_text())
    assert conf["provenance"]["layer_names"] == ["block16", "block32_prenorm"]
    assert "B4" in conf["provisional"]["episode_scorer"] and "B6" in conf["provisional"]["probe"]

    eps = pd.read_csv(cache / "episodes.csv")
    val = eps[eps.split == "seen_val"].set_index("trajectory_id")
    for s in TRAIN_SEEDS:
        m = json.loads((out / f"seed{s}" / "metrics.json").read_text())
        assert m["seen_val"]["n_episodes"] == len(val) and 0 <= m["seen_val"]["roc_auc"] <= 1
        assert set(m["seen_val_per_task"]) == {str(t) for t in val.task_id.unique()}
        scores = pd.read_csv(out / f"seed{s}" / "seen_val_episode_scores.csv")
        assert set(scores.trajectory_id) == set(val.index)
        assert (scores.n_steps.values == val.loc[scores.trajectory_id, "task_min_step"].values).all()  # horizon
        assert scores.prob.between(0, 1).all() and np.isfinite(scores.score).all()
        assert not set(scores.trajectory_id) & set(eps[eps.split == "unseen_test"].trajectory_id)
        ck = torch.load(out / f"seed{s}" / "model.pt", weights_only=False)
        assert ck["seed"] == s and ck["model_kwargs"]["num_layers"] == 2


def test_same_seed_is_reproducible(cache, tmp_path):
    a = FR.run(cfg(tmp_path / "a", cache=str(cache), seeds=[1], save_checkpoints=False))
    b = FR.run(cfg(tmp_path / "b", cache=str(cache), seeds=[1], save_checkpoints=False))
    assert a["metrics"]["roc_auc"]["per_seed"] == b["metrics"]["roc_auc"]["per_seed"]
    sa = pd.read_csv(tmp_path / "a" / "seed1" / "seen_val_episode_scores.csv")
    sb = pd.read_csv(tmp_path / "b" / "seed1" / "seen_val_episode_scores.csv")
    assert np.allclose(sa.score, sb.score)


@pytest.mark.parametrize("model, extra", [("concat_fusion", {"head": "linear"}),
                                          ("static_weight_fusion", {"learn_weights": False}),
                                          ("single_layer_mlp", {})])
def test_other_models_and_options(cache, tmp_path, model, extra):
    layers = [33] if model == "single_layer_mlp" else [0, 16, 33]
    s = FR.run(cfg(tmp_path / "r", cache=str(cache), model=model, layers=layers, seeds=[0],
                   select_best_epoch=True, aggregation="mean", save_checkpoints=False, **extra))
    m = json.loads((tmp_path / "r" / "seed0" / "metrics.json").read_text())
    assert m["selection"] == "best seen_val ROC-AUC" and 1 <= m["chosen_epoch"] <= 2
    if extra.get("learn_weights") is False:
        assert np.allclose(m["layer_weights"], 1 / 3)  # frozen uniform mix
    assert not math.isnan(s["metrics"]["roc_auc"]["mean"])


def test_full_horizon_is_longer_and_warned(cache, tmp_path, capsys):
    FR.run(cfg(tmp_path / "f", cache=str(cache), horizon="full", seeds=[0], save_checkpoints=False))
    assert "not reportable" in capsys.readouterr().err
    scores = pd.read_csv(tmp_path / "f" / "seed0" / "seen_val_episode_scores.csv")
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    assert (scores.n_steps.values == eps.loc[scores.trajectory_id, "length"].values).all()


def test_cli_on_cache(cache, tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_fusion.py"), "--model", "concat_fusion",
                        "--layers", "4", "33", "--cache", str(cache), "--output-dir", str(tmp_path / "cli"),
                        "--epochs", "1", "--seeds", "2", "--pool", "last", "--no-normalize"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    conf = json.loads((tmp_path / "cli" / "config.json").read_text())["config"]
    assert conf["pool"] == "last" and conf["normalize"] is False and conf["seeds"] == [2]


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("kw, msg", [
    ({"layers": [16, 16]}, "duplicate layers"),
    ({"layers": []}, "at least one layer"),
    ({"layers": [34]}, "not in this cache"),
    ({"layers": [-1]}, "not in this cache"),
    ({"horizon": "task-min-step"}, "horizon must be one of"),
    ({"aggregation": "median"}, "aggregation must be one of"),
    ({"seeds": [0, 0]}, "seeds must be"),
    ({"model": "nope"}, "Unknown model"),
    ({"epochs": 0}, "epochs must be"),
    ({"init_weights": [1.0]}, "init_weights has 1 values"),
])
def test_bad_config_refused(cache, tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        FR.run(cfg(tmp_path / "x", cache=str(cache), **kw))


def test_needs_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one data source"):
        FR.run(cfg(tmp_path / "x"))
    with pytest.raises(ValueError, match="exactly one data source"):
        FR.run(cfg(tmp_path / "x", cache="c", synthetic=True))


def test_refuses_to_overwrite_results(tmp_path):
    out = tmp_path / "o"
    FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1))
    with pytest.raises(FileExistsError, match="overwrite"):
        FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1))
    FR.run(cfg(out, synthetic=True, seeds=[1], epochs=1, overwrite=True))
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["seed1"]  # stale seed0 removed


def test_overwrite_only_deletes_our_own_run_folders(tmp_path, monkeypatch):
    foreign = tmp_path / "notes"
    foreign.mkdir()
    (foreign / "thesis.tex").write_text("keep me")
    with pytest.raises(FileExistsError, match="refusing to delete"):
        FR.run(cfg(foreign, synthetic=True, seeds=[0], epochs=1, overwrite=True))
    assert (foreign / "thesis.tex").read_text() == "keep me"

    out = tmp_path / "run"
    FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1))
    (out / "seed0" / "my_analysis.ipynb").write_text("{}")  # inside a seed folder: still ours to replace
    (out / "plot.png").write_bytes(b"x")  # user file next to our outputs -> refuse
    with pytest.raises(FileExistsError, match="refusing to delete"):
        FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1, overwrite=True))
    assert (out / "plot.png").exists()

    monkeypatch.chdir(out)  # cwd is protected even if it is a run folder
    (out / "plot.png").unlink()
    with pytest.raises(FileExistsError, match="refusing to delete"):
        FR.run(cfg(".", synthetic=True, seeds=[0], epochs=1, overwrite=True))
    assert (out / "config.json").exists()


def test_output_path_that_is_a_file_is_refused(tmp_path):
    f = tmp_path / "out.txt"
    f.write_text("x")
    with pytest.raises(FileExistsError, match="not an output folder"):
        FR.run(cfg(f, synthetic=True, seeds=[0], epochs=1, overwrite=True))


def test_synthetic_summary_does_not_claim_a_horizon(tmp_path):
    s = FR.run(cfg(tmp_path / "s", synthetic=True, seeds=[0], epochs=1))
    assert s["synthetic"] and s["horizon"] == "n/a (synthetic)"


def test_normalization_blowup_is_warned(cache, tmp_path, capsys, monkeypatch):
    """A dimension with ~0 train std (A9 note: std floor 1e-6) blows seen_val up by ~1e6; the runner must say so."""
    orig = load_cache.SharedCache.norm_stats

    def tiny_std(self, pool, layers):
        mean, std = orig(self, pool, layers)
        std = std.copy()
        std[:, 0] = 1e-6
        return mean, std
    # layer 33 has no ~0-std dimension in this fixture (layer 16 does: bf16 rounding makes some train dims constant,
    # and seen_val then reaches ~2e6 - exactly the A9 std-floor note), so it is the clean control
    run33 = dict(layers=[33], seeds=[0], epochs=1, save_checkpoints=False, cache=str(cache))
    FR.run(cfg(tmp_path / "ok", **run33))
    assert "~0 train std" not in capsys.readouterr().err
    monkeypatch.setattr(load_cache.SharedCache, "norm_stats", tiny_std)
    FR.run(cfg(tmp_path / "w", **run33))
    assert "~0 train std" in capsys.readouterr().err
    monkeypatch.setattr(load_cache.SharedCache, "norm_stats", orig)
    FR.run(cfg(tmp_path / "real", cache=str(cache), layers=[16], seeds=[0], epochs=1, save_checkpoints=False))
    assert "~0 train std" in capsys.readouterr().err  # the real near-zero-std dims in layer 16 are caught too


def _recs(prefix, n=6, k=2, d=4, seed=0):
    rs = make_synthetic_records(n=n, k=k, d=d, seed=seed)
    for r in rs:
        r["trajectory_ids"] = [f"{prefix}{r['trajectory_ids'][0]}"]
    return rs


def test_split_checks():
    ok = {"train": _recs("a/"), "seen_val": _recs("b/")}
    assert FR.check_splits(ok, 2) == 4
    with pytest.raises(ValueError, match="share trajectories"):
        FR.check_splits({"train": _recs("a/"), "seen_val": _recs("a/")}, 2)
    with pytest.raises(ValueError, match="expected exactly the splits"):
        FR.check_splits({"train": ok["train"], "unseen_test": ok["seen_val"]}, 2)
    with pytest.raises(ValueError, match="no episodes"):
        FR.check_splits({"train": ok["train"], "seen_val": []}, 2)
    with pytest.raises(ValueError, match=r"expected \[1,T,3,D\]"):
        FR.check_splits(ok, 3)
    dup = _recs("b/")
    dup[1]["trajectory_ids"] = dup[0]["trajectory_ids"]
    with pytest.raises(ValueError, match="duplicate trajectory"):
        FR.check_splits({"train": ok["train"], "seen_val": dup}, 2)
    bad = _recs("b/")
    bad[0]["features"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        FR.check_splits({"train": ok["train"], "seen_val": bad}, 2)
    mixed = _recs("b/", d=5)
    with pytest.raises(ValueError, match="inconsistent feature dims"):
        FR.check_splits({"train": ok["train"], "seen_val": mixed}, 2)


# --------------------------------------------------------------------------- scoring

def test_aggregation_in_logit_space():
    x = torch.tensor([-2.0, 3.0, 0.0])
    assert FR.aggregate(x, "max") == pytest.approx(3.0)
    assert FR.aggregate(x, "last") == pytest.approx(0.0)
    assert FR.to_prob(FR.aggregate(x, "mean")) == pytest.approx(torch.sigmoid(x.double()).mean().item())
    assert FR.aggregate(torch.tensor([5.0]), "mean") == pytest.approx(5.0)  # one step: mean == that step
    with pytest.raises(ValueError, match="no valid timesteps"):
        FR.aggregate(torch.tensor([]), "max")
    with pytest.raises(ValueError, match="non-finite"):
        FR.aggregate(torch.tensor([1.0, float("inf")]), "max")


@pytest.mark.parametrize("how", ["max", "last", "mean"])
def test_confident_episodes_do_not_tie(how):
    """sigmoid saturates to 1.0 (float32 at x>~17, float64 at x>~37); scores must still rank 20 < 40 < 80."""
    s = [FR.aggregate(torch.tensor([v, v]), how) for v in (20.0, 40.0, 80.0)]
    assert s[0] < s[1] < s[2] and all(np.isfinite(s))
    assert FR.episode_metrics([0, 1], s[:2])["roc_auc"] == 1.0
    lo = [FR.aggregate(torch.tensor([v, v]), how) for v in (-80.0, -40.0)]
    assert lo[0] < lo[1]


def test_episode_scores_ignore_padding():
    """A padded step with a huge logit must not raise an episode's max score."""
    class Probe(torch.nn.Module):
        def forward(self, batch):
            from failure_prob.models.base import Output
            return Output(batch["features"][:, :, 0, 0])
    long_ = {"features": torch.zeros(1, 5, 1, 1), "mask": torch.ones(1, 5, dtype=torch.bool),
             "labels": torch.tensor([1.0]), "task_ids": [1], "trajectory_ids": ["long"]}
    short = {"features": torch.full((1, 2, 1, 1), -4.0), "mask": torch.ones(1, 2, dtype=torch.bool),
             "labels": torch.tensor([0.0]), "task_ids": [1], "trajectory_ids": ["short"]}
    rows = FR.episode_scores(Probe(), [long_, short], batch_size=2, how="max")
    assert rows[1]["n_steps"] == 2 and rows[1]["score"] == pytest.approx(-4.0)
    assert rows[0]["score"] == pytest.approx(0.0) and rows[0]["prob"] == pytest.approx(0.5)


def test_episode_metrics_edge_cases():
    m = FR.episode_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["tpr_at_5_fpr"] == 1.0
    one = FR.episode_metrics([1, 1], [0.3, 0.4])
    assert math.isnan(one["roc_auc"]) and one["n_failures"] == 2
    tie = FR.episode_metrics([0, 1, 0, 1], [0.5] * 4)
    assert tie["roc_auc"] == 0.5
    with pytest.raises(ValueError, match="same length"):
        FR.episode_metrics([0, 1], [0.5])
    with pytest.raises(ValueError, match="non-finite"):
        FR.episode_metrics([0, 1], [0.5, float("nan")])


def test_summary_handles_nan_and_single_seed():
    per_seed = {0: {"seen_val": {"roc_auc": 0.7, "pr_auc": 0.6, "tpr_at_5_fpr": 0.1}},
                1: {"seen_val": {"roc_auc": float("nan"), "pr_auc": 0.5, "tpr_at_5_fpr": 0.0}}}
    s = FR.summarize(per_seed)
    assert s["roc_auc"]["mean"] == 0.7 and math.isnan(s["roc_auc"]["std"]) and s["roc_auc"]["n_seeds_defined"] == 1
    assert s["pr_auc"]["n_seeds_defined"] == 2
    assert s["pr_auc"]["std"] == pytest.approx(np.std([0.6, 0.5], ddof=1))
