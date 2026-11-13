"""F1/F2/F3 hyper-parameter search and F6 K-ablation drivers (synthetic 34-layer cache, no GPU)."""
import json
import subprocess
import sys

import pandas as pd
import pytest

from test_fusion_runner import cache, cfg  # noqa: F401  (module-scoped cache fixture)
from test_shared_cache import REPO
import load_cache
from failure_prob import fusion_experiments as FX
from failure_prob import fusion_runner as FR


def count_loads(monkeypatch):
    n = {"loads": 0, "splits": []}
    orig = FR.load_cache_splits

    def spy(c):
        n["loads"] += 1
        return orig(c)
    monkeypatch.setattr(FR, "load_cache_splits", spy)
    orig_arr = load_cache.SharedCache._array
    monkeypatch.setattr(load_cache.SharedCache, "_array",
                        lambda self, pool, split, layer: (n["splits"].append(split), orig_arr(self, pool, split, layer))[1])
    return n


def test_search_loads_once_runs_every_point_and_picks_best(cache, tmp_path, monkeypatch):  # noqa: F811
    n = count_loads(monkeypatch)
    out = tmp_path / "F1"
    res = FX.hparam_search(cfg(out, model="concat_fusion", cache=str(cache), seeds=[0, 1], epochs=1,
                               save_checkpoints=False),
                           grid={"lr": [1e-3, 3e-4], "dropout": [0.0, 0.2]}, label="F1 two-layer concat")
    assert n["loads"] == 1 and set(n["splits"]) == {"train", "seen_val"}
    table = pd.read_csv(out / "search.csv")
    assert len(table) == 4 and table.selected.sum() == 1
    assert table.roc_auc_mean.max() == pytest.approx(table[table.selected].roc_auc_mean.iloc[0])
    best = json.loads((out / "best.json").read_text())
    assert best["selected"] == res["selected"] and best["n_grid_points"] == 4
    assert "selection on seen_val" in best["sheet_row"]["Notes"] and best["sheet_row"]["Experiment"] == "F1 two-layer concat"
    for run in table.run:
        assert (pd.read_csv(f"{run}/sheet_row.csv").shape[0] == 1)
    assert list(pd.read_csv(out / "sheet_rows.csv").columns) == FR.SHEET_COLUMNS


def test_search_refuses_foreign_folder_and_bad_grid(tmp_path):
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        FX.hparam_search(cfg(foreign, synthetic=True, seeds=[0], epochs=1, overwrite=True), grid={"lr": [1e-3]})
    assert (foreign / "notes.txt").exists()
    with pytest.raises(ValueError, match="cannot search"):
        FX.hparam_search(cfg(tmp_path / "a", synthetic=True), grid={"batch_size": [4]})
    with pytest.raises(ValueError, match="unique"):
        FX.hparam_search(cfg(tmp_path / "b", synthetic=True), grid={"lr": [1e-3, 1e-3]})
    with pytest.raises(ValueError, match="lr > 0"):
        FX.hparam_search(cfg(tmp_path / "c", synthetic=True), grid={"lr": [0.0]})


def test_search_rerun_with_overwrite(tmp_path):
    out = tmp_path / "s"
    FX.hparam_search(cfg(out, synthetic=True, seeds=[0], epochs=1), grid={"lr": [1e-3]})
    with pytest.raises(FileExistsError):
        FX.hparam_search(cfg(out, synthetic=True, seeds=[0], epochs=1), grid={"lr": [1e-3]})
    FX.hparam_search(cfg(out, synthetic=True, seeds=[0], epochs=1, overwrite=True), grid={"lr": [1e-3]})


def test_k_ablation_uses_prefixes_of_the_ranked_layers(cache, tmp_path, monkeypatch):  # noqa: F811
    n = count_loads(monkeypatch)
    out = tmp_path / "F6"
    rows = FX.k_ablation(cfg(out, model="static_weight_fusion", proj_dim=8, layers=[16, 33, 4], cache=str(cache),
                             seeds=[0], epochs=1, save_checkpoints=False), ks=[1, 2, 3])
    assert n["loads"] == 1
    curve = pd.read_csv(out / "k_curve.csv")
    assert curve.K.tolist() == [1, 2, 3] and curve.layers.tolist() == ["16", "16/33", "16/33/4"]
    assert curve.delta_vs_K1_roc_auc_pp.iloc[0] == 0
    assert curve.Parameters.is_monotonic_increasing  # one projection per layer
    for k, names in [(1, ["block16"]), (3, ["block16", "block32_prenorm", "block04"])]:
        conf = json.loads((out / f"K{k}" / "config.json").read_text())
        assert conf["config"]["layers"] == [16, 33, 4][:k] and conf["provenance"]["layer_names"] == names


def test_k_ablation_validation(tmp_path):
    with pytest.raises(ValueError, match="ks must lie"):
        FX.k_ablation(cfg(tmp_path / "a", synthetic=True, layers=[1, 2]), ks=[1, 2, 3])
    with pytest.raises(ValueError, match="ks must lie"):
        FX.k_ablation(cfg(tmp_path / "b", synthetic=True, layers=[1, 2]), ks=[0, 1])
    with pytest.raises(ValueError, match="init_weights"):
        FX.k_ablation(cfg(tmp_path / "c", synthetic=True, layers=[1, 2], init_weights=[1, 1]), ks=[1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        FX.k_ablation(cfg(tmp_path / "d", synthetic=True, layers=[1, 1]), ks=[1, 2])


def test_first_k_is_a_view():
    import torch
    rec = {"features": torch.zeros(1, 5, 4, 3), "mask": torch.ones(1, 5, dtype=torch.bool)}
    sub = FX.first_k({"train": [rec]}, 2)["train"][0]["features"]
    assert sub.shape == (1, 5, 2, 3) and sub.data_ptr() == rec["features"].data_ptr()


def test_cli_search_and_kcurve(cache, tmp_path):  # noqa: F811
    script = REPO / "scripts" / "fusion_experiments.py"
    r = subprocess.run([sys.executable, str(script), "search", "--model", "concat_fusion", "--layers", "16", "33",
                        "--cache", str(cache), "--output-dir", str(tmp_path / "s"), "--epochs", "1", "--seeds", "0",
                        "--lrs", "1e-3", "--dropouts", "0", "--weight-decays", "0", "--no-checkpoints"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "s" / "best.json").exists()
    r = subprocess.run([sys.executable, str(script), "kcurve", "--model", "concat_fusion", "--layers", "16", "33",
                        "--ks", "1", "2", "3", "--cache", str(cache), "--output-dir", str(tmp_path / "k")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "ks must lie" in r.stderr
