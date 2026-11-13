# Layer fusion: Dhyan's rows (F1, F2, F3, F6)

| Task | Sheet row | Experiment | Code | Layers (from selection) | Compare against |
|---|---|---|---|---|---|
| F1 | 19 | Two-layer concatenation + MLP | `concat_fusion` | complementary pair (S2) | best single-layer MLP (R1/R2) |
| F2 | 20 | Three-layer concatenation + MLP | `concat_fusion` | early / middle / late (S3) | F1 |
| F3 | 21 | Static learned layer weights | `static_weight_fusion --proj-dim 64` | selected 3 layers | F2 (about the same parameter count) |
| F6 | 25 | Number-of-layers ablation, K = 1, 2, 3, 4 | best of F1–F5 | ranked selection, first K | K = 1 |

- **Concatenation** (`failure_prob/models/concat_fusion.py`) joins the K layers per timestep (`[B,T,K·D]`) and probes the result. The head grows with K.
- **Static learned weights** (`failure_prob/models/static_weight_fusion.py`) compute `gamma · Σ_k softmax(α)_k · P_k(x_k)`. Each layer first goes through its own `Linear(D, 64)` into a shared 64-d space ("project features to shared dimension"). The weights are one scalar per layer, the same for every timestep and episode. With `proj_dim = hidden_dim = 64`, F3 has about the same number of parameters as concatenating the same layers (K·4096·64), so F3 vs F2 is a parameter-matched comparison. Without `--proj-dim` it is a pure scalar mix of the z-scored layers, and `--fixed-weights` freezes the mix (uniform, the S6 reference).
- **Probe heads:** both models take `head="mlp"` (default) or `"linear"` (`failure_prob/models/probe_heads.py`). Once B6 fixes the probe, pass its head, width and budget as flags.

## Run (CARC, CPU node)

```bash
# CACHE = the shared cache v1/ folder (team handoff); only train + seen_val files are read
python scripts/fusion_experiments.py search --model concat_fusion --layers A B --cache "$CACHE" \
    --output-dir outputs/fusion/F1 --label "F1 two-layer concat"
python scripts/fusion_experiments.py search --model concat_fusion --layers A B C --cache "$CACHE" \
    --output-dir outputs/fusion/F2 --label "F2 three-layer concat"
python scripts/fusion_experiments.py search --model static_weight_fusion --proj-dim 64 --layers A B C --cache "$CACHE" \
    --output-dir outputs/fusion/F3 --label "F3 static learned weights"
# F6: --layers is the RANKED selection (best first); pass the winner's searched hyper-parameters
python scripts/fusion_experiments.py kcurve --model concat_fusion --layers A B C D --lr 3e-4 --dropout 0.2 \
    --cache "$CACHE" --output-dir outputs/fusion/F6 --label "F6 K ablation"
# a single configuration (no search)
python scripts/run_fusion.py --model static_weight_fusion --proj-dim 64 --layers A B C --cache "$CACHE" \
    --output-dir outputs/fusion/one_run
# CPU smoke test, random data
python scripts/run_fusion.py --model static_weight_fusion --layers 8 16 --synthetic --epochs 2 --output-dir /tmp/f
```

Until S1–S3 pick the layers, any layers run, but the numbers are development numbers only. Use layer 33, not 32, for raw block 32.

**Protocol** (the same as Akash's R1–R4 depth sweep, so fusion rows compare like-for-like with single layers):
- Train on `train` at the full horizon (`--train-horizon`).
- Score every `seen_val` rollout at `task_min_step` (`--horizon`; the only reportable horizon).
- Seeds 0/1/2.
- Probe: mlp 64, 5 epochs, batch 8, lr 1e-3, the sweep's TEMPORARY probe budget.
- Train-only z-scoring is on by default (`--no-normalize` is the raw-feature check). The sweep's single-layer reference should therefore also be run with `--normalize`.
- Episode score: the max step logit (`--aggregation max|mean|last`).

**Compact hyper-parameter search** (`search`, as the protocol asks within each leading experiment):
- The grid is lr {1e-3, 3e-4} × dropout {0, 0.2} × weight decay {0, 1e-4}. Change it with `--lrs --dropouts --weight-decays`.
- Every point runs all three seeds. The winner has the best mean seen_val ROC-AUC.
- The cache is read once for the whole grid (and once for the whole K curve).
- Selection and reporting both use seen_val, so the winner's numbers are optimistic. `search.csv` keeps every grid point, and the sheet row says so in Notes.

**Shared evaluator (B4):** pass `--evaluator package.module:factory`, or a name registered in the depth sweep's `sweep_plugins.EVALUATORS` (PR #4, e.g. `temporary_max`). The contract is the sweep's: `Evaluator(EvalInput) -> dict`. The evaluator gets per-step scores (logits) of every seen_val episode, already cut to `task_min_step`. Once a non-TEMPORARY evaluator is used, the `episode_scorer` provisional flag disappears from every output.

## Output

In `--output-dir` of one run:
- `config.json`: run config, cache version, layer names, the train and seen_val horizons, the evaluator, and the git commit.
- `seed<s>/metrics.json`:
  - seen_val metrics and per-task seen_val metrics;
  - train metrics (a fit check at the train horizon, not comparable with seen_val);
  - learning curve, chosen epoch, trainable parameters, latency, and F3's learned layer weights.
- `seed<s>/seen_val_episode_scores.csv`: one row per episode (`trajectory_id, task_id, failure, n_steps, score, prob`), the input for bootstrap CIs. `score` is in logit space and is what the AUCs rank. `prob` is for reading only (probabilities saturate at 1.0).
- `seed<s>/model.pt`: the checkpoint.
- `summary.json`: mean and std over seeds.
- `sheet_row.csv`: one row in the experiments sheet's columns.
  - Seen Val ROC-AUC / PR-AUC / TPR @ 5% FPR, in **percent (0–100)**, mean over seeds, with the std in Notes.
  - **Parameters** (trainable).
  - **Latency (ms)**: median detector forward time per timestep at batch 1 on the CPU it ran on. Features are assumed already extracted, so this is what fusion adds on top of the VLA.
  - The unseen-test columns stay empty until N3.

`search` adds `search.csv`, `best.json` and `sheet_rows.csv`, plus one `hp_<grid point>/` run per grid point. `kcurve` adds `K<k>/` runs, `k_curve.csv` (with the change against K = 1 in points) and `sheet_rows.csv`.

The runners refuse to overwrite a non-empty folder unless you pass `--overwrite`. Even then they only replace folders they wrote themselves (marker file), and never the current directory, your home folder or the repo.

**Memory:** records are float32, about rows × K × 4096 × 4 bytes. Train at the full horizon is about 1.4 GB per layer, so run K ≥ 3 on a CARC CPU node, not a laptop.

## Guarantees and what is still provisional

- **No unseen data.** Only `train` and `seen_val` are ever requested. Records that overlap between splits, have the wrong shape, are non-finite or repeat an ID are refused. The tests check that no `unseen_test` file is opened, including for `search` and `kcurve`.
- **Horizons are enforced.** Unknown horizons and duplicate or out-of-range layers are refused. Every episode is checked to have exactly the expected number of steps for its split.
- **Padding never reaches a score.**
- **Provisional until B4 and B6 land.** Every output file says so in its `provisional` field. These are development numbers on a single fixed split (SAFE seed-0 unseen tasks), not results.
- **B7 overlap.** B7 (Aniruth) lists "static weights, uniform average". `static_weight_fusion` already covers both (`--fixed-weights` is the uniform average), so agree with Aniruth on one model before F3's real run.
