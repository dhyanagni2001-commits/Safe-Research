# Layer fusion: concatenation and static weights (F1–F3)

| Plan row | Code | What it is |
|---|---|---|
| F1 | `failure_prob/models/concat_fusion.py` (`concat_fusion`) | Concatenate the K chosen layers per timestep (`[B,T,K·D]`) and probe the result. The head grows with K. |
| F2 | `failure_prob/models/static_weight_fusion.py` (`static_weight_fusion`) | `gamma · Σ_k softmax(α)_k · x_k`: one learned scalar per layer, the same for every timestep and episode, then probe the fused `[B,T,D]`. The head is the size of a single-layer probe; fusion adds K+1 parameters. `--fixed-weights` freezes the mix (uniform, or `--init-weights`). |
| F3 | `failure_prob/fusion_runner.py`, `scripts/run_fusion.py` | Train on `train`, score `seen_val` per episode at the `task_min_step` horizon, for seeds 0/1/2. Writes the metrics, per-episode scores and checkpoints. |

Both models take `head="mlp"` (the repo's `MLPHead`, the default) or `head="linear"` (a logistic probe), via `failure_prob/models/probe_heads.py`. Once the probe spec (B6) is fixed, pass its head, width and training budget as arguments. The fusion code doesn't change.

## Run

```bash
# CACHE = the shared cache v1/ folder (see the team handoff); train + seen_val files only
python scripts/run_fusion.py --model static_weight_fusion --layers 16 33 --cache "$CACHE" \
    --output-dir outputs/fusion/swf_L16_L33
python scripts/run_fusion.py --model concat_fusion --layers 16 33 --head linear --cache "$CACHE" \
    --output-dir outputs/fusion/concat_L16_L33
# reference with the identical head and loop: the uniform average of the same layers
python scripts/run_fusion.py --model static_weight_fusion --fixed-weights --layers 16 33 --cache "$CACHE" \
    --output-dir outputs/fusion/uniform_L16_L33
# CPU smoke test, random data
python scripts/run_fusion.py --model static_weight_fusion --layers 8 16 --synthetic --epochs 2 --output-dir /tmp/f
```

Defaults: pool `mean`, train-only z-scoring (`--no-normalize` for the raw-feature sensitivity check), horizon `task_min_step`, seeds 0/1/2, 10 epochs, last epoch kept (`--select-best-epoch` picks by seen_val ROC-AUC), episode score = max step probability (`--aggregation max|mean|last`). Use layer 33, not 32, for raw block 32. `--layers` follows the layer sets from S1–S6.

Output, in `--output-dir`:
- `config.json`: the run config, cache version, layer names and git commit.
- `seed<s>/metrics.json`: seen_val and train episode metrics (ROC-AUC, PR-AUC, TPR@5%FPR), per-task seen_val metrics, learning curve, chosen epoch, trainable parameter count, and the learned layer weights for F2.
- `seed<s>/seen_val_episode_scores.csv`: one row per episode (`trajectory_id, task_id, failure, n_steps, score, prob`). This is the input for bootstrap CIs. `score` is in logit space and is what the AUCs rank; `prob = sigmoid(score)` is for reading only, because probabilities saturate at 1.0 and would tie confident episodes.
- `seed<s>/model.pt`: the checkpoint.
- `summary.json`: mean and std over seeds.

The runner refuses to overwrite a non-empty output folder unless you pass `--overwrite`. Even then it only deletes a folder it wrote itself (marker in `config.json`, only its own files at the top level). It never deletes the current directory, your home folder or the repo.

Memory: records are float32, so you need about rows × K × 4096 × 4 bytes. At `task_min_step` that is under 1.4 GB per layer for train, plus less for seen_val. Run K ≥ 4 on a CARC CPU node, not a laptop.

## Guarantees and what is still provisional

- **No unseen data.** The runner only ever requests `train` and `seen_val`. It refuses records that share trajectory IDs across the two splits, records with the wrong shape, non-finite records and duplicate IDs (`tests/test_fusion_runner.py` checks that `unseen_test` files are never opened).
- **The horizon is enforced.** The runner rejects an unknown `horizon` and duplicate or out-of-range layers. It also checks that each episode has exactly `task_min_step` steps, so it does not depend on loader findings A9-1/2/3 (`docs/reviews/A9_loader_export_review.md`).
- **Padding never reaches a score.** Padded timesteps are excluded from the loss and from episode scores.
- **Provisional until the shared pieces land.** The episode scorer (until B4) and the probe settings (until B6) are both provisional, and every output file says so in its `provisional` field. These are development numbers on a single fixed split (SAFE seed-0 unseen tasks), not results. To switch to the shared evaluator, replace `episode_scores` / `episode_metrics` in `fusion_runner.py`.
