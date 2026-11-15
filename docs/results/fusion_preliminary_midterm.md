# Preliminary fusion results (midterm)

> **Preliminary.** These numbers use example layers and the temporary probe. They are **not** the F1/F2/F3/F6 results: those use layers from the real depth sweep and selection (S2/S3), the B6 probe and the B4 scorer. Do not choose layers from these numbers, and do not copy them into the experiments sheet.

## Setup

- Shared cache v1 on CARC, mean pooling, train-only z-scoring.
- Train on `train` (full horizon). Score every `seen_val` episode at `task_min_step`. 140 episodes, 71 failures.
- Probe: MLP, hidden 64, 5 epochs, batch 8, lr 1e-3, no hyper-parameter search. Seeds 0, 1, 2.
- Episode score = max step logit. 95% CIs: stratified bootstrap over seen_val episodes (1000 resamples).
- Example layers: L33 (final block, before the final norm), L16, L24. These were not selected.
- unseen_test was not read (it is locked until N3). Single fixed split (SAFE seed-0 unseen tasks).
- Code: branch `dhyan/a9-f1-f3`, commit `a2873a3`. Slurm jobs 14266616 and 14266617, about 3 min each, about 6.5 GB peak memory.

## Results (seen_val, mean ± std over 3 seeds)

| Model | Layers | ROC-AUC (%) | ROC-AUC 95% CI | PR-AUC (%) | TPR @ 5% FPR (%) | Parameters | Latency (ms/step) |
|---|---|---|---|---|---|---|---|
| Single layer (K=1) | L33 | **73.19** ± 1.71 | 65.23–80.89 | **74.54** ± 1.18 | 16.90 ± 2.82 | 262,273 | 0.058 |
| Concat (K=2) | L33 + L16 | 68.40 ± 2.54 | 59.80–76.41 | 71.02 ± 2.71 | **21.13** ± 1.41 | 524,417 | 0.075 |
| Concat (K=3) | L33 + L16 + L24 | 69.85 ± 2.57 | 61.06–77.73 | 71.61 ± 2.83 | 17.84 ± 0.81 | 786,561 | 0.091 |
| Static learned weights, 64-d projection | L33 + L16 + L24 | 69.78 ± 2.63 | 61.26–77.26 | 71.23 ± 2.59 | 20.19 ± 2.15 | 790,853 | 0.180 |

Latency is one detector call per timestep at batch 1 on CPU (8 threads), with features already extracted.

## Reading

- Adding these example layers did **not** improve ROC-AUC. The final layer alone was best (73.2%), and fusion was 3–5 points lower.
- The differences are **not significant**: every CI overlaps the others (about ±8 points with 140 validation episodes).
- Likely causes:
  - the layers were not selected to be complementary;
  - fusion has 2–3× the parameters with only 210 training episodes, so it overfits;
  - there was no hyper-parameter search.
- Fusion raised TPR @ 5% FPR (21.1% vs 16.9%), a weak hint of extra signal at strict thresholds.
- Fusion adds under 0.2 ms per step.

## Next

Run F1/F2/F3/F6 once the real sweep and selection fix the layers (S2/S3) and B6 fixes the probe. Each leading experiment gets a compact lr / dropout / weight-decay search. Every gain over the best single layer is checked with the paired bootstrap (`scripts/fusion_compare.py`).
