# E1 — matched-input comparison

These are recorded development results. Car/fountain pose numbers measure agreement with an existing full-data COLMAP pseudo-reference; only Temple has supplied GT cameras.

| Scene | Method | Status | Registered | ATE % | AUC@5 | Median reprojection px | Time s | Peak RSS MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car_s1 | ours | completed | 10/20 | 11.307 | 0.000 | 0.735 | 67.2 | 389.3 |
| car_s1 | colmap | completed | 10/20 | 0.344 | 0.141 | 0.555 | 8.6 | 1325.7 |
| car_s1 | vggt | completed | 20/20 | 6.832 | 0.353 | — | 53.5 | 3647.0 |
| fountain_s2 | ours | completed | 11/11 | 3.461 | 0.025 | 0.753 | 48.9 | 340.9 |
| fountain_s2 | colmap | completed | 11/11 | 2.048 | 0.083 | 0.828 | 21.3 | 1333.6 |
| fountain_s2 | vggt | completed | 11/11 | 0.460 | 0.793 | — | 26.3 | 3822.9 |
| temple_gt | ours | completed | 16/16 | 0.270 | 0.864 | 0.139 | 57.6 | 188.6 |
| temple_gt | colmap | completed | 13/16 | 0.094 | 0.606 | 0.139 | 3.0 | 457.2 |
| temple_gt | vggt | completed | 16/16 | 0.710 | 0.554 | — | 39.9 | 3820.9 |

## Interpretation and scope

- Missing registrations count as failures in pose AUC. ATE only evaluates shared registered cameras; registration rate must be read beside it.
- The car subset spans the existing curated views. Both classical algorithms can split/lose coverage; low image reprojection residuals do not demonstrate globally correct camera geometry.
- Our relaxed configuration was previously tuned on Temple; car selection previously used COLMAP poses. This is not an unseen generalization benchmark.
- Our SfM and COLMAP share fixed calibration. VGGT receives the same photo bytes but uses its own 518px preprocessing and predicted intrinsics.
- VGGT timing is local Apple MPS when that device is recorded, not a claimed CUDA/T4 result. End-to-end and forward-only timings have different scope.
- Cloud agreement is measured against COLMAP sparse points, not ground-truth surface geometry. Pixel reprojection error is omitted for VGGT because it does not export observed feature tracks.

## Reproduction

```bash
.venv/bin/python code/experiments/run_e1.py
.venv/bin/python code/experiments/report_e1.py
```

Frozen protocol: `protocol.yaml`. Image identities: `datasets/*/inputs.json`. Per-method logs, performance and evaluation: `E1/<scene>/<method>/`. Figures: `E1/figures/`.
