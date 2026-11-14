# Measured validation

These files contain actual runs, not target values. `temple_ours.json` uses
`configs/sfm_relaxed.yaml`; the default config previously registered only 5/16.
The reference is the supplied calibrated camera trajectory. ATE is measured on
shared registered cameras after Sim(3); AUC includes missing registrations as
failures. One co-located Temple GT pair has undefined translation direction and
is explicitly excluded. The scene was used for parameter tuning; these are
preliminary development measurements, not a held-out evaluation.

| Logged run | Registered | ATE / GT trajectory extent | AUC@5 |
|---|---:|---:|---:|
| sample_ours.json | 8/8 | 0.047% | 0.982 |
| sample_colmap.json | 8/8 | 0.040% | 0.983 |
| temple_ours.json | 16/16 | 0.433% | 0.766 |
| temple_colmap.json | 13/16 | 0.081% | 0.612 |

The generated sample is an image-based integration test. Its precomputed COLMAP
output was produced during development; its summary retains the original run
revision. Final core and Temple validation refer to commit `28fc80c` plus a
source fingerprint. Package versions, configurations, timing, and per-step
histories are embedded in JSONs. COLMAP itself has nondeterministic behavior;
reruns may differ. Timing includes warm disk/download caches as appropriate.

`development_ablations/` holds the generated-sample BA and 5/8-view runs, not the
full mandatory experiment matrix. `fresh_source_tests.txt` logs all 13 tests
passing after extracting tracked source into another folder, using the already
installed development environment. `tests.txt` is the working-copy check.

`temple_preview.png` is a sparse-cloud preview. Downloaded public images and model
weights are excluded from Git. Full raw outputs are available locally in
`outputs/` and `results/`. No VGGT inference result is claimed.
