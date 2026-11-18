# E7 — failure and robustness study

Real glossy car and deliberately sparse subsets reuse their exact E1/E3 measurements. Controlled blur is applied identically to every method, preserving calibrated camera geometry. Dino is the official real low-texture public scene.

| Condition | Method | Registered | ATE % | AUC@5 | Status | Reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| glossy_car | ours | 10/20 | 11.307 | 0.000 | completed |  |
| glossy_car | colmap | 10/20 | 0.344 | 0.141 | completed |  |
| glossy_car | vggt | 20/20 | 6.832 | 0.353 | completed |  |
| sparse_car_5 | ours | 0/5 | — | 0.000 | failed | Initialization yielded too few usable 3D points |
| sparse_car_5 | colmap | 0/5 | — | 0.000 | failed | COLMAP did not reconstruct a model |
| sparse_car_5 | vggt | 5/5 | 1.307 | 0.511 | completed |  |
| blurred_temple | ours | 0/16 | — | 0.000 | failed | Initialization yielded too few usable 3D points |
| blurred_temple | colmap | 0/16 | — | 0.000 | failed | COLMAP did not reconstruct a model |
| blurred_temple | vggt | 16/16 | 1.642 | 0.318 | completed |  |
| low_texture_dino | ours | 4/16 | 5.498 | 0.000 | completed |  |
| low_texture_dino | colmap | 6/16 | 0.104 | 0.114 | completed |  |
| low_texture_dino | vggt | 16/16 | 1.805 | 0.137 | completed |  |

## Limits

- The car has a COLMAP pseudo-reference, not true GT. Its appearance is glossy, but these results do not isolate reflectivity as the only cause of failure.
- Sparse car views use the existing five-image E3 subset, with substantial viewpoint gaps. No additional photos or poses were invented.
- Blur is a controlled artificial degradation (Gaussian radius 4px), explicitly not a newly captured blurry scene. Temple GT camera poses remain valid.
- Dino provides supplied GT camera poses. Its data were not used to tune our frozen relaxed configuration.
- Failed reconstructions are retained with zero registration and zero AUC; undefined global alignment does not receive an invented ATE.
- This study evaluates reconstruction outputs and geometry agreement; it does not claim success from attractive point-cloud appearance alone.
