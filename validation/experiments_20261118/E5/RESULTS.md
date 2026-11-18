# E5 — measured ablation

Same prepared images and reference cameras as E1. Settings vary only as named. Feature caches are warm where a matching feature configuration already exists.

| Scene | Variant | Registered | ATE % | AUC@5 | Reprojection px | Seconds | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| temple_gt | approximate | 16/16 | 6.970 | 0.051 | 0.485 | 29.552 | completed |
| temple_gt | calibrated | 16/16 | 0.270 | 0.864 | 0.139 | 29.982 | completed |
| temple_gt | approximate_refined | 16/16 | 8.297 | 0.037 | 0.480 | 29.410 | completed |

Car and fountain references are COLMAP agreement, not GT accuracy. Temple has supplied GT camera poses. ATE only uses shared cameras; missing registrations are penalized in AUC. Failures are retained.

## Calibration scope

The user approved Temple in place of unavailable real car checkerboard data. Calibrated means supplied Middlebury K, not a checkerboard calibration we performed.
Approximate means f=1.2*max(width,height) with a centered principal point. The refined condition starts from the same approximate K and fits a shared fx/fy scale only during global BA after four cameras register. Principal point and distortion remain fixed. No GT focal value enters that optimization.

| Condition | Initial median fx | Final median fx |
| --- | ---: | ---: |
| approximate | 768.000 | 768.0 |
| calibrated | 1520.400 | 1520.4 |
| approximate_refined | 768.000 | 782.3361305269098 |
