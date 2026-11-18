# E6 — measured ablation

Same prepared images and reference cameras as E1. Settings vary only as named. Feature caches are warm where a matching feature configuration already exists.

| Scene | Variant | Registered | ATE % | AUC@5 | Reprojection px | Seconds | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| car_s1 | baseline | 10/20 | 11.307 | 0.000 | 0.735 | 33.344 | completed |
| car_s1 | ratio_0.7 | 10/20 | 0.935 | 0.091 | 0.490 | 8.687 | completed |
| car_s1 | ratio_0.8 | 9/20 | 8.803 | 0.000 | 0.758 | 24.755 | completed |
| car_s1 | ratio_0.9 | 10/20 | 1.539 | 0.042 | 0.568 | 34.755 | completed |
| car_s1 | ransac_0.75 | 2/20 | — | 0.000 | 0.510 | 32.805 | completed |
| car_s1 | ransac_3.0 | 2/20 | — | 0.000 | 0.881 | 30.339 | completed |

Car and fountain references are COLMAP agreement, not GT accuracy. Temple has supplied GT camera poses. ATE only uses shared cameras; missing registrations are penalized in AUC. Failures are retained.
