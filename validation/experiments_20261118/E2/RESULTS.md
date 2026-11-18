# E2 — measured ablation

Same prepared images and reference cameras as E1. Settings vary only as named. Feature caches are warm where a matching feature configuration already exists.

| Scene | Variant | Registered | ATE % | AUC@5 | Reprojection px | Seconds | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| car_s1 | none | 10/20 | 11.324 | 0.000 | 1.157 | 31.067 | completed |
| car_s1 | local | 10/20 | 11.220 | 0.000 | 1.168 | 31.715 | completed |
| car_s1 | local_global | 10/20 | 11.307 | 0.000 | 0.735 | 31.912 | completed |
| temple_gt | none | 16/16 | 0.377 | 0.811 | 0.188 | 20.603 | completed |
| temple_gt | local | 16/16 | 0.273 | 0.863 | 0.140 | 23.674 | completed |
| temple_gt | local_global | 16/16 | 0.270 | 0.864 | 0.139 | 29.014 | completed |

Car and fountain references are COLMAP agreement, not GT accuracy. Temple has supplied GT camera poses. ATE only uses shared cameras; missing registrations are penalized in AUC. Failures are retained.
