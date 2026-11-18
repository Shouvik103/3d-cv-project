# E4 — measured ablation

Same prepared images and reference cameras as E1. Settings vary only as named. Feature caches are warm where a matching feature configuration already exists.

| Scene | Variant | Registered | ATE % | AUC@5 | Reprojection px | Seconds | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| car_s1 | sift | 10/20 | 3.244 | 0.018 | 0.632 | 35.923 | completed |
| car_s1 | rootsift | 10/20 | 11.307 | 0.000 | 0.735 | 34.195 | completed |
| car_s1 | orb | 0/20 | — | — | — | 37.227 | failed |
| fountain_s2 | sift | 11/11 | 3.016 | 0.041 | 0.690 | 29.266 | completed |
| fountain_s2 | rootsift | 11/11 | 3.461 | 0.025 | 0.753 | 24.461 | completed |
| fountain_s2 | orb | 11/11 | 3.928 | 0.045 | 0.563 | 16.231 | completed |

Car and fountain references are COLMAP agreement, not GT accuracy. Temple has supplied GT camera poses. ATE only uses shared cameras; missing registrations are penalized in AUC. Failures are retained.
