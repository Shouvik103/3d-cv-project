# E3 — number of input views

Uniformly spaced image subsets are frozen independently for each view count and shared by all methods. These cover the available trajectory; subsets are not claimed to be nested.
The car has 5/10/20/30-view conditions. Temple has only 16 original images, so its GT conditions are 5/10/16; no 20/30-view GT results are invented.
Byte-identical E1 inputs/configs reuse their original measured runs with provenance. Other conditions are fresh processes. Model weights are cached; feature-cache conditions are recorded in run folders.

| Scene | Views | Method | Registered | ATE % | AUC@5 | Seconds | Status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| car_s1 | 5 | ours | 0/5 | — | 0.000 | 2.217 | failed |
| car_s1 | 5 | colmap | 0/5 | — | 0.000 | 1.299 | failed |
| car_s1 | 5 | vggt | 5/5 | 1.307 | 0.511 | 16.637 | completed |
| car_s1 | 10 | ours | 0/10 | — | 0.000 | 9.207 | failed |
| car_s1 | 10 | colmap | 5/10 | 0.520 | 0.099 | 2.059 | completed |
| car_s1 | 10 | vggt | 10/10 | 1.965 | 0.397 | 22.250 | completed |
| car_s1 | 20 | ours | 10/20 | 11.307 | 0.000 | 67.215 | completed |
| car_s1 | 20 | colmap | 10/20 | 0.344 | 0.141 | 8.629 | completed |
| car_s1 | 20 | vggt | 20/20 | 6.832 | 0.353 | 53.500 | completed |
| car_s1 | 30 | ours | 2/30 | — | 0.000 | 73.436 | completed |
| car_s1 | 30 | colmap | 14/30 | 0.411 | 0.119 | 5.561 | completed |
| car_s1 | 30 | vggt | 30/30 | 2.935 | 0.451 | 87.901 | completed |
| temple_gt | 5 | ours | 2/5 | — | 0.058 | 2.907 | completed |
| temple_gt | 5 | colmap | 0/5 | — | 0.000 | 0.689 | failed |
| temple_gt | 5 | vggt | 5/5 | 0.648 | 0.456 | 16.011 | completed |
| temple_gt | 10 | ours | 5/10 | 0.051 | 0.196 | 10.574 | completed |
| temple_gt | 10 | colmap | 3/10 | — | 0.043 | 1.004 | completed |
| temple_gt | 10 | vggt | 10/10 | 0.683 | 0.568 | 23.850 | completed |
| temple_gt | 16 | ours | 16/16 | 0.270 | 0.864 | 57.567 | completed |
| temple_gt | 16 | colmap | 13/16 | 0.094 | 0.606 | 2.959 | completed |
| temple_gt | 16 | vggt | 16/16 | 0.710 | 0.554 | 39.886 | completed |

Camera error on the car is COLMAP agreement, not ground truth. Temple is a development GT scene. Failed runs retain their reason; their registration rate is zero. AUC on failed reconstructions is zero rather than omitted from the aggregate curve.
