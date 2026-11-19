# Analysis

Cases: 20 (splits: {'train': 18, 'val': 2}). Medians [IQR]; p = paired Wilcoxon signed-rank, normal vs fast.

Excluded from head sizing: 6 of 40 hips (5 implant, 1 cut-off head only):

| image_id   | side   |   metal_voxels |   head_below_mm |
|:-----------|:-------|---------------:|----------------:|
| s0058      | right  |          12799 |            43   |
| s0358      | right  |          14953 |           239.3 |
| s0454      | left   |          22215 |           144.1 |
| s0573      | left   |          21735 |           177   |
| s0591      | left   |              0 |           -31.1 |
| s0591      | right  |           3049 |           -23   |


## Segmentation, all sides

| bone        |   n | dice normal         | dice fast           |   p dice | assd normal      | assd fast        |   p assd | hd95 normal      | hd95 fast        |   p hd95 |
|:------------|----:|:--------------------|:--------------------|---------:|:-----------------|:-----------------|---------:|:-----------------|:-----------------|---------:|
| femur_left  |  20 | 0.989 [0.983–0.992] | 0.953 [0.949–0.956] |    2e-06 | 0.18 [0.15–0.31] | 0.75 [0.66–0.77] |    2e-06 | 1.50 [1.50–1.50] | 2.12 [1.50–2.12] |   0.002  |
| femur_right |  20 | 0.991 [0.980–0.993] | 0.953 [0.952–0.956] |    4e-06 | 0.17 [0.15–0.38] | 0.73 [0.70–0.78] |    4e-06 | 1.50 [1.50–1.50] | 2.12 [1.97–2.12] |   0.0004 |
| hip_left    |  20 | 0.986 [0.984–0.989] | 0.933 [0.928–0.938] |    2e-06 | 0.19 [0.16–0.22] | 0.77 [0.75–0.81] |    2e-06 | 1.50 [1.50–1.50] | 2.12 [2.12–2.12] |   4e-05  |
| hip_right   |  20 | 0.986 [0.985–0.989] | 0.934 [0.928–0.937] |    2e-06 | 0.20 [0.16–0.21] | 0.78 [0.76–0.83] |    2e-06 | 1.50 [1.50–1.50] | 2.12 [2.12–2.12] |   4e-05  |


Pooled femur medians: Dice 0.991 normal vs 0.953 fast; ASSD 0.18 vs 0.74 mm (fast/normal = 4.2x).


## Segmentation, native sides only (implant and cut-off sides removed)

| bone        |   n | dice normal         | dice fast           |   p dice | assd normal      | assd fast        |   p assd | hd95 normal      | hd95 fast        |   p hd95 |
|:------------|----:|:--------------------|:--------------------|---------:|:-----------------|:-----------------|---------:|:-----------------|:-----------------|---------:|
| femur_left  |  17 | 0.991 [0.984–0.992] | 0.954 [0.950–0.956] |    2e-05 | 0.18 [0.15–0.30] | 0.75 [0.67–0.77] |    2e-05 | 1.50 [1.50–1.50] | 2.12 [1.50–2.12] |   0.004  |
| femur_right |  17 | 0.992 [0.987–0.993] | 0.953 [0.952–0.958] |    2e-05 | 0.16 [0.14–0.24] | 0.73 [0.68–0.77] |    2e-05 | 1.50 [1.50–1.50] | 2.12 [1.50–2.12] |   0.0005 |
| hip_left    |  17 | 0.986 [0.985–0.989] | 0.934 [0.929–0.938] |    2e-05 | 0.19 [0.15–0.21] | 0.76 [0.75–0.80] |    2e-05 | 1.50 [1.50–1.50] | 2.12 [2.12–2.12] |   0.0002 |
| hip_right   |  17 | 0.987 [0.986–0.990] | 0.934 [0.931–0.937] |    2e-05 | 0.19 [0.15–0.20] | 0.78 [0.76–0.81] |    2e-05 | 1.50 [1.50–1.50] | 2.12 [2.12–2.12] |   0.0001 |


Pooled femur medians: Dice 0.991 normal vs 0.954 fast; ASSD 0.17 vs 0.73 mm (fast/normal = 4.3x).


## Femoral head (native, complete heads: n = 34 per model)

GT diameter mean 48.9 mm (SD 3.0, range 44.2–56.6); GT sphere-fit RMSE median 0.76 mm.

| model   | subset                            |   n |   median abs diam err (mm) |   max abs diam err (mm) |   bias (mm) |   LoA ± (mm) |   median centre err (mm) |   diam err > 1 mm |
|:--------|:----------------------------------|----:|---------------------------:|------------------------:|------------:|-------------:|-------------------------:|------------------:|
| fast    | all native                        |  34 |                       0.49 |                    9.78 |        0.54 |         4.28 |                     0.72 |                 5 |
| fast    | fit check passed (>= 2500 points) |  32 |                       0.48 |                    1.26 |        0.03 |         1.21 |                     0.7  |                 3 |
| normal  | all native                        |  34 |                       0.1  |                   16.9  |        0.46 |         5.7  |                     0.17 |                 1 |
| normal  | fit check passed (>= 2500 points) |  33 |                       0.1  |                    0.36 |       -0.04 |         0.28 |                     0.17 |                 0 |


Paired normal vs fast (all native heads): |diameter error| p = 2e-06; centre error p = 5e-07.

Spearman(|diameter error|, femur Dice), fast: rho = -0.02, p = 0.89.

Spearman(|diameter error|, femur Dice), full: rho = -0.11, p = 0.53.


### Fit failures (fewer than the minimum number of fit points)

| image_id   | side   | mode   |   dice |   gt_diam_mm |   pred_diam_mm |   diam_err_mm |   centre_err_mm |   n_points |   fit_rmse_mm |
|:-----------|:-------|:-------|-------:|-------------:|---------------:|--------------:|----------------:|-----------:|--------------:|
| s0477      | right  | fast   |  0.949 |       45.359 |         53.082 |         7.723 |           6.691 |       1773 |         1.127 |
| s0519      | right  | fast   |  0.958 |       49.642 |         59.424 |         9.783 |           8.254 |       1959 |         1.092 |
| s0573      | right  | full   |  0.992 |       48.162 |         65.06  |        16.898 |          13.35  |       1347 |         1.156 |


Passing fits use at least 3314 points (max fit RMSE 1.01 mm).


### Excluded hips (for reference)

| image_id   | side   | mode   |   gt_diam_mm |   pred_diam_mm |   diam_err_mm |   centre_err_mm |
|:-----------|:-------|:-------|-------------:|---------------:|--------------:|----------------:|
| s0058      | right  | full   |        55.08 |          63.52 |          8.44 |            4.56 |
| s0058      | right  | fast   |        55.08 |          61.62 |          6.54 |            4.97 |
| s0358      | right  | full   |        42.28 |          43.07 |          0.79 |            2.33 |
| s0358      | right  | fast   |        42.28 |          38.48 |         -3.8  |            3.19 |
| s0454      | left   | full   |        49.67 |          49.63 |         -0.04 |            0.89 |
| s0454      | left   | fast   |        49.67 |          50.31 |          0.63 |            0.61 |
| s0573      | left   | full   |        52.13 |          51.11 |         -1.02 |            0.39 |
| s0573      | left   | fast   |        52.13 |          51.81 |         -0.31 |            0.92 |
| s0591      | left   | full   |        50.91 |          50.38 |         -0.53 |            0.36 |
| s0591      | right  | full   |        42.17 |          38.37 |         -3.8  |            3.11 |
| s0591      | left   | fast   |        50.91 |          49.07 |         -1.84 |            1.22 |
| s0591      | right  | fast   |        42.17 |          37.35 |         -4.82 |            4.65 |
