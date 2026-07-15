Table S1. Remaining dataset, acquisition, mask, and conformal audit parameters supporting Table 1.

This supplementary table reports extended parameters that support the compact dataset, split, leakage-check, and evaluation-population summary reported in main Table 1.

| Category                     | Parameter                                   | Value                                 | Notes                                                                                     |
|:-----------------------------|:--------------------------------------------|:--------------------------------------|:------------------------------------------------------------------------------------------|
| Dataset audit                | Raw data source                             | NYU fastMRI brain multicoil           | Public de-identified raw k-space dataset used for computational analysis.                 |
| Dataset audit                | Acquisition subset                          | AXT2                                  | Axial T2-weighted brain acquisition subset retained for all experiments.                  |
| Dataset audit                | Retained readable AXT2 volumes              | 281                                   | Volumes retained after HDF5 readability and acquisition-type audit.                       |
| Acquisition heterogeneity    | Coil-count range                            | 4–20                                  | Variable coil counts were retained rather than forcing fixed coil-count geometry.         |
| Acquisition heterogeneity    | Mean acquired k-space width                 | 367.00                                | Mean phase-encoding width across retained AXT2 volumes.                                   |
| Acquisition heterogeneity    | Observed k-space widths                     | 320, 396                              | Representative width heterogeneity retained during evaluation.                            |
| Four-way mask design         | ACS lines                                   | 24                                    | Central autocalibration signal lines were assigned strictly to the reconstruction subset. |
| Four-way mask design         | Outer-k-space allocation                    | 50% Θ, 25% Λrec, 15% Λrisk, 10% Λhold | Fixed outer-line proportions used after ACS assignment.                                   |
| Four-way mask audit          | Mean acquired sampled lines |Ω|             | 91.75                                 | Mean number of acquired sampled phase-encoding lines.                                     |
| Four-way mask audit          | Mean reconstruction-input lines |Θ|         | 58.17                                 | Mean number of lines assigned to reconstruction input.                                    |
| Four-way mask audit          | Mean reconstruction-loss lines |Λrec|       | 17.10                                 | Mean number of lines assigned to SSDU reconstruction-loss evaluation.                     |
| Four-way mask audit          | Mean residual-risk learning lines |Λrisk|   | 9.86                                  | Mean number of lines assigned to residual-risk learning target construction.              |
| Four-way mask audit          | Mean independent verification lines |Λhold| | 6.62                                  | Mean number of lines assigned to independent holdout verification.                        |
| Conformal audit              | Nominal coverage                            | 0.900                                 | Target marginal coverage used for split conformal residual-risk bounds.                   |
| Conformal audit              | Affine scaling split                        | Calibration split only                | Affine correction was estimated without using the independent test split.                 |
| Conformal audit              | Conformal quantile split                    | Calibration split only                | Conformal residual-risk margins were estimated before independent test evaluation.        |
| Independent evaluation audit | Independent test slices                     | 636                                   | Number of slices represented in the independent verification analysis.                    |
| Independent evaluation audit | Total independent test pixel samples        | 651,264                               | Total sampled pixels used for independent residual-risk verification.                     |
