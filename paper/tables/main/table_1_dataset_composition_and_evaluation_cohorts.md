## Table 1. Dataset composition and evaluation cohorts

| Cohort | Sequence | Volumes | Slices | Permitted use |
|---|---|---:|---:|---|
| AXT2 training | AXT2 | 201 | 3,190 | SSDU reconstruction and residual-risk predictor development. |
| AXT2 calibration | AXT2 | 40 | 636 | Checkpoint selection, calibration-derived thresholds, affine fitting, and interval calibration; no final test verification. |
| AXT2 test | AXT2 | 40 | 636 | Reserved AXT2 verification after model development. |
| AXFLAIR stress test | AXFLAIR | 26 | 416 | Patient-disjoint within-fastMRI cross-sequence stress test; no AXFLAIR training, tuning, or recalibration. |

**Note.** AXT2, axial T2-weighted; AXFLAIR, axial fluid-attenuated inversion recovery. Cohorts are patient-disjoint. The AXFLAIR cohort is an internal cross-sequence stress test.
