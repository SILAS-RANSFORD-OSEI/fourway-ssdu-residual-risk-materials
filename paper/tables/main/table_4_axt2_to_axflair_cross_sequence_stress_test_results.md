## Table 4. AXT2-to-AXFLAIR cross-sequence stress-test results

| Predictor | Cohort | MAE | Spearman | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| Full-CNN | AXT2 | 0.0834 | 0.9389 | 0.9818 | 0.8348 |
| Full-CNN | AXFLAIR | 0.1601 | 0.9646 | 0.9656 | 0.7854 |
| A4-CNN | AXT2 | 0.0829 | 0.9388 | 0.9822 | 0.8452 |
| A4-CNN | AXFLAIR | 0.1288 | 0.9573 | 0.9694 | 0.8147 |

**Note.** The AXFLAIR cohort contained 26 patient-disjoint volumes and 416 slices. No AXFLAIR training, tuning, or recalibration was performed. The AXT2 calibration-derived high-residual-risk threshold was transferred unchanged. This is an internal cross-sequence stress test, not external validation.
