## Table S5. Calibration performance and paired Full-CNN comparator contrasts

### A. Calibration-cohort point estimates

| Predictor | MAE | Spearman | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Full-CNN | 0.084350 | 0.958580 | 0.980439 | 0.826863 |
| A4-CNN | 0.086503 | 0.947048 | 0.980688 | 0.831774 |

**Note.** Calibration estimates use the reserved verification target in the calibration cohort. These volumes also supported model selection; the independent AXT2 test estimates are reported in the main manuscript.

### B. Reserved-test Full-CNN versus comparator contrasts

| Comparator | Δ AUPRC [95% CI] | Δ MAE [95% CI] | Δ AUROC [95% CI] | Δ Spearman [95% CI] |
|---|---|---|---|---|
| A4-CNN | −0.010329 [−0.017178, −0.003849] | +0.000455 [−0.001163, +0.002012] | −0.000431 [−0.001063, +0.000220] | +0.000199 [−0.008356, +0.008978] |
| Image-gradient magnitude | +0.257244 [+0.232251, +0.283436] | −0.334955 [−0.354527, −0.318511] | +0.040364 [+0.037855, +0.042870] | +0.132843 [+0.117509, +0.147915] |
| Reconstruction magnitude | +0.267086 [+0.240096, +0.291286] | −0.257006 [−0.274377, −0.241511] | +0.035154 [+0.029998, +0.040230] | +0.043347 [+0.032623, +0.053763] |
| PSF/gain | +0.307012 [+0.279235, +0.333869] | −0.161001 [−0.173917, −0.149661] | +0.032914 [+0.028806, +0.037325] | +0.018044 [+0.011386, +0.025348] |
| Intervention magnitude | +0.679643 [+0.656961, +0.701707] | −0.460052 [−0.475099, −0.445031] | +0.474325 [+0.452543, +0.494600] | +0.561852 [+0.527398, +0.599447] |

**Note.** Δ = Full-CNN − comparator. Entries are mean differences across 1,000 paired MRI-volume bootstrap replicates with percentile 95% CIs. Positive differences favor Full-CNN for AUPRC, AUROC, and Spearman correlation; negative differences favor Full-CNN for MAE. All methods used identical volume draws and the fixed calibration-derived threshold.
