## Table S8. Paired contrasts between residual-risk predictions and predictive variance

| Comparison | Δ AUPRC [95% CI] | Δ AUROC [95% CI] | Δ Spearman [95% CI] |
|---|---|---|---|
| Full-CNN − MC-dropout variance | +0.1503 [0.1216, 0.1789] | +0.0158 [0.0127, 0.0191] | +0.2682 [0.2479, 0.2898] |
| A4-CNN − MC-dropout variance | +0.1607 [0.1315, 0.1900] | +0.0162 [0.0131, 0.0198] | +0.2681 [0.2466, 0.2916] |
| Full-CNN − ensemble variance | +0.5222 [0.4926, 0.5483] | +0.1195 [0.1102, 0.1300] | +0.1321 [0.1214, 0.1440] |
| A4-CNN − ensemble variance | +0.5325 [0.5020, 0.5602] | +0.1200 [0.1105, 0.1306] | +0.1320 [0.1197, 0.1465] |
| MC-dropout variance − ensemble variance | +0.3719 [0.3330, 0.4093] | +0.1038 [0.0952, 0.1135] | −0.1361 [−0.1573, −0.1158] |

**Note.** Δ = first score − second score. Intervals use 1,000 paired MRI-volume bootstrap replicates on the 40-volume, 999,792-pixel test sample. All comparisons use the same calibration-derived high-residual-risk threshold. Positive differences favor the first score. Main Figure 6 reports the performance of Full-CNN, A4-CNN, MC-dropout variance, and ensemble variance. MAE is excluded because predictive-variance scores do not share the residual-risk numerical scale.
