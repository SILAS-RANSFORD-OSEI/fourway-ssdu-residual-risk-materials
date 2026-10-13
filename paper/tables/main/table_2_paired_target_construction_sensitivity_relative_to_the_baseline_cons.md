## Table 2. Paired target-construction sensitivity relative to the baseline construction

| Target-construction change | Δ MAE [95% CI] | Δ Spearman [95% CI] | Δ AUROC [95% CI] | Δ AUPRC [95% CI] |
|---|---|---|---|---|
| DCF kernel = 5 | −0.00003 [−0.00201, +0.00172] | +0.00048 [−0.00029, +0.00137] | −0.00063 [−0.00168, +0.00023] | −0.00659 [−0.01998, +0.00406] |
| DCF kernel = 13 | +0.00064 [−0.00062, +0.00199] | −0.00070 [−0.00145, −0.00011] | +0.00002 [−0.00061, +0.00065] | −0.00062 [−0.00876, +0.00842] |
| Gain probes = 2 | +0.00068 [+0.00036, +0.00101] | −0.00032 [−0.00051, −0.00007] | −0.00026 [−0.00039, −0.00012] | −0.00160 [−0.00309, +0.00001] |
| Gain probes = 8 | −0.00047 [−0.00066, −0.00028] | +0.00018 [+0.00002, +0.00034] | +0.00008 [−0.00005, +0.00021] | +0.00084 [−0.00050, +0.00207] |
| Normalization percentile = Q95 | +0.08286 [+0.07631, +0.08898] | +0.00134 [+0.00014, +0.00251] | −0.00424 [−0.00616, −0.00233] | −0.02129 [−0.03949, −0.00400] |
| Normalization percentile = Q99.5 | +0.00191 [−0.00006, +0.00375] | −0.00082 [−0.00120, −0.00039] | −0.00037 [−0.00062, −0.00007] | −0.00591 [−0.00904, −0.00247] |
| Log coefficient α = 5 | +0.06480 [+0.05705, +0.07207] | +0.00000 [−0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |
| Log coefficient α = 20 | +0.11313 [+0.10279, +0.12506] | −0.00000 [−0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |

**Note.** Δ = variant − baseline. Entries are mean paired within-volume metric differences across 40 test volumes, with 95% CIs from 1,000 paired MRI-volume bootstrap replicates. The complete sensitivity experiment comprised the baseline plus the eight alternatives shown here. Baseline settings were DCF width 9, four probes, percentile 99, and α = 10. Variant-specific calibration thresholds are reported in Supplementary Table S3.
