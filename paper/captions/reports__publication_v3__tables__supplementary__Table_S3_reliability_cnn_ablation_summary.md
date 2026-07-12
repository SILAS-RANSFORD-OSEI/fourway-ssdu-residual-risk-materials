| Variant                           | Input modification                                       | MAE   | Spearman   | Top-10% AUPRC   | Top-10% AUROC   | Best epoch   |
|:----------------------------------|:---------------------------------------------------------|:------|:-----------|:----------------|:----------------|:-------------|
| A0: full model + PSF skip         | All channels with explicit PSF skip connection.          | 0.079 | 0.943      | 0.847           | 0.983           | 16           |
| A1: full channels, no PSF skip    | All channels retained, explicit PSF skip removed.        | 0.078 | 0.956      | 0.843           | 0.983           | 17           |
| A2: no $q_{\mathrm{PSF}}$ channel | $q_{\mathrm{PSF}}$ channel removed; PSF skip removed.    | 0.079 | 0.952      | 0.846           | 0.983           | 13           |
| A3: no analytical PSF channel     | Analytical PSF channel removed; PSF skip retained.       | 0.078 | 0.946      | 0.845           | 0.983           | 16           |
| A4: image-derived input           | Only $|\hat{x}|$, $|x_0|$, and $|\hat{x}-x_0|$ retained. | 0.077 | 0.956      | 0.846           | 0.983           | 42           |
| A5: no intervention channel       | Intervention channel $|\hat{x}-x_0|$ removed.            | 0.079 | 0.948      | 0.846           | 0.983           | 16           |