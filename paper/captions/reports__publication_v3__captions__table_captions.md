# Table Captions

## Table 1. Independent \(\Lambda_{\mathrm{hold}}\) residual-risk verification performance.

Performance of residual-risk predictors evaluated against the independent \(u_{\Lambda_{\mathrm{hold}}}\) target. Values are volume-bootstrap means with 95% confidence intervals. Top-10% AUPRC is the primary metric for high-risk pixel localization; Spearman correlation measures rank agreement with the independent residual-risk target; MAE measures residual-risk scale error; and top-10% AUROC measures high-risk pixel separability. Lower MAE is better.

## Table 2. Paired volume-bootstrap comparison of A4 CNN against competing residual-risk predictors.

Paired bootstrap differences are reported as A4 CNN minus comparator, with 95% confidence intervals and associated \(p\)-values. Positive \(\Delta\) top-10% AUPRC and positive \(\Delta\) Spearman favor A4. Negative \(\Delta\) MAE favors A4 because lower absolute residual-risk error is better. The table tests whether A4 improves independent \(\Lambda_{\mathrm{hold}}\) residual-risk localization, rank agreement, and scale accuracy relative to direct residual-based and heuristic predictors.

## Table 3. Split conformal residual-risk bounds on independent \(\Lambda_{\mathrm{hold}}\) verification.

Split conformal residual-risk bounds are reported at nominal marginal coverage 0.900. Observed coverage is the empirical test-set coverage against the independent \(u_{\Lambda_{\mathrm{hold}}}\) target. The interval width is \(2\hat{q}\), where \(\hat{q}\) is the conformal quantile of the absolute residual-risk error. Narrower intervals indicate more compact calibrated residual-risk bounds when observed coverage is comparable. Coverage is reported for the defined residual-risk target and should not be interpreted as clinical diagnostic coverage.

## Table S1. Dataset, acquisition, split, and evaluation summary.

Summary of the dataset scope, volume-level split, independent test population, and conformal evaluation sample counts used in the residual-risk verification experiments. The study used the fastMRI brain multicoil AXT2 subset. The table reports the retained AXT2 volume count, train/calibration/test volume counts, split-overlap checks, retained acquisition heterogeneity, independent test pixel-sampling population, and conformal scale/calibration/test sample counts. Learned/heuristic conformal counts and direct residual-risk conformal counts are reported separately because the corresponding analyses used different sampled evaluation populations.

## Table S2. Full independent \(\Lambda_{\mathrm{hold}}\) residual-risk verification metrics.

Full point-estimate metric report for residual-risk predictors evaluated against the independent \(u_{\Lambda_{\mathrm{hold}}}\) target. MAE and MSE measure residual-risk scale error; Pearson and Spearman correlation measure linear and rank association; and top-10% AUROC and AUPRC measure separability and localization of high-risk pixels defined by the independent residual-risk threshold. Table 1 reports the primary subset of these metrics with volume-bootstrap confidence intervals.

## Table S3. CNN residual-risk predictor ablation summary.

Ablation results for CNN residual-risk predictor input variants evaluated in Experiment 008. Metrics are reported on the ablation test split for the residual-risk learning target. The A4 image-derived variant retained only \(|\hat{x}|\), \(|x_0|\), and \(|\hat{x}-x_0|\), removing explicit PSF/mask-related channels. A4 was selected for subsequent independent \(\Lambda_{\mathrm{hold}}\) verification because it preserved competitive top-risk localization while reducing input complexity and achieving the lowest MAE among the ablation variants. These ablation results are used for model-selection evidence and are separate from the final independent holdout verification results.

## Table S4. Residual-risk target construction parameters.

Fixed parameters used to construct the residual-risk target \(u_{\Lambda}\) from held-out k-space inconsistency. The target is obtained by computing the held-out k-space residual, projecting the residual back to image space with density compensation, pooling residual energy, applying a soft anatomical/support mask, normalizing by a percentile scale, and applying log compression. These values define the reported experimental target construction and should not be interpreted as clinically optimized parameters.

## Table S5. Volume-level A4 CNN win rates against competing residual-risk predictors.

Volume-level win rates compare A4 CNN against each competing predictor across the 40 independent test volumes. AUPRC and Spearman wins indicate volumes where A4 achieved a higher value than the comparator. MAE wins indicate volumes where A4 achieved lower absolute residual-risk error. This table assesses whether the aggregate bootstrap results are consistent across volumes rather than driven by a small number of favorable cases.
