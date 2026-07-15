Table S2. Full independent verification metrics.

Full point-estimate metric report for residual-risk predictors evaluated against the independent verification target. MAE and MSE measure residual-risk scale error; Pearson and Spearman measure linear and rank association; Top-10% AUROC and Top-10% AUPRC measure separation and localization of high-risk pixels defined by the top-risk threshold. Main Table 4 reports the primary subset of these metrics with volume-bootstrap confidence intervals.

| Predictor                   |   MAE |   MSE |   Pearson |   Spearman |   Top-risk fraction |   Top-risk threshold |   Top-10% AUROC |   Top-10% AUPRC |   Evaluation pixels |
|:----------------------------|------:|------:|----------:|-----------:|--------------------:|---------------------:|----------------:|----------------:|--------------------:|
| A4 CNN                      | 0.08  | 0.027 |     0.971 |      0.952 |                 0.1 |                 1.73 |           0.982 |           0.841 |              651264 |
| Full CNN                    | 0.083 | 0.028 |     0.971 |      0.941 |                 0.1 |                 1.73 |           0.982 |           0.841 |              651264 |
| Direct risk-target transfer | 0.092 | 0.039 |     0.96  |      0.959 |                 0.1 |                 1.73 |           0.977 |           0.78  |              651264 |
| Direct raw residual         | 0.435 | 0.663 |     0.685 |      0.948 |                 0.1 |                 1.73 |           0.965 |           0.718 |              651264 |
| Gain-envelope               | 0.244 | 0.202 |     0.924 |      0.921 |                 0.1 |                 1.73 |           0.949 |           0.531 |              651264 |
| Reconstruction magnitude    | 0.34  | 0.403 |     0.749 |      0.882 |                 0.1 |                 1.73 |           0.947 |           0.57  |              651264 |
| Image gradient              | 0.418 | 0.617 |     0.637 |      0.806 |                 0.1 |                 1.73 |           0.941 |           0.583 |              651264 |
| Intervention magnitude      | 0.542 | 0.5   |     0.101 |      0.381 |                 0.1 |                 1.73 |           0.504 |           0.16  |              651264 |
