| Component            | Parameter                | Value    | Purpose                                                                     |
|:---------------------|:-------------------------|:---------|:----------------------------------------------------------------------------|
| Log scaling          | $\alpha$                 | 10       | Compresses the residual-energy dynamic range in the log-scaled risk target. |
| Normalization        | $Q_{99}$ percentile      | 99       | Normalizes residual-energy scale before log transformation.                 |
| Numerical stability  | $\epsilon$               | 1.0e-08  | Prevents division by zero during residual-energy normalization.             |
| Local pooling        | Pooling patch size       | 16       | Aggregates residual energy into local spatial risk regions.                 |
| Density compensation | DCF kernel size          | 9        | Controls smoothing used in density-compensated adjoint residual projection. |
| PSF/gain estimation  | Number of PSF probes     | 4        | Sets the number of probes used for PSF/gain-related residual normalization. |
| PSF/gain estimation  | PSF random seed          | 20260530 | Fixes the random probe generation for reproducibility.                      |
| Soft support mask    | Support source           | x0_rss   | Defines the image used to construct the soft anatomy/support mask.          |
| Soft support mask    | Support threshold        | 0.05     | Sets the intensity threshold for support-mask construction.                 |
| Soft support mask    | Support softness         | 0.02     | Controls the smooth transition of the support mask.                         |
| Soft support mask    | Support smoothing kernel | 15       | Controls spatial smoothing of the support mask.                             |
| Soft support mask    | Support percentile       | 99       | Defines the reference percentile used to scale the support source.          |
| Soft support mask    | Support power            | 1        | Controls nonlinear weighting of the support mask.                           |