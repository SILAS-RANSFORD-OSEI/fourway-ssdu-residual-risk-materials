# Reproduction guide

The expected workflow is:

1. Install the package.
2. Download fastMRI brain multicoil data separately.
3. Edit the sanitized configuration files to point to local data, checkpoint, and output directories.
4. Generate four-way masks.
5. Train or supply an SSDU reconstruction model.
6. Build residual-risk targets.
7. Train the A4 CNN predictor.
8. Run independent holdout verification.
9. Generate manuscript figures.

The public repository does not include raw data, trained checkpoints, or large intermediate caches.
