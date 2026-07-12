# Four-way acquired k-space partitioning for residual-risk verification in self-supervised accelerated brain MRI

This repository contains cleaned code, sanitized configuration templates, tests, documentation, and manuscript figures for the paper:

Four-way acquired k-space partitioning for independent residual-risk verification in self-supervised accelerated brain MRI

## Overview

The project studies residual-risk verification in self-supervised accelerated brain MRI reconstruction.

Acquired k-space measurements are assigned to four non-overlapping roles:

1. reconstruction input,
2. self-supervised reconstruction loss,
3. residual-risk learning,
4. independent residual-risk verification.

This repository is a public materials repository. It is not the private development repository used during experimentation.

## Repository structure

- configs/: sanitized configuration templates
- src/: reusable Python package code
- scripts/: selected experiment and analysis entry points
- tests/: lightweight sanity and unit tests
- paper/figures/: final manuscript figures
- docs/: reproduction and data-access notes

## Data

Raw fastMRI files are not included. Users should obtain the fastMRI brain multicoil data from the official fastMRI source and update the local data paths in the configuration files.

## Not included

This repository does not include:

- raw fastMRI HDF5 files,
- trained model checkpoints,
- large cache arrays,
- Google Drive paths,
- authentication credentials,
- private experiment logs,
- intermediate Colab outputs.

## Installation

Run:

    pip install -e .

## Reproduction

See docs/reproduction.md.

## License

See LICENSE.
