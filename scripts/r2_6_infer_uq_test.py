import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fourway_mri.reliability_model import ReliabilityUNetSmall
from fourway_mri.r2_6_uq_model import (
    R26MCDropoutUNet,
    apply_a4_input,
    enable_mc_dropout,
)


# Frozen inference parameters

MC_PASSES = 20
ENSEMBLE_SIZE = 3

MC_INFERENCE_SEED = 20260529

TAU_HOLD = 1.7371634244918823

EXPECTED_TEST_ROWS = 999792
EXPECTED_TEST_SLICES = 636
EXPECTED_TEST_VOLUMES = 40

EXPECTED_PIXELS_PER_SLICE = 1572

ALIGNMENT_TOLERANCE = 1.0e-4


# Helper functions

def sha256_file(path, block=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(block),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def load_checkpoint(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


def load_mc_model(path, device):

    model = R26MCDropoutUNet(
        in_channels=6,
        out_channels=1,
        base_channels=8,
        dropout_probability=0.10,
    )

    ckpt = load_checkpoint(
        path
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    # Enable Dropout2d without changing other evaluation-mode layers.
    enable_mc_dropout(
        model
    )

    return model


def load_ensemble_model(path, device):

    model = ReliabilityUNetSmall(
        in_channels=6,
        out_channels=1,
        base_channels=8,
    )

    ckpt = load_checkpoint(
        path
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# Entry point

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-manifest",
        required=True,
    )

    parser.add_argument(
        "--sampled-pixels",
        required=True,
    )

    parser.add_argument(
        "--mc-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--ensemble-checkpoints",
        nargs=3,
        required=True,
    )

    parser.add_argument(
        "--output-local",
        required=True,
    )

    parser.add_argument(
        "--output-drive",
        required=True,
    )

    parser.add_argument(
        "--lock-file",
        required=True,
    )

    args = parser.parse_args()

    test_manifest_path = Path(
        args.test_manifest
    ).resolve()

    sampled_pixels_path = Path(
        args.sampled_pixels
    ).resolve()

    mc_checkpoint_path = Path(
        args.mc_checkpoint
    ).resolve()

    ensemble_checkpoint_paths = [
        Path(p).resolve()
        for p in args.ensemble_checkpoints
    ]

    output_local = Path(
        args.output_local
    ).resolve()

    output_drive = Path(
        args.output_drive
    ).resolve()

    lock_file = Path(
        args.lock_file
    ).resolve()

    # Preconditions

    for p in [
        test_manifest_path,
        sampled_pixels_path,
        mc_checkpoint_path,
        lock_file,
        *ensemble_checkpoint_paths,
    ]:

        if not p.is_file():

            raise FileNotFoundError(
                p
            )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for frozen R2.6 test inference."
        )

    device = torch.device(
        "cuda"
    )

    seed_everything(
        MC_INFERENCE_SEED
    )

    torch.backends.cudnn.benchmark = False

    try:

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    except Exception:

        pass

    start_time = time.time()

    # Load frozen test inputs

    test_manifest = pd.read_csv(
        test_manifest_path
    )

    pixels = pd.read_csv(
        sampled_pixels_path
    )

    if len(test_manifest) != EXPECTED_TEST_SLICES:

        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SLICES} test slices, "
            f"found {len(test_manifest)}."
        )

    if (
        test_manifest["volume_id"].nunique()
        !=
        EXPECTED_TEST_VOLUMES
    ):

        raise RuntimeError(
            "Unexpected number of test volumes."
        )

    if len(pixels) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS} sampled pixels, "
            f"found {len(pixels)}."
        )

    if (
        pixels["volume_id"].nunique()
        !=
        EXPECTED_TEST_VOLUMES
    ):

        raise RuntimeError(
            "Unexpected sampled-pixel volume count."
        )

    if (
        pixels["sample_id"].nunique()
        !=
        EXPECTED_TEST_SLICES
    ):

        raise RuntimeError(
            "Unexpected sampled-pixel slice count."
        )

    # Every frozen test slice contributes exactly 1572 pixels.
    counts = (
        pixels["sample_id"]
        .value_counts()
    )

    if not (
        counts
        ==
        EXPECTED_PIXELS_PER_SLICE
    ).all():

        raise RuntimeError(
            "The frozen sampled-pixel population is not "
            "1572 pixels per slice."
        )

    manifest_ids = set(
        test_manifest[
            "sample_id"
        ].astype(str)
    )

    pixel_ids = set(
        pixels[
            "sample_id"
        ].astype(str)
    )

    if manifest_ids != pixel_ids:

        raise RuntimeError(
            "Test manifest/sample-table sample IDs do not match exactly."
        )

    # Load frozen models

    mc_model = load_mc_model(
        mc_checkpoint_path,
        device,
    )

    ensemble_models = [
        load_ensemble_model(
            path,
            device,
        )
        for path in ensemble_checkpoint_paths
    ]

    if len(ensemble_models) != ENSEMBLE_SIZE:

        raise RuntimeError(
            "Expected exactly three ensemble members."
        )

    # Preserve the frozen pixel-table row order.

    n_rows = len(
        pixels
    )

    uncertainty_mc = np.empty(
        n_rows,
        dtype=np.float32,
    )

    uncertainty_ens = np.empty(
        n_rows,
        dtype=np.float32,
    )

    # groupby.indices returns positional row indices.
    sample_positions = (
        pixels.groupby(
            "sample_id",
            sort=False,
        ).indices
    )

    pixel_index_all = (
        pixels[
            "pixel_linear_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    # Check flattened index alignment against stored descriptor columns.
    xhat_existing = (
        pixels[
            "xhat_magnitude"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    intervention_existing = (
        pixels[
            "intervention_magnitude"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    psf_gain_existing = (
        pixels[
            "psf_gain_channel"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    slice_rows = []

    max_diff_xhat = 0.0
    max_diff_intervention = 0.0
    max_diff_psf_gain = 0.0

    # Frozen test inference

    for row in tqdm(
        test_manifest.itertuples(
            index=False
        ),
        total=len(test_manifest),
        desc="R2.6 frozen test inference",
    ):

        sample_id = str(
            row.sample_id
        )

        npz_path = Path(
            row.npz_path
        )

        if not npz_path.is_file():

            raise FileNotFoundError(
                npz_path
            )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as z:

            if "x" not in z.files:

                raise RuntimeError(
                    f"x missing from {npz_path}"
                )

            x_np = z[
                "x"
            ].astype(
                np.float32
            )

        if (
            x_np.ndim != 3
            or
            x_np.shape[0] != 6
        ):

            raise RuntimeError(
                f"Unexpected x shape for {sample_id}: "
                f"{x_np.shape}"
            )

        H, W = (
            int(x_np.shape[1]),
            int(x_np.shape[2]),
        )

        positions = np.asarray(
            sample_positions[
                sample_id
            ],
            dtype=np.int64,
        )

        if (
            positions.size
            !=
            EXPECTED_PIXELS_PER_SLICE
        ):

            raise RuntimeError(
                f"{sample_id}: expected "
                f"{EXPECTED_PIXELS_PER_SLICE} sampled pixels, "
                f"found {positions.size}"
            )

        pixel_indices = (
            pixel_index_all[
                positions
            ]
        )

        if (
            pixel_indices.min() < 0
            or
            pixel_indices.max()
            >=
            H * W
        ):

            raise RuntimeError(
                f"Out-of-range pixel index for {sample_id}"
            )

        # Exact row/index alignment check against stored Paper-1 channels

        xhat_now = (
            x_np[0]
            .reshape(-1)[
                pixel_indices
            ]
            .astype(
                np.float64
            )
        )

        intervention_now = (
            x_np[2]
            .reshape(-1)[
                pixel_indices
            ]
            .astype(
                np.float64
            )
        )

        psf_gain_now = (
            x_np[5]
            .reshape(-1)[
                pixel_indices
            ]
            .astype(
                np.float64
            )
        )

        diff_xhat = float(
            np.max(
                np.abs(
                    xhat_now
                    -
                    xhat_existing[
                        positions
                    ]
                )
            )
        )

        diff_intervention = float(
            np.max(
                np.abs(
                    intervention_now
                    -
                    intervention_existing[
                        positions
                    ]
                )
            )
        )

        diff_psf_gain = float(
            np.max(
                np.abs(
                    psf_gain_now
                    -
                    psf_gain_existing[
                        positions
                    ]
                )
            )
        )

        max_diff_xhat = max(
            max_diff_xhat,
            diff_xhat,
        )

        max_diff_intervention = max(
            max_diff_intervention,
            diff_intervention,
        )

        max_diff_psf_gain = max(
            max_diff_psf_gain,
            diff_psf_gain,
        )

        if (
            diff_xhat
            >
            ALIGNMENT_TOLERANCE
            or
            diff_intervention
            >
            ALIGNMENT_TOLERANCE
            or
            diff_psf_gain
            >
            ALIGNMENT_TOLERANCE
        ):

            raise RuntimeError(
                f"Frozen pixel-index alignment failed for {sample_id}: "
                f"xhat={diff_xhat}, "
                f"intervention={diff_intervention}, "
                f"psf_gain={diff_psf_gain}"
            )

        # Apply the frozen A4 input-channel mask.

        x = torch.from_numpy(
            x_np
        )[None, ...].to(
            device
        )

        input_a4 = apply_a4_input(
            x
        )

        index_tensor = torch.as_tensor(
            pixel_indices,
            dtype=torch.long,
            device=device,
        )

        # MC-dropout variance:
        # U_MC = population variance over T=20 stochastic predictions

        mc_predictions = []

        with torch.no_grad():

            for _ in range(
                MC_PASSES
            ):

                prediction_mc_t = (
                    mc_model(
                        input_a4
                    )
                )

                if not torch.isfinite(
                    prediction_mc_t
                ).all():

                    raise RuntimeError(
                        f"Non-finite MC prediction: {sample_id}"
                    )

                mc_predictions.append(
                    prediction_mc_t
                )

        mc_stack = torch.stack(
            mc_predictions,
            dim=0,
        )

        U_MC = torch.var(
            mc_stack,
            dim=0,
            unbiased=False,
        )

        # Deep-ensemble variance:
        # U_ENS = population variance over M=3 frozen members

        ensemble_predictions = []

        with torch.no_grad():

            for model in ensemble_models:

                prediction_ens_m = (
                    model(
                        input_a4
                    )
                )

                if not torch.isfinite(
                    prediction_ens_m
                ).all():

                    raise RuntimeError(
                        f"Non-finite ensemble prediction: {sample_id}"
                    )

                ensemble_predictions.append(
                    prediction_ens_m
                )

        ens_stack = torch.stack(
            ensemble_predictions,
            dim=0,
        )

        U_ENS = torch.var(
            ens_stack,
            dim=0,
            unbiased=False,
        )

        if not torch.isfinite(
            U_MC
        ).all():

            raise RuntimeError(
                f"Non-finite U_MC: {sample_id}"
            )

        if not torch.isfinite(
            U_ENS
        ).all():

            raise RuntimeError(
                f"Non-finite U_ENS: {sample_id}"
            )

        # Select the pre-specified 1,572 pixels per slice.

        mc_values = (
            U_MC
            .reshape(-1)
            .index_select(
                0,
                index_tensor,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        ens_values = (
            U_ENS
            .reshape(-1)
            .index_select(
                0,
                index_tensor,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        uncertainty_mc[
            positions
        ] = mc_values

        uncertainty_ens[
            positions
        ] = ens_values

        slice_rows.append(
            {
                "sample_id":
                    sample_id,

                "volume_id":
                    str(
                        row.volume_id
                    ),

                "n_sampled_pixels":
                    int(
                        positions.size
                    ),

                "height":
                    H,

                "width":
                    W,

                "uncertainty_mc_mean":
                    float(
                        np.mean(
                            mc_values,
                            dtype=np.float64,
                        )
                    ),

                "uncertainty_ens_mean":
                    float(
                        np.mean(
                            ens_values,
                            dtype=np.float64,
                        )
                    ),

                "max_alignment_diff_xhat":
                    diff_xhat,

                "max_alignment_diff_intervention":
                    diff_intervention,

                "max_alignment_diff_psf_gain":
                    diff_psf_gain,
            }
        )

        del (
            x,
            input_a4,
            index_tensor,
            mc_predictions,
            mc_stack,
            U_MC,
            ensemble_predictions,
            ens_stack,
            U_ENS,
        )

    # Global inference validation

    if not np.isfinite(
        uncertainty_mc
    ).all():

        raise RuntimeError(
            "Non-finite values in final U_MC vector."
        )

    if not np.isfinite(
        uncertainty_ens
    ).all():

        raise RuntimeError(
            "Non-finite values in final U_ENS vector."
        )

    if not (
        uncertainty_mc >= 0
    ).all():

        raise RuntimeError(
            "Negative values in U_MC."
        )

    if not (
        uncertainty_ens >= 0
    ).all():

        raise RuntimeError(
            "Negative values in U_ENS."
        )

    # Build aligned table using Paper-1 frozen row order

    aligned = pixels.copy()

    # Explicit notation-preserving alias.
    aligned[
        "u_hold"
    ] = aligned[
        "target"
    ]

    aligned[
        "high_risk_label"
    ] = (
        aligned[
            "u_hold"
        ]
        >=
        TAU_HOLD
    ).astype(
        np.int8
    )

    aligned[
        "uncertainty_mc"
    ] = uncertainty_mc

    aligned[
        "uncertainty_ens"
    ] = uncertainty_ens

    if len(aligned) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Final aligned row count changed."
        )

    # Save aligned inference outputs.

    if output_local.exists():

        shutil.rmtree(
            output_local
        )

    output_local.mkdir(
        parents=True,
        exist_ok=False,
    )

    aligned_path = (
        output_local /
        "R2_6_TEST_SAMPLED_UNCERTAINTY.csv.gz"
    )

    arrays_path = (
        output_local /
        "R2_6_TEST_UNCERTAINTY_ARRAYS.npz"
    )

    slice_audit_path = (
        output_local /
        "R2_6_TEST_INFERENCE_SLICE_AUDIT.csv"
    )

    summary_path = (
        output_local /
        "R2_6_TEST_INFERENCE_SUMMARY.json"
    )

    aligned.to_csv(
        aligned_path,
        index=False,
        compression="gzip",
    )

    np.savez_compressed(
        arrays_path,
        uncertainty_mc=
            uncertainty_mc,

        uncertainty_ens=
            uncertainty_ens,

        high_risk_label=
            aligned[
                "high_risk_label"
            ].to_numpy(
                dtype=np.int8
            ),
    )

    slice_df = pd.DataFrame(
        slice_rows
    )

    slice_df.to_csv(
        slice_audit_path,
        index=False,
    )

    # Inference metadata; performance metrics are evaluated separately.

    elapsed_seconds = float(
        time.time()
        -
        start_time
    )

    summary = {
        "experiment":
            "R2.6",

        "status":
            "TEST_INFERENCE_COMPLETE_FROZEN",

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "device":
            str(device),

        "gpu":
            torch.cuda.get_device_name(
                0
            ),

        "mc_inference_seed":
            MC_INFERENCE_SEED,

        "mc_passes":
            MC_PASSES,

        "ensemble_members":
            ENSEMBLE_SIZE,

        "uncertainty_mc_definition":
            (
                "population variance across "
                "20 stochastic MC-dropout predictions"
            ),

        "uncertainty_ens_definition":
            (
                "population variance across "
                "3 independently trained ensemble predictions"
            ),

        "tau_hold":
            TAU_HOLD,

        "sampled_test_rows":
            int(
                len(aligned)
            ),

        "test_slices":
            int(
                aligned[
                    "sample_id"
                ].nunique()
            ),

        "test_volumes":
            int(
                aligned[
                    "volume_id"
                ].nunique()
            ),

        "pixels_per_slice":
            EXPECTED_PIXELS_PER_SLICE,

        "maximum_alignment_differences": {
            "xhat_magnitude":
                max_diff_xhat,

            "intervention_magnitude":
                max_diff_intervention,

            "psf_gain_channel":
                max_diff_psf_gain,
        },

        "alignment_tolerance":
            ALIGNMENT_TOLERANCE,

        "outputs": {
            "aligned_table":
                str(
                    aligned_path
                ),

            "aligned_table_sha256":
                sha256_file(
                    aligned_path
                ),

            "uncertainty_arrays":
                str(
                    arrays_path
                ),

            "uncertainty_arrays_sha256":
                sha256_file(
                    arrays_path
                ),

            "slice_audit":
                str(
                    slice_audit_path
                ),

            "slice_audit_sha256":
                sha256_file(
                    slice_audit_path
                ),
        },

        "elapsed_seconds":
            elapsed_seconds,

        "performance_metrics_computed":
            False,

        "post_test_tuning_performed":
            False,

        "development_reopened":
            False,
    }

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # Persist inference outputs.

    output_drive.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        aligned_path,
        arrays_path,
        slice_audit_path,
        summary_path,
    ]:

        shutil.copy2(
            path,
            output_drive /
            path.name,
        )

    drive_summary = (
        output_drive /
        summary_path.name
    )

    print(
        "\n" +
        "=" * 120
    )

    print(
        "R2.6 FROZEN TEST INFERENCE COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "Test slices           :",
        EXPECTED_TEST_SLICES
    )

    print(
        "Test volumes          :",
        EXPECTED_TEST_VOLUMES
    )

    print(
        "Sampled pixels        :",
        EXPECTED_TEST_ROWS
    )

    print(
        "MC passes             :",
        MC_PASSES
    )

    print(
        "Ensemble members      :",
        ENSEMBLE_SIZE
    )

    print(
        "MC inference seed     :",
        MC_INFERENCE_SEED
    )

    print(
        "Max alignment xhat    :",
        max_diff_xhat
    )

    print(
        "Max alignment interv. :",
        max_diff_intervention
    )

    print(
        "Max alignment psf     :",
        max_diff_psf_gain
    )

    print(
        "\nAligned table SHA256  :",
        sha256_file(
            output_drive /
            aligned_path.name
        )
    )

    print(
        "Arrays SHA256         :",
        sha256_file(
            output_drive /
            arrays_path.name
        )
    )

    print(
        "Slice audit SHA256    :",
        sha256_file(
            output_drive /
            slice_audit_path.name
        )
    )

    print(
        "Summary SHA256        :",
        sha256_file(
            drive_summary
        )
    )

    print(
        "\nPerformance metrics   : NOT COMPUTED"
    )

    print(
        "Post-test tuning      : NONE"
    )

    print(
        "Development reopened  : NO"
    )

    print(
        "\nNext authorized step  : "
        "frozen AUPRC/AUROC/Spearman evaluation "
        "and paired volume bootstrap"
    )


if __name__ == "__main__":
    main()
