import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fourway_mri.config import load_yaml, ensure_dir
from fourway_mri.masks import generate_fourway_partition, indices_to_binary_mask


def make_example_figure(records, output_path: str, max_examples: int = 9) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = records[:max_examples]

    fig_height = max(3, len(examples) * 1.2)
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, fig_height))

    if len(examples) == 1:
        axes = [axes]

    for ax, rec in zip(axes, examples):
        width = int(rec["width"])

        rows = np.vstack(
            [
                indices_to_binary_mask(width, rec["theta"]),
                indices_to_binary_mask(width, rec["lambda_rec"]),
                indices_to_binary_mask(width, rec["lambda_risk"]),
                indices_to_binary_mask(width, rec["lambda_hold"]),
            ]
        )

        ax.imshow(rows, aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Theta", "Lambda_rec", "Lambda_risk", "Lambda_hold"])
        ax.set_title(
            f"{rec['volume_id']} | split={rec['split']} | W={width} | "
            f"R_eff={rec['effective_acceleration']:.2f}"
        )
        ax.set_xlabel("Phase-encoding line index")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 002: ACS-protected four-way k-space mask generation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp002_fourway_masks.yaml",
    )

    args = parser.parse_args()
    cfg = load_yaml(args.config)

    ensure_dir(cfg["outputs"]["output_dir"])
    ensure_dir(Path(cfg["outputs"]["mask_indices_json"]).parent)

    split_df = pd.read_csv(cfg["inputs"]["split_csv"])

    records = []

    for _, row in split_df.iterrows():
        volume_id = str(row["volume_id"])
        width = int(row["width"])

        partition = generate_fourway_partition(
            width=width,
            volume_id=volume_id,
            base_seed=int(cfg["mask"]["base_seed"]),
            target_acceleration=float(cfg["mask"]["target_acceleration"]),
            acs_lines=int(cfg["mask"]["acs_lines"]),
            density_power=float(cfg["mask"]["density_power"]),
            theta_outer_fraction=float(cfg["partition"]["theta_outer_fraction"]),
            lambda_rec_fraction=float(cfg["partition"]["lambda_rec_fraction"]),
            lambda_risk_fraction=float(cfg["partition"]["lambda_risk_fraction"]),
            lambda_hold_fraction=float(cfg["partition"]["lambda_hold_fraction"]),
        )

        partition["filename"] = str(row["filename"])
        partition["path"] = str(row["path"])
        partition["patient_id"] = str(row["patient_id"])
        partition["split"] = str(row["split"])
        partition["height"] = int(row["height"])
        partition["num_slices"] = int(row["num_slices"])
        partition["num_coils"] = int(row["num_coils"])
        partition["kspace_shape"] = str(row["kspace_shape"])

        records.append(partition)

    with open(cfg["outputs"]["mask_indices_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": cfg["experiment"]["id"],
                "experiment_name": cfg["experiment"]["name"],
                "mask_config": cfg["mask"],
                "partition_config": cfg["partition"],
                "records": records,
            },
            f,
            indent=2,
        )

    stats_columns = [
        "volume_id",
        "patient_id",
        "filename",
        "split",
        "height",
        "width",
        "num_slices",
        "num_coils",
        "kspace_shape",
        "target_acceleration",
        "effective_acceleration",
        "acs_lines",
        "count_omega",
        "count_theta",
        "count_theta_acs",
        "count_theta_outer",
        "count_lambda_rec",
        "count_lambda_risk",
        "count_lambda_hold",
    ]

    stats_df = pd.DataFrame(records)[stats_columns]
    stats_df.to_csv(cfg["outputs"]["mask_stats_csv"], index=False)

    verification_columns = [
        "volume_id",
        "split",
        "width",
        "pairwise_overlap_count",
        "union_matches_omega",
        "acs_subset_of_theta",
        "valid_index_bounds",
    ]

    verification_df = pd.DataFrame(records)[verification_columns]
    verification_df.to_csv(cfg["outputs"]["verification_csv"], index=False)

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "experiment_name": cfg["experiment"]["name"],
        "num_volumes": int(len(records)),
        "target_acceleration": float(cfg["mask"]["target_acceleration"]),
        "acs_lines": int(cfg["mask"]["acs_lines"]),
        "all_pairwise_disjoint": bool(
            (verification_df["pairwise_overlap_count"] == 0).all()
        ),
        "all_unions_match_omega": bool(
            verification_df["union_matches_omega"].all()
        ),
        "all_acs_subset_of_theta": bool(
            verification_df["acs_subset_of_theta"].all()
        ),
        "all_indices_within_bounds": bool(
            verification_df["valid_index_bounds"].all()
        ),
        "split_counts": stats_df["split"].value_counts().to_dict(),
        "width_counts": {
            str(k): int(v)
            for k, v in stats_df["width"].value_counts().sort_index().to_dict().items()
        },
        "effective_acceleration_min": float(stats_df["effective_acceleration"].min()),
        "effective_acceleration_max": float(stats_df["effective_acceleration"].max()),
        "effective_acceleration_mean": float(stats_df["effective_acceleration"].mean()),
        "outputs": cfg["outputs"],
    }

    with open(cfg["outputs"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    example_records = []
    for split_name in ["train", "calibration", "test"]:
        split_records = [r for r in records if r["split"] == split_name]
        example_records.extend(split_records[:3])

    make_example_figure(
        example_records,
        cfg["outputs"]["example_figure"],
        max_examples=9,
    )

    print("=" * 80)
    print("Experiment 002: ACS-Protected Four-Way Mask Generation")
    print("=" * 80)
    print(f"Volumes processed: {len(records)}")
    print(f"Target acceleration: {cfg['mask']['target_acceleration']}")
    print(f"ACS lines: {cfg['mask']['acs_lines']}")
    print(f"All pairwise disjoint: {summary['all_pairwise_disjoint']}")
    print(f"All unions match omega: {summary['all_unions_match_omega']}")
    print(f"All ACS subset of theta: {summary['all_acs_subset_of_theta']}")
    print(f"All indices within bounds: {summary['all_indices_within_bounds']}")
    print(f"Mask index JSON: {cfg['outputs']['mask_indices_json']}")
    print(f"Mask stats CSV: {cfg['outputs']['mask_stats_csv']}")
    print(f"Verification CSV: {cfg['outputs']['verification_csv']}")
    print(f"Summary JSON: {cfg['outputs']['summary_json']}")
    print(f"Example figure: {cfg['outputs']['example_figure']}")


if __name__ == "__main__":
    main()
