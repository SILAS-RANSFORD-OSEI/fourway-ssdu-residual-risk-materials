import argparse
import json
from pathlib import Path

import pandas as pd

from fourway_mri.config import load_yaml, ensure_dir
from fourway_mri.fastmri_io import inspect_fastmri_h5, mark_main_usability


def assign_volume_splits(
    df: pd.DataFrame,
    train_count: int,
    calibration_count: int,
    test_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Assign volume-level train/calibration/test splits.

    This split is volume-level. If patient IDs are unavailable, the script
    records that limitation explicitly in the manifest and summary.
    """
    n_required = train_count + calibration_count + test_count
    n_available = len(df)

    if n_available < n_required:
        raise ValueError(
            f"Not enough usable volumes for requested split. "
            f"Required {n_required}, available {n_available}."
        )

    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    split_labels = (
        ["train"] * train_count
        + ["calibration"] * calibration_count
        + ["test"] * test_count
    )

    split_df = shuffled.iloc[:n_required].copy()
    split_df["split"] = split_labels

    unused_df = shuffled.iloc[n_required:].copy()
    if len(unused_df) > 0:
        unused_df["split"] = "unused"

    out = pd.concat([split_df, unused_df], axis=0).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 001: data audit and volume-level manifest."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp001_data_audit_manifest.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to extracted fastMRI brain multicoil HDF5 folder.",
    )

    args = parser.parse_args()
    cfg = load_yaml(args.config)

    data_root = Path(args.data_root).expanduser()
    output_dir = ensure_dir(cfg["outputs"]["output_dir"])
    ensure_dir(Path(cfg["outputs"]["full_manifest_csv"]).parent)
    ensure_dir(Path(cfg["outputs"]["main_manifest_csv"]).parent)
    ensure_dir(Path(cfg["outputs"]["split_csv"]).parent)

    h5_files = sorted(data_root.glob("*.h5"))

    rows = []
    for path in h5_files:
        row = inspect_fastmri_h5(path)
        row = mark_main_usability(
            row,
            main_acquisition=cfg["selection"]["main_acquisition"],
            require_kspace=bool(cfg["selection"]["require_kspace"]),
            require_multicoil=bool(cfg["selection"]["require_multicoil"]),
            min_num_coils=int(cfg["selection"]["min_num_coils"]),
        )
        rows.append(row)

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(cfg["outputs"]["full_manifest_csv"], index=False)

    main_df = manifest_df[manifest_df["usable_main"] == True].copy()
    main_df.to_csv(cfg["outputs"]["main_manifest_csv"], index=False)

    acquisition_counts = (
        manifest_df["acquisition"]
        .value_counts(dropna=False)
        .rename_axis("acquisition")
        .reset_index(name="count")
    )
    acquisition_counts.to_csv(cfg["outputs"]["acquisition_counts_csv"], index=False)

    kspace_shape_counts = (
        manifest_df["kspace_shape"]
        .value_counts(dropna=False)
        .rename_axis("kspace_shape")
        .reset_index(name="count")
    )
    kspace_shape_counts.to_csv(cfg["outputs"]["kspace_shape_counts_csv"], index=False)

    split_df = assign_volume_splits(
        main_df,
        train_count=int(cfg["split"]["train_count"]),
        calibration_count=int(cfg["split"]["calibration_count"]),
        test_count=int(cfg["split"]["test_count"]),
        random_seed=int(cfg["split"]["random_seed"]),
    )
    split_df.to_csv(cfg["outputs"]["split_csv"], index=False)

    patient_id_available_count = int(main_df["patient_id_available"].sum())

    split_counts = (
        split_df["split"]
        .value_counts()
        .rename_axis("split")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "experiment_name": cfg["experiment"]["name"],
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "num_h5_files_found": int(len(h5_files)),
        "num_readable_files": int(manifest_df["readable"].sum()),
        "main_acquisition": cfg["selection"]["main_acquisition"],
        "num_usable_main_volumes": int(len(main_df)),
        "patient_id_available_count": patient_id_available_count,
        "patient_level_split_possible": bool(patient_id_available_count == len(main_df)),
        "volume_level_split_used": True,
        "split_counts": split_counts,
        "unique_kspace_shapes_main": sorted(main_df["kspace_shape"].dropna().unique().tolist()),
        "unique_num_coils_main": sorted(main_df["num_coils"].dropna().astype(int).unique().tolist()),
        "unique_num_slices_main": sorted(main_df["num_slices"].dropna().astype(int).unique().tolist()),
        "outputs": cfg["outputs"],
    }

    with open(cfg["outputs"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Experiment 001: Data Audit and Volume-Level Manifest")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"HDF5 files found: {len(h5_files)}")
    print(f"Readable files: {manifest_df['readable'].sum()}")
    print(f"Main acquisition: {cfg['selection']['main_acquisition']}")
    print(f"Usable main volumes: {len(main_df)}")
    print("Split counts:")
    print(split_df["split"].value_counts())
    print(f"Full manifest: {cfg['outputs']['full_manifest_csv']}")
    print(f"Main manifest: {cfg['outputs']['main_manifest_csv']}")
    print(f"Split file: {cfg['outputs']['split_csv']}")
    print(f"Summary: {cfg['outputs']['summary_json']}")


if __name__ == "__main__":
    main()
