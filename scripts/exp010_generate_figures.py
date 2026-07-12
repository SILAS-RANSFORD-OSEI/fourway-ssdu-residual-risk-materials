import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from exp009_holdout_verification import (
    gradient_magnitude_2d,
    load_a4_model,
    load_cache_lookup,
    load_cached_input,
    load_full_reliability_model,
    load_ssdu_model,
    make_holdout_target,
    make_loader,
    prepare_batch,
)
from fourway_mri.sensitivity import indices_to_torch_mask
from fourway_mri.ssdu_dataset import load_mask_records


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def robust_limits(x, lo=1.0, hi=99.0):
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin = np.percentile(finite, lo)
    vmax = np.percentile(finite, hi)

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)


def normalize_for_display(x, lo=1.0, hi=99.0):
    vmin, vmax = robust_limits(x, lo, hi)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    return np.clip(y, 0.0, 1.0)




def select_cases(cfg):
    df = pd.read_csv(cfg["inputs"]["holdout_sample_metrics"])
    df = df[df["split"] == cfg["data"]["split"]].copy()

    selected_sample_ids = cfg.get("data", {}).get("selected_sample_ids", None)

    if selected_sample_ids:
        selected_sample_ids = [str(x) for x in selected_sample_ids]

        if "sample_id" not in df.columns:
            raise KeyError("holdout_sample_metrics must contain a sample_id column.")

        selected = df[df["sample_id"].astype(str).isin(selected_sample_ids)].copy()

        if selected.empty:
            raise RuntimeError(
                "No explicit Figure 6 sample_ids were found in holdout_sample_metrics."
            )

        order = {sid: i for i, sid in enumerate(selected_sample_ids)}
        selected["_fig6_order"] = selected["sample_id"].astype(str).map(order)
        selected = selected.sort_values("_fig6_order").drop(columns=["_fig6_order"])

        print("Using explicit Figure 6 selected sample_ids:")
        for sid in selected["sample_id"].astype(str).tolist():
            print(" -", sid)

        return selected

    metric = cfg["data"]["selection_metric"]
    if metric not in df.columns:
        raise ValueError(f"Selection metric not found: {metric}")

    ascending = cfg["data"].get("selection_order", "descending") == "ascending"
    df = df.sort_values(metric, ascending=ascending)

    selected = df.head(int(cfg["data"]["num_cases"])).copy()

    return selected


def build_batch_lookup(cfg, data_root):
    """
    Build batch lookup only for selected Figure 6 qualitative cases.
    """

    import copy
    from pathlib import Path
    import pandas as pd

    selected = select_cases(cfg)

    if "sample_id" not in selected.columns:
        raise KeyError("select_cases(cfg) must return a sample_id column.")

    wanted_sample_ids = set(selected["sample_id"].astype(str).tolist())

    selected_volume_ids = set()
    if "volume_id" in selected.columns:
        selected_volume_ids.update(selected["volume_id"].astype(str).tolist())

    selected_filenames = set()
    if "filename" in selected.columns:
        selected_filenames.update(selected["filename"].astype(str).tolist())

    for vid in selected_volume_ids:
        if vid.endswith(".h5"):
            selected_filenames.add(vid)
            selected_filenames.add(Path(vid).name)
        else:
            selected_filenames.add(vid + ".h5")

    selected_stems = {Path(f).stem for f in selected_filenames}
    selected_stems.update({Path(v).stem for v in selected_volume_ids})

    candidate_paths = []

    def collect_csv_paths(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and v.lower().endswith(".csv"):
                    key = str(k).lower()
                    val = v.lower()
                    if "split" in key or "manifest" in key or "axt2" in val:
                        candidate_paths.append(v)
                elif isinstance(v, dict):
                    collect_csv_paths(v)

    collect_csv_paths(cfg)

    fallback = "data/manifests/fastmri_brain_multicoil_axt2_splits.csv"
    candidate_paths.append(fallback)

    split_manifest_path = None

    for p_str in candidate_paths:
        pth = Path(p_str)
        if not pth.exists():
            continue

        try:
            preview = pd.read_csv(pth, nrows=5)
        except Exception:
            continue

        cols = set(preview.columns)
        if {"filename", "split"}.issubset(cols) or {"volume_id", "split"}.issubset(cols):
            split_manifest_path = pth
            break

    if split_manifest_path is None:
        raise FileNotFoundError(
            "Could not locate a split manifest with filename/volume_id and split columns."
        )

    split_df = pd.read_csv(split_manifest_path)

    split_name = cfg["data"]["split"]
    if "split" in split_df.columns:
        split_df = split_df[split_df["split"].astype(str) == str(split_name)].copy()

    keep = pd.Series(False, index=split_df.index)

    if "filename" in split_df.columns:
        filenames = split_df["filename"].astype(str)
        stems = filenames.map(lambda x: Path(x).stem)
        keep |= filenames.isin(selected_filenames)
        keep |= filenames.map(lambda x: Path(x).name).isin(selected_filenames)
        keep |= stems.isin(selected_stems)

    if "volume_id" in split_df.columns:
        volume_ids = split_df["volume_id"].astype(str)
        volume_stems = volume_ids.map(lambda x: Path(x).stem)
        keep |= volume_ids.isin(selected_volume_ids)
        keep |= volume_stems.isin(selected_stems)

    filtered = split_df[keep].copy()

    if filtered.empty:
        raise RuntimeError(
            "Filtered split manifest is empty. Selected cases could not be matched "
            "to the split manifest."
        )

    temp_manifest = Path("/content/fig6_selected_split_manifest.csv")
    filtered.to_csv(temp_manifest, index=False)

    print(f"Using reduced Figure 6 split manifest: {temp_manifest}")
    print(f"Selected rows in reduced manifest: {len(filtered)}")
    if "filename" in filtered.columns:
        print("Selected volumes:")
        for f in sorted(filtered["filename"].astype(str).unique()):
            print(" -", f)

    cfg_selected = copy.deepcopy(cfg)
    old_manifest_str = str(split_manifest_path)

    def replace_manifest_path(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    if v == old_manifest_str or Path(v).name == split_manifest_path.name:
                        obj[k] = str(temp_manifest)
                elif isinstance(v, dict):
                    replace_manifest_path(v)

    replace_manifest_path(cfg_selected)

    loader = make_loader(cfg_selected, cfg_selected["data"]["split"], data_root)
    lookup = {}

    for batch in loader:
        volume_id = batch["volume_id"][0]
        slice_idx = int(batch["slice_idx"][0].item())
        sample_id = f"{volume_id}_slice{slice_idx:03d}"
        lookup[sample_id] = batch

        if wanted_sample_ids.issubset(set(lookup.keys())):
            break

    missing = sorted(wanted_sample_ids - set(lookup.keys()))
    if missing:
        raise KeyError(
            "Some selected Figure 6 sample_ids were not found after reduced loading:\n"
            + "\n".join(missing)
        )

    return lookup


def generate_case_maps(
    cfg,
    case_row,
    batch_lookup,
    cache_lookup,
    mask_records,
    ssdu_model,
    full_model,
    a4_model,
    device,
):
    sample_id = case_row["sample_id"]
    volume_id = case_row["volume_id"]
    slice_idx = int(case_row["slice_idx"])

    if sample_id not in batch_lookup:
        raise KeyError(f"sample_id not found in batch lookup: {sample_id}")

    batch = batch_lookup[sample_id]

    width = int(batch["width"][0].item())

    if volume_id not in mask_records:
        raise KeyError(f"volume_id missing from mask records: {volume_id}")

    mask_rec = mask_records[volume_id]

    if "lambda_hold" not in mask_rec:
        raise KeyError(f"Mask record for {volume_id} has no lambda_hold field.")

    hold_mask = indices_to_torch_mask(
        width=width,
        indices=mask_rec["lambda_hold"],
        device=device,
        dtype=torch.float32,
    )

    prepared = prepare_batch(batch, device, cfg)

    with torch.no_grad():
        x_hat = ssdu_model(
            x0=prepared["x0"],
            y_theta=prepared["y_theta"],
            sensitivities=prepared["sensitivities"],
            theta_mask=prepared["theta_mask"],
            input_scale=prepared["input_scale"],
        )

        y_hold = make_holdout_target(
            x_hat=x_hat,
            prepared=prepared,
            hold_mask=hold_mask,
            cfg=cfg,
            slice_idx=slice_idx,
        )

        x_cached = load_cached_input(cache_lookup, sample_id).to(device)

        pred_full = full_model(x_cached)

        x_a4 = x_cached.clone()
        x_a4[:, 3, :, :] = 0.0
        x_a4[:, 4, :, :] = 0.0
        x_a4[:, 5, :, :] = 0.0
        pred_a4 = a4_model(x_a4)

    x_np = x_cached[0].detach().cpu().numpy().astype(np.float32)

    maps = {
        "x0": x_np[1],
        "ssdu": x_np[0],
        "intervention": x_np[2],
        "support_mask": x_np[3],
        "psf": x_np[4],
        "psf_gain": x_np[5],
        "holdout_target": y_hold[0].detach().cpu().numpy().astype(np.float32),
        "risk_a4": pred_a4[0, 0].detach().cpu().numpy().astype(np.float32),
        "risk_full": pred_full[0, 0].detach().cpu().numpy().astype(np.float32),
        "image_gradient": gradient_magnitude_2d(x_np[0]),
    }

    return maps


def plot_case_grid(cfg, maps, case_row, output_path):
    fig_cfg = cfg["figures"]

    panels = [
        ("Initial adjoint $|x_0|$", maps["x0"], "image"),
        ("SSDU reconstruction $|\\hat{x}|$", maps["ssdu"], "image"),
        ("True holdout risk $u_{hold}$", maps["holdout_target"], "risk"),
        ("A4 CNN predicted risk", maps["risk_a4"], "risk"),
        ("Full CNN predicted risk", maps["risk_full"], "risk"),
        ("Image-gradient baseline", maps["image_gradient"], "risk"),
        ("PSF/gain baseline", maps["psf_gain"], "risk"),
        ("Intervention $|\\hat{x}-x_0|$", maps["intervention"], "risk"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for ax, (title, img, kind) in zip(axes.ravel(), panels):
        if kind == "image":
            disp = normalize_for_display(
                img,
                lo=float(fig_cfg["percentile_low"]),
                hi=float(fig_cfg["percentile_high"]),
            )
            ax.imshow(disp, cmap=fig_cfg["cmap_image"])
        else:
            vmax = np.percentile(img[np.isfinite(img)], float(fig_cfg["risk_percentile_high"]))
            if vmax <= 0:
                vmax = 1.0
            ax.imshow(img, cmap=fig_cfg["cmap_risk"], vmin=0.0, vmax=vmax)

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{case_row['sample_id']} | target_hold_p99={case_row['target_hold_p99']:.4f}",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=int(fig_cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp010_generate_figures.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-manifest", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.cache_manifest is not None:
        cfg["inputs"]["reliability_cache_manifest"] = args.cache_manifest

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_records = load_mask_records(cfg["inputs"]["mask_indices_json"])
    cache_lookup = load_cache_lookup(cfg["inputs"]["reliability_cache_manifest"])

    ssdu_model = load_ssdu_model(cfg, device)
    full_model = load_full_reliability_model(cfg, device)
    a4_model = load_a4_model(cfg, device)

    selected = select_cases(cfg)
    batch_lookup = build_batch_lookup(cfg, args.data_root)

    manifest_rows = []

    for _, row in selected.iterrows():
        maps = generate_case_maps(
            cfg=cfg,
            case_row=row,
            batch_lookup=batch_lookup,
            cache_lookup=cache_lookup,
            mask_records=mask_records,
            ssdu_model=ssdu_model,
            full_model=full_model,
            a4_model=a4_model,
            device=device,
        )

        sample_id = row["sample_id"]
        out_png = output_dir / f"{sample_id}_figure.png"
        out_npz = output_dir / f"{sample_id}_maps.npz"

        plot_case_grid(cfg, maps, row, out_png)

        np.savez_compressed(out_npz, **maps)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "volume_id": row["volume_id"],
                "filename": row["filename"],
                "slice_idx": int(row["slice_idx"]),
                "target_hold_mean": float(row["target_hold_mean"]),
                "target_hold_p99": float(row["target_hold_p99"]),
                "figure_png": str(out_png),
                "maps_npz": str(out_npz),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "num_cases": int(len(manifest)),
        "selection_metric": cfg["data"]["selection_metric"],
        "outputs": {
            "figure_manifest_csv": str(output_dir / "figure_manifest.csv"),
            "output_dir": str(output_dir),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
