import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from exp009_holdout_verification import (
    load_a4_model,
    load_cache_lookup,
    load_full_reliability_model,
    load_ssdu_model,
)
from exp010_generate_figures import build_batch_lookup, generate_case_maps
from fourway_mri.ssdu_dataset import load_mask_records


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def robust_norm(x, lo=1.0, hi=99.0):
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)

    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def load_affine_table(path):
    df = pd.read_csv(path)
    table = {}

    for _, row in df.iterrows():
        table[row["predictor"]] = (
            float(row["affine_a"]),
            float(row["affine_b"]),
        )

    return table


def affine_scale(name, arr, affine_table):
    if name not in affine_table:
        return arr

    a, b = affine_table[name]
    return a * arr + b



def plot_overlay(cfg, maps, row, affine_table, output_path):
    """
    Final Figure 6 plotting function.

    Final design:
    - no super-title
    - no panel titles
    - labels only: (a)-(f)
    - manual axes placement
    - no middle white band
    - restored anatomical aspect ratio
    - tight anatomy-centered crop
    - shared 0.0-2.5 residual-risk colorbar
    """

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    from matplotlib import cm
    from matplotlib.colors import Normalize

    try:
        from scipy import ndimage
    except Exception:
        ndimage = None

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def pick(d, candidates, label):
        for k in candidates:
            if k in d and d[k] is not None:
                return np.asarray(d[k])
        raise KeyError(
            f"Could not find {label}. Tried {candidates}. "
            f"Available keys: {list(d.keys())}"
        )

    def robust_norm(x, p_low=0.5, p_high=99.5):
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x)

        if not np.any(finite):
            return np.zeros_like(x, dtype=np.float32)

        lo = np.percentile(x[finite], p_low)
        hi = np.percentile(x[finite], p_high)

        if hi <= lo:
            hi = lo + 1e-6

        return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # ------------------------------------------------------------
    # Resolve maps using actual keys produced by the pipeline
    # ------------------------------------------------------------
    recon = pick(
        maps,
        ["ssdu", "recon", "ssdu_recon", "ssdu_reconstruction", "reconstruction", "xhat", "x_hat"],
        "SSDU reconstruction",
    )

    target = pick(
        maps,
        ["holdout_target", "target", "u_holdout", "u_hold", "u_hold_v", "u_lambda_hold"],
        "independent verification target",
    )

    a4_pred = pick(
        maps,
        ["risk_a4", "a4_prediction", "a4_pred", "a4", "a4_cnn"],
        "A4 CNN prediction",
    )

    full_pred = pick(
        maps,
        ["risk_full", "full_prediction", "full_pred", "full", "full_cnn"],
        "Full CNN prediction",
    )

    gradient = pick(
        maps,
        ["image_gradient", "image_gradient_baseline", "gradient", "grad", "grad_mag"],
        "image-gradient baseline",
    )

    gain = pick(
        maps,
        ["psf_gain", "gain", "gain_envelope", "gain_envelope_baseline", "q_psf", "psf"],
        "gain-envelope baseline",
    )

    support_mask = maps.get("support_mask", None)

    recon = np.abs(recon).astype(np.float32)
    target = np.asarray(target, dtype=np.float32)
    a4_pred = np.asarray(a4_pred, dtype=np.float32)
    full_pred = np.asarray(full_pred, dtype=np.float32)
    gradient = np.asarray(gradient, dtype=np.float32)
    gain = np.asarray(gain, dtype=np.float32)

    recon01 = robust_norm(recon)

    # ------------------------------------------------------------
    # Tight anatomy-centered crop
    # ------------------------------------------------------------
    positive = recon01[recon01 > 0]

    if positive.size > 0:
        support = recon01 > np.percentile(positive, 35)
    else:
        support = recon01 > 0.06

    if support_mask is not None:
        support_mask = np.asarray(support_mask).astype(bool)
        if support_mask.shape == support.shape:
            support |= support_mask

    if ndimage is not None:
        support = ndimage.binary_closing(support, structure=np.ones((7, 7)))
        support = ndimage.binary_fill_holes(support)
        support = ndimage.binary_dilation(support, iterations=3)

    yy, xx = np.where(support)

    if yy.size == 0 or xx.size == 0:
        y0, y1 = 0, recon.shape[0]
        x0, x1 = 0, recon.shape[1]
    else:
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1

        h = y1 - y0
        w = x1 - x0

        y_pad_top = max(4, int(0.12 * h))
        y_pad_bottom = max(4, int(0.10 * h))
        x_pad = max(4, int(0.10 * w))

        y0 = max(0, y0 - y_pad_top)
        y1 = min(recon.shape[0], y1 + y_pad_bottom)
        x0 = max(0, x0 - x_pad)
        x1 = min(recon.shape[1], x1 + x_pad)

    # Limit excessive vertical field while keeping anatomical aspect.
    crop_h = y1 - y0
    crop_w = x1 - x0
    max_h = int(1.22 * crop_w)

    if crop_h > max_h:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - max_h // 2)
        y1 = min(recon.shape[0], y0 + max_h)
        y0 = max(0, y1 - max_h)

    def crop(x):
        return np.asarray(x)[y0:y1, x0:x1]

    recon_c = crop(recon01)
    target_c = crop(target)
    a4_c = crop(a4_pred)
    full_c = crop(full_pred)
    grad_c = crop(gradient)
    gain_c = crop(gain)

    # ------------------------------------------------------------
    # Manual layout: no middle white band + true image aspect
    # ------------------------------------------------------------
    vmin = 0.0
    vmax = 2.5
    overlay_alpha = float(cfg.get("figures", {}).get("overlay_alpha", 0.72))

    fig_width = 10.8

    left = 0.000
    panel_right = 0.910
    col_gap = 0.002

    panel_w = (panel_right - left - 2 * col_gap) / 3.0
    row_h = 0.500

    crop_h, crop_w = recon_c.shape[:2]
    image_aspect = crop_w / max(crop_h, 1)

    # Choose figure height so each axes box has the same aspect
    # as the cropped image. This keeps aspect="equal" without
    # producing internal white space.
    fig_height = (fig_width * panel_w) / (row_h * image_aspect)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    fig.patch.set_facecolor("white")

    x_positions = [
        left,
        left + panel_w + col_gap,
        left + 2 * (panel_w + col_gap),
    ]

    y_bottom = 0.000
    y_top = 0.500

    axes = [
        fig.add_axes([x_positions[0], y_top, panel_w, row_h]),
        fig.add_axes([x_positions[1], y_top, panel_w, row_h]),
        fig.add_axes([x_positions[2], y_top, panel_w, row_h]),
        fig.add_axes([x_positions[0], y_bottom, panel_w, row_h]),
        fig.add_axes([x_positions[1], y_bottom, panel_w, row_h]),
        fig.add_axes([x_positions[2], y_bottom, panel_w, row_h]),
    ]

    cax = fig.add_axes([0.930, 0.000, 0.026, 1.000])

    panels = [recon_c, target_c, a4_c, full_c, grad_c, gain_c]
    kinds = ["recon", "risk", "risk", "risk", "risk", "risk"]
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, panel, kind, label in zip(axes, panels, kinds, labels):
        if kind == "recon":
            ax.imshow(
                panel,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
        else:
            ax.imshow(
                recon_c,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            ax.imshow(
                panel,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                alpha=overlay_alpha,
                interpolation="nearest",
                aspect="equal",
            )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.025,
            0.965,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        )

    # ------------------------------------------------------------
    # Shared colorbar
    # ------------------------------------------------------------
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="magma")
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cbar.set_label("Log residual-risk units", fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=11, width=1.0, length=4)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0, facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.0, facecolor="white")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-manifest", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["inputs"]["reliability_cache_manifest"] = args.cache_manifest

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(cfg["inputs"]["selected_cases_csv"]).head(
        int(cfg["data"]["num_cases"])
    )

    affine_table = load_affine_table(cfg["inputs"]["conformal_summary"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_records = load_mask_records(cfg["inputs"]["mask_indices_json"])
    cache_lookup = load_cache_lookup(cfg["inputs"]["reliability_cache_manifest"])

    ssdu_model = load_ssdu_model(cfg, device)
    full_model = load_full_reliability_model(cfg, device)
    a4_model = load_a4_model(cfg, device)

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

        out_png = output_dir / f"{sample_id}_overlay.png"
        out_npz = output_dir / f"{sample_id}_overlay_maps.npz"

        plot_overlay(cfg, maps, row, affine_table, out_png)
        np.savez_compressed(out_npz, **maps)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "volume_id": row["volume_id"],
                "filename": row["filename"],
                "slice_idx": int(row["slice_idx"]),
                "target_mean": float(row.get("target_mean", np.nan)),
                "a4_auprc": float(row.get("reliability_a4_image_only_auprc", np.nan)),
                "image_gradient_auprc": float(row.get("image_gradient_auprc", np.nan)),
                "psf_gain_auprc": float(row.get("psf_gain_channel_auprc", np.nan)),
                "a4_minus_best_baseline_auprc": float(
                    row.get("a4_minus_best_baseline_auprc", np.nan)
                ),
                "figure_png": str(out_png),
                "maps_npz": str(out_npz),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "overlay_figure_manifest.csv", index=False)

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "num_cases": int(len(manifest)),
        "output_dir": str(output_dir),
        "manifest_csv": str(output_dir / "overlay_figure_manifest.csv"),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
