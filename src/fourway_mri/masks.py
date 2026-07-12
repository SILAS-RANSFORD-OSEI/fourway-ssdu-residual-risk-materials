from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np


def stable_int_seed(volume_id: str, base_seed: int) -> int:
    text = f"{base_seed}_{volume_id}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def center_acs_indices(width: int, acs_lines: int) -> np.ndarray:
    if acs_lines <= 0:
        raise ValueError("acs_lines must be positive.")

    if acs_lines >= width:
        raise ValueError(
            f"acs_lines={acs_lines} must be smaller than width={width}."
        )

    center = width // 2
    start = center - acs_lines // 2
    end = start + acs_lines

    return np.arange(start, end, dtype=int)


def variable_density_probabilities(
    width: int,
    excluded_indices: np.ndarray,
    density_power: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(width, dtype=int)
    keep = np.ones(width, dtype=bool)
    keep[excluded_indices] = False

    candidates = all_indices[keep]

    center = (width - 1) / 2.0
    distance = np.abs(candidates - center)
    distance = distance / distance.max()

    weights = (1.0 - distance) ** density_power
    weights = weights + 1e-6
    probabilities = weights / weights.sum()

    return candidates, probabilities


def allocate_partition_counts(
    n_outer_sampled: int,
    theta_outer_fraction: float,
    lambda_rec_fraction: float,
    lambda_risk_fraction: float,
    lambda_hold_fraction: float,
) -> Dict[str, int]:
    fractions = {
        "theta_outer": theta_outer_fraction,
        "lambda_rec": lambda_rec_fraction,
        "lambda_risk": lambda_risk_fraction,
        "lambda_hold": lambda_hold_fraction,
    }

    total_fraction = sum(fractions.values())
    if not np.isclose(total_fraction, 1.0):
        raise ValueError(f"Partition fractions must sum to 1. Got {total_fraction}.")

    raw = {name: frac * n_outer_sampled for name, frac in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}

    remainder = n_outer_sampled - sum(counts.values())

    fractional_order = sorted(
        raw.keys(),
        key=lambda name: raw[name] - counts[name],
        reverse=True,
    )

    for name in fractional_order[:remainder]:
        counts[name] += 1

    return counts


def verify_fourway_partition(partition: Dict) -> Dict[str, int | bool]:
    width = int(partition["width"])

    omega = set(partition["omega"])
    theta = set(partition["theta"])
    theta_acs = set(partition["theta_acs"])
    lambda_rec = set(partition["lambda_rec"])
    lambda_risk = set(partition["lambda_risk"])
    lambda_hold = set(partition["lambda_hold"])

    subsets = [theta, lambda_rec, lambda_risk, lambda_hold]

    pairwise_overlaps = 0
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            pairwise_overlaps += len(subsets[i] & subsets[j])

    union = set().union(*subsets)
    valid_bounds = all((0 <= idx < width) for idx in omega)

    return {
        "count_omega": int(len(omega)),
        "count_theta": int(len(theta)),
        "count_theta_acs": int(len(theta_acs)),
        "count_theta_outer": int(len(partition["theta_outer"])),
        "count_lambda_rec": int(len(lambda_rec)),
        "count_lambda_risk": int(len(lambda_risk)),
        "count_lambda_hold": int(len(lambda_hold)),
        "pairwise_overlap_count": int(pairwise_overlaps),
        "union_matches_omega": bool(union == omega),
        "acs_subset_of_theta": bool(theta_acs.issubset(theta)),
        "valid_index_bounds": bool(valid_bounds),
    }


def generate_fourway_partition(
    width: int,
    volume_id: str,
    base_seed: int,
    target_acceleration: float,
    acs_lines: int,
    density_power: float,
    theta_outer_fraction: float,
    lambda_rec_fraction: float,
    lambda_risk_fraction: float,
    lambda_hold_fraction: float,
) -> Dict:
    if target_acceleration <= 1.0:
        raise ValueError("target_acceleration must be greater than 1.")

    acs = center_acs_indices(width=width, acs_lines=acs_lines)

    target_omega_count = int(round(width / target_acceleration))
    if target_omega_count <= acs_lines:
        raise ValueError(
            f"Target sampled lines {target_omega_count} must exceed ACS lines {acs_lines}."
        )

    n_outer_to_sample = target_omega_count - acs_lines

    candidates, probabilities = variable_density_probabilities(
        width=width,
        excluded_indices=acs,
        density_power=density_power,
    )

    seed = stable_int_seed(volume_id=volume_id, base_seed=base_seed)
    rng = np.random.default_rng(seed)

    outer_sampled = rng.choice(
        candidates,
        size=n_outer_to_sample,
        replace=False,
        p=probabilities,
    )

    outer_sampled = np.sort(outer_sampled)

    shuffled_outer = outer_sampled.copy()
    rng.shuffle(shuffled_outer)

    counts = allocate_partition_counts(
        n_outer_sampled=n_outer_to_sample,
        theta_outer_fraction=theta_outer_fraction,
        lambda_rec_fraction=lambda_rec_fraction,
        lambda_risk_fraction=lambda_risk_fraction,
        lambda_hold_fraction=lambda_hold_fraction,
    )

    a = counts["theta_outer"]
    b = a + counts["lambda_rec"]
    c = b + counts["lambda_risk"]

    theta_outer = np.sort(shuffled_outer[:a])
    lambda_rec = np.sort(shuffled_outer[a:b])
    lambda_risk = np.sort(shuffled_outer[b:c])
    lambda_hold = np.sort(shuffled_outer[c:])

    theta = np.sort(np.concatenate([acs, theta_outer]))
    omega = np.sort(np.concatenate([theta, lambda_rec, lambda_risk, lambda_hold]))

    partition = {
        "volume_id": volume_id,
        "width": int(width),
        "seed": int(seed),
        "target_acceleration": float(target_acceleration),
        "effective_acceleration": float(width / len(omega)),
        "acs_lines": int(acs_lines),
        "omega": omega.astype(int).tolist(),
        "theta": theta.astype(int).tolist(),
        "theta_acs": acs.astype(int).tolist(),
        "theta_outer": theta_outer.astype(int).tolist(),
        "lambda_rec": lambda_rec.astype(int).tolist(),
        "lambda_risk": lambda_risk.astype(int).tolist(),
        "lambda_hold": lambda_hold.astype(int).tolist(),
    }

    partition.update(verify_fourway_partition(partition))

    return partition


def indices_to_binary_mask(width: int, indices: List[int]) -> np.ndarray:
    mask = np.zeros(width, dtype=np.uint8)
    mask[np.asarray(indices, dtype=int)] = 1
    return mask
