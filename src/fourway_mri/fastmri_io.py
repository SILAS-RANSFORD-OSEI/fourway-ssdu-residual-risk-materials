
from pathlib import Path
from typing import Any, Dict, Optional

import h5py


def decode_attr(value: Any) -> Optional[str]:
    """
    Decode an HDF5 attribute into a safe string representation.
    """
    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def inspect_fastmri_h5(path: str | Path) -> Dict[str, Any]:
    """
    Inspect one fastMRI-style HDF5 file.

    The function records metadata needed for dataset auditing and
    volume-level manifest construction. It does not load full k-space
    data into memory.

    Parameters
    ----------
    path:
        Path to one HDF5 file.

    Returns
    -------
    dict
        File-level metadata.
    """
    path = Path(path)

    row: Dict[str, Any] = {
        "filename": path.name,
        "stem": path.stem,
        "path": str(path),
        "size_mb": None,
        "readable": False,
        "has_kspace": False,
        "has_reconstruction_rss": False,
        "has_ismrmrd_header": False,
        "acquisition": None,
        "kspace_shape": None,
        "num_slices": None,
        "num_coils": None,
        "height": None,
        "width": None,
        "kspace_dtype": None,
        "volume_id": path.stem,
        "patient_id": None,
        "patient_id_available": False,
        "usable_main": False,
        "exclusion_reason": None,
        "error": None,
    }

    try:
        row["size_mb"] = round(path.stat().st_size / (1024**2), 3)

        with h5py.File(path, "r") as hf:
            row["readable"] = True

            row["acquisition"] = decode_attr(hf.attrs.get("acquisition", None))

            patient_id = hf.attrs.get("patient_id", None)
            if patient_id is not None:
                row["patient_id"] = decode_attr(patient_id)
                row["patient_id_available"] = True

            row["has_kspace"] = "kspace" in hf
            row["has_reconstruction_rss"] = "reconstruction_rss" in hf
            row["has_ismrmrd_header"] = "ismrmrd_header" in hf

            if row["has_kspace"]:
                shape = hf["kspace"].shape
                row["kspace_shape"] = str(tuple(int(x) for x in shape))
                row["kspace_dtype"] = str(hf["kspace"].dtype)

                if len(shape) == 4:
                    row["num_slices"] = int(shape[0])
                    row["num_coils"] = int(shape[1])
                    row["height"] = int(shape[2])
                    row["width"] = int(shape[3])

    except Exception as exc:
        row["error"] = str(exc)
        row["exclusion_reason"] = f"read_error: {exc}"

    return row


def mark_main_usability(
    row: Dict[str, Any],
    main_acquisition: str = "AXT2",
    require_kspace: bool = True,
    require_multicoil: bool = True,
    min_num_coils: int = 2,
) -> Dict[str, Any]:
    """
    Mark whether a file belongs to the main controlled study cohort.
    """
    if not row["readable"]:
        row["usable_main"] = False
        if row["exclusion_reason"] is None:
            row["exclusion_reason"] = "not_readable"
        return row

    if require_kspace and not row["has_kspace"]:
        row["usable_main"] = False
        row["exclusion_reason"] = "missing_kspace"
        return row

    if row["acquisition"] != main_acquisition:
        row["usable_main"] = False
        row["exclusion_reason"] = f"non_{main_acquisition}"
        return row

    if require_multicoil:
        if row["num_coils"] is None:
            row["usable_main"] = False
            row["exclusion_reason"] = "unknown_num_coils"
            return row

        if row["num_coils"] < min_num_coils:
            row["usable_main"] = False
            row["exclusion_reason"] = "not_multicoil"
            return row

    row["usable_main"] = True
    row["exclusion_reason"] = None
    return row
