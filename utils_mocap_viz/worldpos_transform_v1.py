"""World-position transforms for skeleton visualization (BVH coordinates).

Horizontal root center (X/Z only) keeps vertical motion and a fixed ground at Y=0.
No per-frame foot minimum — Y is left as in the capture (up = BVH Y axis).
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = "Hips"
DEFAULT_LEFT_SHOULDER = "LeftShoulder"
DEFAULT_RIGHT_SHOULDER = "RightShoulder"
DEFAULT_FRONTAL_METHOD = "frame"


def marker_names_from_columns(columns):
    names = []
    for col in columns:
        if col == "Time" or col.endswith("End.X"):
            continue
        if col.endswith(".X"):
            names.append(col[:-2])
    return list(dict.fromkeys(names))


def stack_markers(df, labels):
    """Return (n_frames, n_markers, 3) in column order X, Y, Z."""
    n = len(df)
    m = len(labels)
    arr = np.empty((n, m, 3), dtype=np.float64)
    for j, label in enumerate(labels):
        arr[:, j, 0] = df[f"{label}.X"].to_numpy()
        arr[:, j, 1] = df[f"{label}.Y"].to_numpy()
        arr[:, j, 2] = df[f"{label}.Z"].to_numpy()
    return arr


def unstack_markers(df_template, labels, arr):
    out = df_template.copy()
    for j, label in enumerate(labels):
        out[f"{label}.X"] = arr[:, j, 0]
        out[f"{label}.Y"] = arr[:, j, 1]
        out[f"{label}.Z"] = arr[:, j, 2]
    return out


def rotate_y(points, theta):
    """Rotate (..., 3) around Y (BVH up)."""
    rot = Rotation.from_euler("y", theta, degrees=False)
    flat = points.reshape(-1, 3)
    return rot.apply(flat).reshape(points.shape)


def root_center_horizontal(pos, hips):
    """Subtract hips X and Z per frame; keep Y (ground reference Y=0)."""
    offset = np.zeros_like(hips)
    offset[:, 0] = hips[:, 0]
    offset[:, 2] = hips[:, 2]
    return pos - offset[:, np.newaxis, :]


def root_centered_worldpos_filename(base_name):
    return f"{base_name}_worldpos_root_centered.csv"


def root_centered_worldpos_path(base_name, dir_csv="bvh_to_csv_centered"):
    return os.path.join(dir_csv, root_centered_worldpos_filename(base_name))


def transform_worldpos_to_root_centered(
    df,
    root=DEFAULT_ROOT,
    left=DEFAULT_LEFT_SHOULDER,
    right=DEFAULT_RIGHT_SHOULDER,
    frontal_method=DEFAULT_FRONTAL_METHOD,
):
    """
    Horizontal root center + frontal yaw in BVH space (Y up, Z forward).

    - X/Z: per-frame hip subtract (removes floor-plane drift / translation)
    - Y: unchanged (fixed ground at Y=0 in capture coordinates; no foot-min shift)
    """
    labels = marker_names_from_columns(df.columns)
    if root not in labels:
        raise ValueError(f"Root marker '{root}' not in CSV")
    for marker in (left, right):
        if marker not in labels:
            raise ValueError(f"Marker '{marker}' not in CSV")

    pos = stack_markers(df, labels)
    root_idx = labels.index(root)
    left_idx = labels.index(left)
    right_idx = labels.index(right)

    hips = pos[:, root_idx, :]
    pos = root_center_horizontal(pos, hips)

    shoulder = pos[:, left_idx, :] - pos[:, right_idx, :]
    vx = shoulder[:, 0]
    vz = shoulder[:, 2]

    if frontal_method == "mean":
        theta = np.arctan2(vz.mean(), vx.mean())
        pos = rotate_y(pos, theta)
    elif frontal_method == "frame":
        theta = np.arctan2(vz, vx)
        for t in range(pos.shape[0]):
            pos[t] = rotate_y(pos[t], theta[t])
    else:
        raise ValueError(f"Unknown frontal_method: {frontal_method}")

    return unstack_markers(df, labels, pos)


def ensure_root_centered_worldpos_csv(
    source_worldpos_csv,
    output_csv=None,
    dir_csv="bvh_to_csv_centered",
    force=False,
    debug=False,
    **transform_kwargs,
):
    """
    Build or load cached root-centered worldpos CSV.

    Raw `_worldpos.csv` is read from `source_worldpos_csv`; output goes to
    `dir_csv` (default ``bvh_to_csv_centered``), not necessarily beside the source.

    Returns path to the root-centered CSV.
    """
    if output_csv is None:
        base_name = os.path.basename(source_worldpos_csv).replace("_worldpos.csv", "")
        output_csv = root_centered_worldpos_path(base_name, dir_csv)

    if (
        not force
        and os.path.exists(output_csv)
        and os.path.exists(source_worldpos_csv)
        and os.path.getmtime(output_csv) >= os.path.getmtime(source_worldpos_csv)
    ):
        if debug:
            print(f"Using cached root-centered worldpos: {output_csv}")
        return output_csv

    if debug:
        print(f"Building root-centered worldpos (XZ hips, Y fixed): {output_csv}")

    raw_df = pd.read_csv(source_worldpos_csv)
    transformed_df = transform_worldpos_to_root_centered(raw_df, **transform_kwargs)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    transformed_df.to_csv(output_csv, index=False)
    return output_csv
