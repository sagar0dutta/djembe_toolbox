"""World-position transforms for skeleton visualization (BVH coordinates).

Horizontal root center (X/Z only) keeps vertical motion and a fixed ground at Y=0.
No per-frame foot minimum — Y is left as in the capture (up = BVH Y axis).
"""

import os

import fcntl
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = "Hips"
DEFAULT_LEFT_SHOULDER = "LeftShoulder"
DEFAULT_RIGHT_SHOULDER = "RightShoulder"
DEFAULT_FRONTAL_METHOD = "frame_smooth"
DEFAULT_MARKERS = "both"
DEFAULT_FPS = 240
DEFAULT_SMOOTH_SEC = 0.4

# Lateral (left-minus-right) marker pairs that define body facing.
# "both" averages shoulder line + hip line as unit vectors -> most stable.
LATERAL_PAIRS = {
    "shoulders": [("LeftShoulder", "RightShoulder")],
    "hips": [("LeftHip", "RightHip")],
    "both": [("LeftShoulder", "RightShoulder"), ("LeftHip", "RightHip")],
}

# Frontal-alignment strategies compared in testing.ipynb:
#   "none"         - hip-center only, no yaw (shows raw facing)
#   "mean"         - one yaw from the mean lateral vector over the frames passed in
#                    (whole clip if full df -> current production; a window if sliced -> "A")
#   "frame"        - per-frame yaw, always frontal, removes all turning (JYU 'frame') -> "B"
#   "frame_smooth" - per-frame yaw low-passed over smooth_sec -> always frontal,
#                    no high-freq jitter, keeps slow natural sway -> "C" (recommended)
FRONTAL_METHODS = ("none", "mean", "frame", "frame_smooth")


class file_lock:
    """Process-wide exclusive lock via a lock file (safe for parallel composite jobs)."""

    def __init__(self, lock_path):
        self.lock_path = lock_path
        self._fd = None

    def __enter__(self):
        lock_dir = os.path.dirname(self.lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        self._fd = open(self.lock_path, "w")
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._fd.close()
        self._fd = None
        return False


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


def lateral_dir(pos, labels, markers=DEFAULT_MARKERS):
    """Per-frame (x, z) lateral body vector (left-minus-right), unit-summed over pairs.

    For ``"both"`` the shoulder line and hip line are each normalized to unit length
    before summing, so neither dominates and torso twist is averaged out.
    Returns array of shape (n_frames, 2): columns are X and Z.
    """
    pairs = LATERAL_PAIRS[markers]
    out = np.zeros((pos.shape[0], 2), dtype=np.float64)
    for left, right in pairs:
        v = pos[:, labels.index(left), :] - pos[:, labels.index(right), :]
        vx, vz = v[:, 0], v[:, 2]
        n = np.hypot(vx, vz)
        n[n == 0] = 1.0
        out[:, 0] += vx / n
        out[:, 1] += vz / n
    return out


def lateral_yaw(pos, labels, markers=DEFAULT_MARKERS):
    """Per-frame yaw (radians) of the lateral vector measured from +X."""
    d = lateral_dir(pos, labels, markers)
    return np.arctan2(d[:, 1], d[:, 0])


def smooth_angle(theta, fps=DEFAULT_FPS, smooth_sec=DEFAULT_SMOOTH_SEC):
    """Low-pass an angle series (radians) via unwrap + centered moving average."""
    from scipy.ndimage import uniform_filter1d

    win = max(1, int(round(fps * smooth_sec)))
    return uniform_filter1d(np.unwrap(theta), size=win, mode="nearest")


def apply_frontal(
    pos,
    labels,
    markers=DEFAULT_MARKERS,
    method=DEFAULT_FRONTAL_METHOD,
    fps=DEFAULT_FPS,
    smooth_sec=DEFAULT_SMOOTH_SEC,
):
    """Yaw-align ``pos`` (n_frames, n_markers, 3) so the lateral vector -> +X.

    The frames passed in define the alignment window: pass a sliced window for the
    per-window ``"mean"`` strategy, or the whole clip for production-style global mean.
    """
    if method == "none":
        return pos.copy()
    if method == "mean":
        d = lateral_dir(pos, labels, markers)
        theta = np.arctan2(d[:, 1].mean(), d[:, 0].mean())
        return rotate_y(pos, theta)

    theta = lateral_yaw(pos, labels, markers)
    if method == "frame_smooth":
        theta = smooth_angle(theta, fps=fps, smooth_sec=smooth_sec)
    elif method != "frame":
        raise ValueError(f"Unknown frontal_method: {method}")

    out = pos.copy()
    for t in range(out.shape[0]):
        out[t] = rotate_y(out[t], theta[t])
    return out


def frontal_residual_deg(pos, labels, markers=DEFAULT_MARKERS):
    """Remaining facing error (deg in [-180, 180]); 0 == lateral aligned to +X (front)."""
    return np.degrees(lateral_yaw(pos, labels, markers))


def frontal_tag(markers=DEFAULT_MARKERS, frontal_method=DEFAULT_FRONTAL_METHOD,
                smooth_sec=DEFAULT_SMOOTH_SEC):
    """Short cache tag so each option gets its own CSV (switching never reuses stale data)."""
    if frontal_method == "frame_smooth":
        return f"{markers}_{frontal_method}{smooth_sec:g}"
    return f"{markers}_{frontal_method}"


def root_centered_worldpos_filename(base_name, markers=DEFAULT_MARKERS,
                                    frontal_method=DEFAULT_FRONTAL_METHOD,
                                    smooth_sec=DEFAULT_SMOOTH_SEC):
    tag = frontal_tag(markers, frontal_method, smooth_sec)
    return f"{base_name}_worldpos_rc_{tag}.csv"


def root_centered_worldpos_path(base_name, dir_csv="bvh_to_csv_centered",
                                markers=DEFAULT_MARKERS,
                                frontal_method=DEFAULT_FRONTAL_METHOD,
                                smooth_sec=DEFAULT_SMOOTH_SEC):
    return os.path.join(
        dir_csv,
        root_centered_worldpos_filename(base_name, markers, frontal_method, smooth_sec),
    )


def transform_worldpos_to_root_centered(
    df,
    root=DEFAULT_ROOT,
    markers=DEFAULT_MARKERS,
    frontal_method=DEFAULT_FRONTAL_METHOD,
    fps=DEFAULT_FPS,
    smooth_sec=DEFAULT_SMOOTH_SEC,
):
    """
    Horizontal root center + frontal yaw in BVH space (Y up, Z forward).

    - X/Z: per-frame hip subtract (removes floor-plane drift / translation)
    - Y: unchanged (fixed ground at Y=0 in capture coordinates; no foot-min shift)
    - yaw: ``frontal_method`` over the chosen lateral ``markers`` (see FRONTAL_METHODS)

    NOTE: ``"mean"`` aligns from the mean over *all frames in df*. To get the
    per-window behaviour, transform a time-sliced df (the visualizer renders
    short windows, so per-window alignment must happen at render time).
    """
    labels = marker_names_from_columns(df.columns)
    if root not in labels:
        raise ValueError(f"Root marker '{root}' not in CSV")
    if markers not in LATERAL_PAIRS:
        raise ValueError(f"Unknown markers '{markers}', expected one of {list(LATERAL_PAIRS)}")
    for left, right in LATERAL_PAIRS[markers]:
        for marker in (left, right):
            if marker not in labels:
                raise ValueError(f"Marker '{marker}' not in CSV")

    pos = stack_markers(df, labels)
    hips = pos[:, labels.index(root), :]
    pos = root_center_horizontal(pos, hips)
    pos = apply_frontal(
        pos, labels, markers=markers, method=frontal_method, fps=fps, smooth_sec=smooth_sec
    )
    return unstack_markers(df, labels, pos)


def ensure_root_centered_worldpos_csv(
    source_worldpos_csv,
    output_csv=None,
    dir_csv="bvh_to_csv_centered",
    force=False,
    debug=False,
    root=DEFAULT_ROOT,
    markers=DEFAULT_MARKERS,
    frontal_method=DEFAULT_FRONTAL_METHOD,
    fps=DEFAULT_FPS,
    smooth_sec=DEFAULT_SMOOTH_SEC,
):
    """
    Build or load cached root-centered worldpos CSV for a given frontal option.

    Raw `_worldpos.csv` is read from `source_worldpos_csv`; output goes to
    `dir_csv` (default ``bvh_to_csv_centered``). The output filename encodes
    ``markers``/``frontal_method``/``smooth_sec`` so different options never
    collide and switching options rebuilds (instead of reusing a stale CSV).

    Returns path to the root-centered CSV.
    """
    if output_csv is None:
        base_name = os.path.basename(source_worldpos_csv).replace("_worldpos.csv", "")
        output_csv = root_centered_worldpos_path(
            base_name, dir_csv, markers=markers,
            frontal_method=frontal_method, smooth_sec=smooth_sec,
        )

    lock_path = output_csv + ".lock"
    with file_lock(lock_path):
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
            print(f"Building root-centered worldpos ({frontal_tag(markers, frontal_method, smooth_sec)}): {output_csv}")

        raw_df = pd.read_csv(source_worldpos_csv)
        transformed_df = transform_worldpos_to_root_centered(
            raw_df, root=root, markers=markers, frontal_method=frontal_method,
            fps=fps, smooth_sec=smooth_sec,
        )
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        transformed_df.to_csv(output_csv, index=False)
    return output_csv
