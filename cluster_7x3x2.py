"""
Build a (7, 3, 2, 101) per-cycle cluster representation from the Nov-13 cluster_data.

This operates on the dict produced by ``clustering.process_all_files`` (the
"Clustering data extraction Nov-13 Latest" output). It does NOT re-read the mocap
pickles and does NOT modify the original extraction.

Pipeline
--------
Step 1-2 (real time, per cycle):
    For each of the 7 clusters, take the per-frame mean of its member segments'
    world ``position[:, 0:3]`` and ``velocity[:, 0:3]``. COM is special: the
    precomputed ``center_of_mass`` is used as-is (not averaged from segments).
    -> (7, 3, 2, N)   # cluster, xyz, [pos=0, vel=1], frames-in-cycle (variable N)
    xyz stay in world / mocap space (meters); no phase normalization yet.

Step 3 (resample to cycle-phase grid):
    C(T) = (cycle_frame_times - cycle_start) / (cycle_end - cycle_start)  in ~[0, 1]
    X    = np.arange(0, 1.01, 0.01)                                       (101 points)
    Each scalar channel F is linearly interpolated F(C) -> F(X).
    -> (7, 3, 2, 101)  per cycle. Only the time axis changes; xyz stay world coords.

Stacking across all cycles -> (n_cycles, 7, 3, 2, 101).

Cluster membership (per user spec, Jun-3):
    0 COM  : center_of_mass (as-is)
    1 RL   : RightUpperLeg, RightLowerLeg, RightFoot
    2 LL   : LeftUpperLeg,  LeftLowerLeg,  LeftFoot
    3 RH   : RightUpperArm, RightForeArm,  RightHand
    4 LH   : LeftUpperArm,  LeftForeArm,   LeftHand
    5 body : Pelvis, T8
    6 Head : Head
"""

import os
import pickle
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import savemat


# ---------------------------------------------------------------------------
# Configuration (edit here to change cluster definitions or the phase grid)
# ---------------------------------------------------------------------------

# Fixed cluster index order. COM is special and taken from center_of_mass as-is.
CLUSTER_NAMES = ["COM", "RL", "LL", "RH", "LH", "body", "Head"]

# Member MVN segments per cluster. ``None`` means "use center_of_mass as-is".
CLUSTER_MEMBERS = {
    "COM":  None,
    "RL":   ["RightUpperLeg", "RightLowerLeg", "RightFoot"],
    "LL":   ["LeftUpperLeg",  "LeftLowerLeg",  "LeftFoot"],
    "RH":   ["RightUpperArm", "RightForeArm",  "RightHand"],
    "LH":   ["LeftUpperArm",  "LeftForeArm",   "LeftHand"],
    "body": ["Pelvis", "T8"],
    "Head": ["Head"],
}

AXIS_NAMES = ["x", "y", "z"]            # the 3 in (7, 3, 2, 101)
CHANNEL_NAMES = ["position", "velocity"]  # the 2 in (7, 3, 2, 101): pos=0, vel=1

# Per-cycle metadata carried over (aligned by cycle index with ``data``).
METADATA_KEYS = [
    "file_name", "dmode_name", "dmode_seg_idx", "dmode_start", "dmode_end",
    "cycle_idx", "cycle_start", "cycle_end",
    "location", "ensemble", "day", "rec_no", "piece",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_phase_grid(phase_start=0.0, phase_end=1.0, phase_step=0.01):
    """Fixed phase grid X, robust to float drift (0..1 step 0.01 -> 101 points)."""
    n_points = int(round((phase_end - phase_start) / phase_step)) + 1
    return np.linspace(phase_start, phase_end, n_points)


def _cluster_posvel_realtime(cluster_data, cluster_name, members, cycle_idx):
    """Return (pos, vel) each (N, 3) in world coords for one cluster, one cycle.

    COM (members is None) uses center_of_mass as-is; otherwise the per-frame mean
    of the member segments' position/velocity.
    """
    if members is None:  # COM, used as-is
        com = cluster_data["center_of_mass"]
        pos = com["position"][cycle_idx]
        vel = com["velocity"][cycle_idx]
        if pos is None or vel is None:
            return None, None
        return np.asarray(pos, dtype=float)[:, 0:3], np.asarray(vel, dtype=float)[:, 0:3]

    body_parts = cluster_data["body_parts"]
    pos_stack, vel_stack = [], []
    for seg in members:
        if seg not in body_parts:
            raise KeyError(
                f"Cluster '{cluster_name}' member '{seg}' not found in "
                f"cluster_data['body_parts']. Available: {sorted(body_parts.keys())}"
            )
        pos_stack.append(np.asarray(body_parts[seg]["position"][cycle_idx], dtype=float)[:, 0:3])
        vel_stack.append(np.asarray(body_parts[seg]["velocity"][cycle_idx], dtype=float)[:, 0:3])

    # mean over members -> (N, 3)
    pos = np.mean(np.stack(pos_stack, axis=0), axis=0)
    vel = np.mean(np.stack(vel_stack, axis=0), axis=0)
    return pos, vel


def _resample_channel(C, F, X):
    """Linear resample of scalar channel F sampled at phase C onto grid X."""
    N = len(F)
    if N == 0:
        return np.full(X.shape, np.nan)
    if N == 1:
        return np.full(X.shape, F[0], dtype=float)
    f = interp1d(C, F, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(X)


def _infer_n_cycles(cluster_data):
    """Infer cycle count from metadata or stored COM/body-part arrays."""
    if "file_name" in cluster_data and len(cluster_data["file_name"]) > 0:
        return len(cluster_data["file_name"])

    com_pos = cluster_data.get("center_of_mass", {}).get("position", [])
    if len(com_pos) > 0:
        return len(com_pos)

    body_parts = cluster_data.get("body_parts", {})
    if body_parts:
        first_part = next(iter(body_parts.values()))
        return len(first_part.get("position", []))

    return 0


def _cycle_phase_for_samples(cluster_data, cycle_idx, n_samples):
    """Return phase C for one cycle.

    Prefer stored absolute frame times when available and length-matched. If the
    combined Nov-13 ``cluster_data`` has an empty ``cycle_frame_times`` list, fall
    back to evenly spaced samples over [0, 1] so the output still has one entry
    per extracted cycle without changing the original extraction code.
    """
    if n_samples == 0:
        return np.array([], dtype=float)
    if n_samples == 1:
        return np.array([0.0], dtype=float)

    frame_times = cluster_data.get("cycle_frame_times", [])
    if cycle_idx < len(frame_times):
        t = np.asarray(frame_times[cycle_idx], dtype=float)
        if len(t) == n_samples:
            c_start = float(cluster_data["cycle_start"][cycle_idx])
            c_end = float(cluster_data["cycle_end"][cycle_idx])
            denom = c_end - c_start
            if denom > 0:
                return (t - c_start) / denom

    return np.linspace(0.0, 1.0, n_samples)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_cluster_7x3x2(
    cluster_data,
    cluster_names=CLUSTER_NAMES,
    cluster_members=CLUSTER_MEMBERS,
    phase_start=0.0,
    phase_end=1.0,
    phase_step=0.01,
    verbose=True,
):
    """Build the (n_cycles, 7, 3, 2, n_phase) cluster array from Nov-13 cluster_data.

    Parameters
    ----------
    cluster_data : dict
        Output of ``clustering.process_all_files`` (must contain ``body_parts``,
        ``center_of_mass``, ``cycle_frame_times``, ``cycle_start``, ``cycle_end``).
    cluster_names, cluster_members : list / dict
        Cluster order and membership (defaults defined at module top).
    phase_start, phase_end, phase_step : float
        Phase grid definition; defaults give X = 0..1 step 0.01 (101 points).
    verbose : bool
        Print progress / summary.

    Returns
    -------
    out : dict
        {
          "data": (n_cycles, 7, 3, 2, n_phase) float64,
          "phase_grid": (n_phase,),
          "cluster_names": [...7...],
          "axis_names": ["x","y","z"],
          "channel_names": ["position","velocity"],
          "cluster_members": {...},
          <per-cycle metadata lists aligned with data>,
        }
    """
    X = make_phase_grid(phase_start, phase_end, phase_step)
    n_phase = len(X)
    n_clusters = len(cluster_names)
    n_cycles = _infer_n_cycles(cluster_data)

    if verbose:
        print(f"[cluster_7x3x2] cycles={n_cycles}, clusters={n_clusters}, phase points={n_phase}")
        frame_times = cluster_data.get("cycle_frame_times", [])
        if len(frame_times) != n_cycles:
            print(
                "[cluster_7x3x2] NOTE: cycle_frame_times missing or length-mismatched "
                f"({len(frame_times)} vs {n_cycles}); using evenly spaced phase samples "
                "within each cycle."
            )

    data = np.empty((n_cycles, n_clusters, 3, 2, n_phase), dtype=float)
    skipped = []

    for i in range(n_cycles):
        for ci, cname in enumerate(cluster_names):
            members = cluster_members[cname]
            pos, vel = _cluster_posvel_realtime(cluster_data, cname, members, i)

            if pos is None:  # e.g. COM missing for this cycle
                data[i, ci, :, :, :] = np.nan
                if i not in skipped:
                    skipped.append(i)
                continue

            C = _cycle_phase_for_samples(cluster_data, i, len(pos))
            for a in range(3):  # x, y, z
                data[i, ci, a, 0, :] = _resample_channel(C, pos[:, a], X)  # position
                data[i, ci, a, 1, :] = _resample_channel(C, vel[:, a], X)  # velocity

    out = {
        "data": data,
        "phase_grid": X,
        "cluster_names": list(cluster_names),
        "axis_names": list(AXIS_NAMES),
        "channel_names": list(CHANNEL_NAMES),
        "cluster_members": {k: cluster_members[k] for k in cluster_names},
    }

    # Carry per-cycle metadata aligned by index.
    for key in METADATA_KEYS:
        if key in cluster_data:
            out[key] = list(cluster_data[key])

    if verbose:
        print(f"[cluster_7x3x2] data shape: {data.shape} "
              f"(n_cycles, cluster, xyz, [pos,vel], phase)")
        if skipped:
            print(f"[cluster_7x3x2] WARNING: {len(skipped)} cycle(s) had missing COM "
                  f"and were filled with NaN.")
    return out


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _to_matlab_dict(out):
    """Make a savemat-friendly copy (no Python ``None`` values)."""
    mat = {}
    for k, v in out.items():
        if k == "cluster_members":
            mm = {}
            for name, members in v.items():
                mm[name] = "center_of_mass" if members is None else list(members)
            mat[k] = mm
        else:
            mat[k] = v
    return mat


def save_cluster_7x3x2(
    out,
    out_dir="data/cluster_data",
    tag=None,
    today=None,
    save_pkl=True,
    save_mat=True,
    save_npy=False,
):
    """Save the builder output. File stem: ``cluster_7x3x2[_tag]_<today>``."""
    if today is None:
        today = datetime.now().strftime("%d%b").lower()
    stem = "cluster_7x3x2"
    if tag:
        stem = f"{stem}_{tag}"
    stem = f"{stem}_{today}"

    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    if save_pkl:
        p = os.path.join(out_dir, stem + ".pkl")
        with open(p, "wb") as f:
            pickle.dump(out, f)
        paths["pkl"] = p

    if save_mat:
        p = os.path.join(out_dir, stem + ".mat")
        savemat(p, _to_matlab_dict(out))
        paths["mat"] = p

    if save_npy:
        p = os.path.join(out_dir, stem + ".npy")
        np.save(p, out["data"])
        paths["npy"] = p

    return paths
