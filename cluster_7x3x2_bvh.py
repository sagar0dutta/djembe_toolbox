"""
Build a (7, 3, 2, 101) per-cycle cluster representation from BVH cluster_data.

Operates on the dict produced by ``clustering_bvh.process_all_files``.
Pipeline and output format are identical to cluster_7x3x2.py; only the
cluster membership definitions change to match BVH joint names.

BVH joint → MVNX segment equivalences used here
------------------------------------------------
Pelvis       -> Hips
T8           -> Chest3
RightUpperLeg-> RightHip        LeftUpperLeg -> LeftHip
RightLowerLeg-> RightKnee       LeftLowerLeg -> LeftKnee
RightFoot    -> RightAnkle      LeftFoot     -> LeftAnkle
               + RightToe                    + LeftToe
RightUpperArm-> RightShoulder   LeftUpperArm -> LeftShoulder
RightForeArm -> RightElbow      LeftForeArm  -> LeftElbow
RightHand    -> RightWrist       LeftHand    -> LeftWrist
Head         -> Head

Pipeline
--------
Step 1-2 (real time, per cycle):
    For each of the 7 clusters, per-frame mean of member joints' world
    position[:, 0:3] and velocity[:, 0:3].  COM is taken from
    cluster_data["center_of_mass"] as-is.
    -> (7, 3, 2, N)   cluster × xyz × [pos=0, vel=1] × frames (variable N)

Step 3 (resample to cycle-phase grid):
    C(T) = (cycle_frame_times - cycle_start) / (cycle_end - cycle_start)
    X    = np.arange(0, 1.01, 0.01)   (101 points)
    Each scalar channel F: linear interpolation F(C) -> F(X).
    -> (7, 3, 2, 101) per cycle.

Stacking across all cycles -> (n_cycles, 7, 3, 2, 101).

Cluster membership
------------------
    0 COM  : center_of_mass  (as-is from clustering_bvh)
    1 RL   : RightHip, RightKnee, RightAnkle, RightToe
    2 LL   : LeftHip,  LeftKnee,  LeftAnkle,  LeftToe
    3 RH   : RightShoulder, RightElbow, RightWrist
    4 LH   : LeftShoulder,  LeftElbow,  LeftWrist
    5 body : Hips, Chest3
    6 Head : Head
"""

import os
import pickle
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import savemat


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CLUSTER_NAMES = ["COM", "RL", "LL", "RH", "LH", "body", "Head"]

# None = use center_of_mass as-is (not averaged from body_parts)
CLUSTER_MEMBERS = {
    "COM":  None,
    "RL":   ["RightHip", "RightKnee", "RightAnkle", "RightToe"],
    "LL":   ["LeftHip",  "LeftKnee",  "LeftAnkle",  "LeftToe"],
    "RH":   ["RightShoulder", "RightElbow", "RightWrist"],
    "LH":   ["LeftShoulder",  "LeftElbow",  "LeftWrist"],
    "body": ["Hips", "Chest3"],
    "Head": ["Head"],
}

AXIS_NAMES    = ["x", "y", "z"]
CHANNEL_NAMES = ["position", "velocity"]

METADATA_KEYS = [
    "file_name", "dmode_name", "dmode_seg_idx", "dmode_start", "dmode_end",
    "cycle_idx", "cycle_start", "cycle_end",
    "location", "ensemble", "day", "rec_no", "piece", "dancer_id",
    "hand_clap_onsets", "both_feet_onsets", "left_foot_onsets", "right_foot_onsets",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_phase_grid(phase_start=0.0, phase_end=1.0, phase_step=0.01):
    """Fixed phase grid X robust to float drift (0..1 step 0.01 → 101 points)."""
    n_points = int(round((phase_end - phase_start) / phase_step)) + 1
    return np.linspace(phase_start, phase_end, n_points)


def _cluster_posvel_realtime(cluster_data, cluster_name, members, cycle_idx):
    """Return (pos, vel) each shape (N, 3) for one cluster, one cycle.

    COM (members=None) uses center_of_mass as-is; otherwise the per-frame
    mean of the member joints' position/velocity from body_parts.
    """
    if members is None:
        com = cluster_data["center_of_mass"]
        pos = com["position"][cycle_idx]
        vel = com["velocity"][cycle_idx]
        if pos is None or vel is None:
            return None, None
        return np.asarray(pos, dtype=float)[:, 0:3], np.asarray(vel, dtype=float)[:, 0:3]

    body_parts = cluster_data["body_parts"]
    pos_stack, vel_stack = [], []
    for jnt in members:
        if jnt not in body_parts:
            raise KeyError(
                f"Cluster '{cluster_name}' member '{jnt}' not found in "
                f"cluster_data['body_parts']. Available: {sorted(body_parts.keys())}"
            )
        pos_stack.append(
            np.asarray(body_parts[jnt]["position"][cycle_idx], dtype=float)[:, 0:3]
        )
        vel_stack.append(
            np.asarray(body_parts[jnt]["velocity"][cycle_idx], dtype=float)[:, 0:3]
        )

    pos = np.mean(np.stack(pos_stack, axis=0), axis=0)
    vel = np.mean(np.stack(vel_stack, axis=0), axis=0)
    return pos, vel


def _resample_channel(C, F, X):
    """Linear resample scalar channel F sampled at phase C onto grid X."""
    N = len(F)
    if N == 0:
        return np.full(X.shape, np.nan)
    if N == 1:
        return np.full(X.shape, F[0], dtype=float)
    f = interp1d(C, F, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(X)


def _infer_n_cycles(cluster_data):
    """Infer cycle count from metadata or stored arrays."""
    if cluster_data.get("file_name"):
        return len(cluster_data["file_name"])
    com_pos = cluster_data.get("center_of_mass", {}).get("position", [])
    if com_pos:
        return len(com_pos)
    body_parts = cluster_data.get("body_parts", {})
    if body_parts:
        first = next(iter(body_parts.values()))
        return len(first.get("position", []))
    return 0


def _cycle_phase(cluster_data, cycle_idx, n_samples):
    """Return phase C ∈ [0, 1] for one cycle.

    Uses stored absolute frame times when available; falls back to uniform
    spacing so the output always has one entry per cycle.
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
            c_end   = float(cluster_data["cycle_end"][cycle_idx])
            denom   = c_end - c_start
            if denom > 0:
                return (t - c_start) / denom

    return np.linspace(0.0, 1.0, n_samples)


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #

def build_cluster_7x3x2(
    cluster_data,
    cluster_names=CLUSTER_NAMES,
    cluster_members=CLUSTER_MEMBERS,
    phase_start=0.0,
    phase_end=1.0,
    phase_step=0.01,
    verbose=True,
):
    """Build the (n_cycles, 7, 3, 2, n_phase) BVH cluster array.

    Parameters
    ----------
    cluster_data    : dict   output of ``clustering_bvh.process_all_files``
    cluster_names   : list   ordered cluster labels  (default: CLUSTER_NAMES)
    cluster_members : dict   joint lists per cluster (default: CLUSTER_MEMBERS)
    phase_start, phase_end, phase_step : float
    verbose         : bool

    Returns
    -------
    out : dict
        {
          "data"           : (n_cycles, 7, 3, 2, n_phase) float64,
          "phase_grid"     : (n_phase,),
          "cluster_names"  : list[str],
          "axis_names"     : ["x","y","z"],
          "channel_names"  : ["position","velocity"],
          "cluster_members": dict,
          <per-cycle metadata lists aligned with data>,
        }
    """
    X          = make_phase_grid(phase_start, phase_end, phase_step)
    n_phase    = len(X)
    n_clusters = len(cluster_names)
    n_cycles   = _infer_n_cycles(cluster_data)

    if verbose:
        print(f"[cluster_7x3x2_bvh] cycles={n_cycles}, clusters={n_clusters}, "
              f"phase points={n_phase}")
        frame_times = cluster_data.get("cycle_frame_times", [])
        if len(frame_times) != n_cycles:
            print(
                f"[cluster_7x3x2_bvh] NOTE: cycle_frame_times length mismatch "
                f"({len(frame_times)} vs {n_cycles}); using evenly spaced phase."
            )

    data    = np.empty((n_cycles, n_clusters, 3, 2, n_phase), dtype=float)
    skipped = []

    for i in range(n_cycles):
        for ci, cname in enumerate(cluster_names):
            members  = cluster_members[cname]
            pos, vel = _cluster_posvel_realtime(cluster_data, cname, members, i)

            if pos is None:
                data[i, ci, :, :, :] = np.nan
                if i not in skipped:
                    skipped.append(i)
                continue

            C = _cycle_phase(cluster_data, i, len(pos))
            for a in range(3):
                data[i, ci, a, 0, :] = _resample_channel(C, pos[:, a], X)
                data[i, ci, a, 1, :] = _resample_channel(C, vel[:, a], X)

    out = {
        "data":            data,
        "phase_grid":      X,
        "cluster_names":   list(cluster_names),
        "axis_names":      list(AXIS_NAMES),
        "channel_names":   list(CHANNEL_NAMES),
        "cluster_members": {k: cluster_members[k] for k in cluster_names},
    }

    for key in METADATA_KEYS:
        if key in cluster_data:
            out[key] = list(cluster_data[key])

    if verbose:
        print(f"[cluster_7x3x2_bvh] data shape: {data.shape} "
              f"(n_cycles, cluster, xyz, [pos,vel], phase)")
        if skipped:
            print(f"[cluster_7x3x2_bvh] WARNING: {len(skipped)} cycle(s) had "
                  f"missing COM and were filled with NaN.")
    return out


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def _to_matlab_dict(out):
    """Make a savemat-friendly copy (no Python None values)."""
    mat = {}
    for k, v in out.items():
        if k == "cluster_members":
            mat[k] = {
                name: "center_of_mass" if members is None else list(members)
                for name, members in v.items()
            }
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
    """Save the builder output.

    File stem: ``cluster_7x3x2_bvh[_tag]_<today>``
    """
    if today is None:
        today = datetime.now().strftime("%d%b").lower()
    stem = "cluster_7x3x2_bvh"
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
        print(f"[cluster_7x3x2_bvh] Saved pkl: {p}")

    if save_mat:
        p = os.path.join(out_dir, stem + ".mat")
        savemat(p, _to_matlab_dict(out))
        paths["mat"] = p
        print(f"[cluster_7x3x2_bvh] Saved mat: {p}")

    if save_npy:
        p = os.path.join(out_dir, stem + ".npy")
        np.save(p, out["data"])
        paths["npy"] = p
        print(f"[cluster_7x3x2_bvh] Saved npy: {p}")

    return paths
