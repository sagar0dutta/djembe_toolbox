"""
BVH clustering data extraction functions.

Extracts position, velocity, acceleration, magnitudes, and centre-of-mass data
for all BVH joints from root-centred world-position CSV files,
organised by dance mode and virtual cycle.

CSV filename convention (bvh_to_csv_centered/):
    {file_name}_T_worldpos_rc_both_frame_smooth0.4.csv
    e.g. BKO_E1_D1_01_Suku_T_worldpos_rc_both_frame_smooth0.4.csv

Coordinate system (BVH root-centred, frontal-aligned):
    X = mediolateral (left-right)   -- root-centred (Hips.X = 0 at t=0)
    Y = vertical (superior)         -- not centred  (Hips.Y ≈ 89 cm)
    Z = anterior (forward-backward) -- root-centred (Hips.Z = 0 at t=0)
"""

import os
import pickle
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Joint catalogue
# --------------------------------------------------------------------------- #

BVH_JOINTS = [
    "Hips",
    "LeftHip",   "LeftKnee",  "LeftAnkle",  "LeftToe",  "LeftToeEnd",
    "RightHip",  "RightKnee", "RightAnkle", "RightToe", "RightToeEnd",
    "Chest",     "Chest2",    "Chest3",     "Chest4",
    "LeftCollar",  "LeftShoulder",  "LeftElbow",  "LeftWrist",  "LeftWristEnd",
    "RightCollar", "RightShoulder", "RightElbow", "RightWrist", "RightWristEnd",
    "Neck", "Head", "HeadEnd",
]

# Joints used to approximate the centre of mass (torso + proximal limbs)
COM_JOINTS = [
    "Hips",
    "LeftHip",  "RightHip",
    "Chest",    "Chest2",   "Chest3", "Chest4",
    "LeftShoulder", "RightShoulder",
    "Neck",     "Head",
]

BVH_CSV_SUFFIX = "_T_worldpos_rc_both_frame_smooth0.4.csv"
BVH_DIR_DEFAULT = "bvh_to_csv_centered"

DANCER_MAP = {
    ("E1", "D1"): "dancer_1",
    ("E1", "D5"): "dancer_1",
    ("E1", "D2"): "dancer_2",
    ("E2", "D3"): "dancer_3",
    ("E2", "D4"): "dancer_3",
    ("E3", "D5"): "dancer_4",
    ("E3", "D6"): "dancer_4",
}


# --------------------------------------------------------------------------- #
# Data-structure initialiser
# --------------------------------------------------------------------------- #

def initialize_cluster_data(joint_names):
    """
    Create an empty cluster_data dict with the standard structure.

    Parameters
    ----------
    joint_names : list[str]
        BVH joint names to include as body_parts.

    Returns
    -------
    cluster_data : dict
    """
    cluster_data = {
        # --- metadata ---
        "file_name":         [],
        "dmode_name":        [],
        "dmode_seg_idx":     [],
        "dmode_start":       [],
        "dmode_end":         [],
        "cycle_idx":         [],
        "cycle_start":       [],
        "cycle_end":         [],
        "cycle_frame_times": [],
        "location":          [],
        "ensemble":          [],
        "day":               [],
        "rec_no":            [],
        "piece":             [],
        "dancer_id":         [],
        # --- onset metadata (arrays of times in seconds within each cycle) ---
        "hand_clap_onsets":  [],
        "both_feet_onsets":  [],
        "left_foot_onsets":  [],
        "right_foot_onsets": [],
        # --- per-joint trajectories ---
        "body_parts": {
            jnt: {
                "position":               [],
                "velocity":               [],
                "acceleration":           [],
                "velocity_magnitude":     [],
                "acceleration_magnitude": [],
                "distance_from_com":      [],
            }
            for jnt in joint_names
        },
        # --- centre of mass ---
        "center_of_mass": {
            "position":     [],
            "velocity":     [],
            "acceleration": [],
        },
    }
    print(f"[DEBUG] Initialized cluster_data with {len(joint_names)} joints")
    return cluster_data


# --------------------------------------------------------------------------- #
# Kinematic helpers
# --------------------------------------------------------------------------- #

def compute_magnitude(vector_array):
    """L2 norm per frame.  (N, 3) -> (N,)   or   (N,) -> (N,)."""
    if vector_array.ndim == 1:
        return np.abs(vector_array)
    return np.linalg.norm(vector_array, axis=1)


def compute_distance_from_com(body_part_pos, com_pos):
    """Euclidean distance from COM per frame.  (N, 3), (N, 3) -> (N,)."""
    return np.linalg.norm(body_part_pos - com_pos, axis=1)


def compute_kinematics(positions, frame_times):
    """
    Compute velocity and acceleration using central differences (np.gradient).

    Parameters
    ----------
    positions   : np.ndarray  (N, 3)
    frame_times : np.ndarray  (N,)   seconds

    Returns
    -------
    velocity     : np.ndarray  (N, 3)
    acceleration : np.ndarray  (N, 3)
    """
    velocity     = np.gradient(positions,  frame_times, axis=0)
    acceleration = np.gradient(velocity,   frame_times, axis=0)
    return velocity, acceleration


# --------------------------------------------------------------------------- #
# BVH CSV loading
# --------------------------------------------------------------------------- #

def load_bvh_csv(file_name, bvh_dir=BVH_DIR_DEFAULT):
    """
    Load a root-centred BVH world-position CSV.

    Parameters
    ----------
    file_name : str   e.g. "BKO_E1_D1_01_Suku"
    bvh_dir   : str   directory containing the CSV files

    Returns
    -------
    frame_times : np.ndarray  (N,)        seconds
    joint_pos   : dict  joint_name -> np.ndarray (N, 3)
    """
    csv_path = os.path.join(bvh_dir, file_name + BVH_CSV_SUFFIX)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BVH CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    frame_times = df["Time"].values.astype(np.float64)

    joint_pos = {
        jnt: df[[f"{jnt}.X", f"{jnt}.Y", f"{jnt}.Z"]].values.astype(np.float64)
        for jnt in BVH_JOINTS
    }
    return frame_times, joint_pos


def compute_com_trajectory(joint_pos, com_joints=COM_JOINTS):
    """
    Approximate the centre of mass as the mean of the specified joints.

    Parameters
    ----------
    joint_pos  : dict  joint_name -> (N, 3)
    com_joints : list  subset of joint names to average

    Returns
    -------
    com_pos : np.ndarray  (N, 3)
    """
    available = [j for j in com_joints if j in joint_pos]
    return np.stack([joint_pos[j] for j in available], axis=0).mean(axis=0)


def derive_piece_list(bvh_dir=BVH_DIR_DEFAULT):
    """
    Derive piece_list from all CSV files present in bvh_dir.

    Returns
    -------
    piece_list : list[str]   e.g. ["BKO_E1_D1_01_Suku", ...]  sorted
    """
    files = [
        f.replace(BVH_CSV_SUFFIX, "")
        for f in os.listdir(bvh_dir)
        if f.endswith(BVH_CSV_SUFFIX)
    ]
    return sorted(files)


# --------------------------------------------------------------------------- #
# Per-file processor
# --------------------------------------------------------------------------- #

def process_single_file(
    file_name,
    modes,
    bvh_dir=BVH_DIR_DEFAULT,
    base_path_cycles="data/virtual_cycles",
    debug=True,
):
    """
    Process one recording and return a cycle-segmented cluster_data dict.

    Parameters
    ----------
    file_name         : str    e.g. "BKO_E1_D1_01_Suku"
    modes             : list   dance modes, e.g. ["group","individual","audience"]
    bvh_dir           : str    path to bvh_to_csv_centered/
    base_path_cycles  : str    path to data/virtual_cycles/
    debug             : bool

    Returns
    -------
    file_data : dict or None (if critical files are missing)
    """
    if debug:
        print(f"\n[DEBUG] Processing: {file_name}")

    # Parse metadata from filename  (BKO_E1_D1_01_Suku)
    parts = file_name.split("_")
    if len(parts) < 5:
        print(f"[WARNING] Unexpected filename format: {file_name} — skipping")
        return None

    location     = parts[0]            # BKO
    ensemble     = parts[1]            # E1
    day          = parts[2]            # D1
    recording_no = parts[3]            # 01
    piece        = "_".join(parts[4:]) # Suku
    dancer_id    = DANCER_MAP.get((ensemble, day), "unknown")

    # ---------------------------------------------------------------------- #
    # Load BVH CSV
    # ---------------------------------------------------------------------- #
    try:
        frame_times, joint_pos = load_bvh_csv(file_name, bvh_dir)
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc} — skipping")
        return None

    n_frames = len(frame_times)
    if debug:
        print(f"[DEBUG] Loaded {n_frames} frames, duration={frame_times[-1]:.2f}s")

    # ---------------------------------------------------------------------- #
    # Pre-compute kinematics for every joint
    # ---------------------------------------------------------------------- #
    joint_vel = {}
    joint_acc = {}
    for jnt in BVH_JOINTS:
        v, a = compute_kinematics(joint_pos[jnt], frame_times)
        joint_vel[jnt] = v
        joint_acc[jnt] = a

    com_pos              = compute_com_trajectory(joint_pos)
    com_vel, com_acc     = compute_kinematics(com_pos, frame_times)

    # ---------------------------------------------------------------------- #
    # Load onset files
    # ---------------------------------------------------------------------- #
    def _load_onsets(path, col):
        """Load a single-column onset CSV; return empty array if file missing."""
        try:
            return pd.read_csv(path)[col].dropna().values.astype(float)
        except Exception:
            if debug:
                print(f"[DEBUG] Onset file not found or unreadable: {path}")
            return np.array([], dtype=float)

    hand_clap_onsets = _load_onsets(
        os.path.join("data", "hand_clap_onsets_05jun2026",
                     f"{file_name}_hand_clap_onsets.csv"),
        "contact_times",
    )
    foot_onset_dir = os.path.join(
        "data", "dance_onsets_v4_0.007_foot_jun3", f"{file_name}_T", "onset_info"
    )
    both_feet_onsets  = _load_onsets(
        os.path.join(foot_onset_dir, f"{file_name}_T_both_feet_onsets.csv"),  "time_sec"
    )
    left_foot_onsets  = _load_onsets(
        os.path.join(foot_onset_dir, f"{file_name}_T_left_foot_onsets.csv"),  "time_sec"
    )
    right_foot_onsets = _load_onsets(
        os.path.join(foot_onset_dir, f"{file_name}_T_right_foot_onsets.csv"), "time_sec"
    )

    if debug:
        print(f"[DEBUG] Onsets loaded — hand_clap: {len(hand_clap_onsets)}, "
              f"both_feet: {len(both_feet_onsets)}, "
              f"left_foot: {len(left_foot_onsets)}, "
              f"right_foot: {len(right_foot_onsets)}")

    # ---------------------------------------------------------------------- #
    # Load virtual cycles
    # ---------------------------------------------------------------------- #
    cycles_csv = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    if not os.path.exists(cycles_csv):
        print(f"[WARNING] Cycles CSV not found: {cycles_csv} — skipping")
        return None

    try:
        cyc_df     = pd.read_csv(cycles_csv)
        all_onsets = cyc_df["Virtual Onset"].values
        if debug:
            print(f"[DEBUG] Loaded {len(all_onsets)} virtual onsets")
    except Exception as exc:
        print(f"[ERROR] Failed to load cycles CSV for {file_name}: {exc}")
        return None

    # ---------------------------------------------------------------------- #
    # Initialise output
    # ---------------------------------------------------------------------- #
    file_data = initialize_cluster_data(BVH_JOINTS)

    # ---------------------------------------------------------------------- #
    # Loop over dance modes → segments → cycles
    # ---------------------------------------------------------------------- #
    for dance_mode in modes:
        dmode_path = f"data/dance_modes_ts/{file_name}_{dance_mode}.pkl"
        if not os.path.exists(dmode_path):
            if debug:
                print(f"[DEBUG] {dance_mode} not found for {file_name} — skipping")
            continue

        try:
            with open(dmode_path, "rb") as fh:
                dmode_ts = pickle.load(fh)
        except Exception as exc:
            print(f"[ERROR] Failed to load dance-mode file {dmode_path}: {exc}")
            continue

        if debug:
            print(f"[DEBUG] {dance_mode}: {len(dmode_ts)} segment(s)")

        for dmode_idx, (dmode_start, dmode_end) in enumerate(dmode_ts):
            mode_mask   = (all_onsets >= dmode_start) & (all_onsets <= dmode_end)
            mode_onsets = all_onsets[mode_mask]

            if len(mode_onsets) < 2:
                if debug:
                    print(f"[DEBUG]   Segment {dmode_idx+1}: only "
                          f"{len(mode_onsets)} onset(s) — skipping")
                continue

            cycle_boundaries = [
                (round(mode_onsets[i], 3), round(mode_onsets[i + 1], 3))
                for i in range(len(mode_onsets) - 1)
            ]
            if debug:
                print(f"[DEBUG]   Segment {dmode_idx+1}: "
                      f"{len(cycle_boundaries)} cycle(s)")

            for c_idx, (c_start, c_end) in enumerate(cycle_boundaries):
                win_mask = (frame_times >= c_start) & (frame_times <= c_end)
                if not np.any(win_mask):
                    continue

                cycle_frame_times = frame_times[win_mask]

                # COM window
                com_pos_win = com_pos[win_mask]
                com_vel_win = com_vel[win_mask]
                com_acc_win = com_acc[win_mask]

                file_data["center_of_mass"]["position"].append(com_pos_win)
                file_data["center_of_mass"]["velocity"].append(com_vel_win)
                file_data["center_of_mass"]["acceleration"].append(com_acc_win)

                # Per-joint window
                for jnt in BVH_JOINTS:
                    pos_win = joint_pos[jnt][win_mask]
                    vel_win = joint_vel[jnt][win_mask]
                    acc_win = joint_acc[jnt][win_mask]

                    file_data["body_parts"][jnt]["position"].append(pos_win)
                    file_data["body_parts"][jnt]["velocity"].append(vel_win)
                    file_data["body_parts"][jnt]["acceleration"].append(acc_win)
                    file_data["body_parts"][jnt]["velocity_magnitude"].append(
                        compute_magnitude(vel_win)
                    )
                    file_data["body_parts"][jnt]["acceleration_magnitude"].append(
                        compute_magnitude(acc_win)
                    )
                    file_data["body_parts"][jnt]["distance_from_com"].append(
                        compute_distance_from_com(pos_win, com_pos_win)
                    )

                # Onsets windowed to this cycle
                file_data["hand_clap_onsets"].append(
                    hand_clap_onsets[(hand_clap_onsets >= c_start) & (hand_clap_onsets <= c_end)]
                )
                file_data["both_feet_onsets"].append(
                    both_feet_onsets[(both_feet_onsets >= c_start) & (both_feet_onsets <= c_end)]
                )
                file_data["left_foot_onsets"].append(
                    left_foot_onsets[(left_foot_onsets >= c_start) & (left_foot_onsets <= c_end)]
                )
                file_data["right_foot_onsets"].append(
                    right_foot_onsets[(right_foot_onsets >= c_start) & (right_foot_onsets <= c_end)]
                )

                # Metadata
                file_data["file_name"].append(file_name)
                file_data["dmode_name"].append(dance_mode)
                file_data["dmode_seg_idx"].append(dmode_idx + 1)
                file_data["dmode_start"].append(dmode_start)
                file_data["dmode_end"].append(dmode_end)
                file_data["cycle_idx"].append(c_idx + 1)
                file_data["cycle_start"].append(c_start)
                file_data["cycle_end"].append(c_end)
                file_data["cycle_frame_times"].append(cycle_frame_times)
                file_data["location"].append(location)
                file_data["ensemble"].append(ensemble)
                file_data["day"].append(day)
                file_data["rec_no"].append(recording_no)
                file_data["piece"].append(piece)
                file_data["dancer_id"].append(dancer_id)

    if debug:
        print(f"[DEBUG] Done {file_name}: {len(file_data['file_name'])} cycles extracted")
    return file_data


# --------------------------------------------------------------------------- #
# All-files processor
# --------------------------------------------------------------------------- #

def process_all_files(
    piece_list,
    modes=("group", "individual", "audience"),
    bvh_dir=BVH_DIR_DEFAULT,
    base_path_cycles="data/virtual_cycles",
    debug=True,
):
    """
    Process every file in piece_list and merge into one cluster_data dict.

    Parameters
    ----------
    piece_list        : list[str]
    modes             : list[str]   dance modes to include
    bvh_dir           : str         path to bvh_to_csv_centered/
    base_path_cycles  : str
    debug             : bool

    Returns
    -------
    cluster_data : dict
    """
    modes = list(modes)
    print(f"[INFO] Processing {len(piece_list)} file(s), modes={modes}")

    cluster_data    = initialize_cluster_data(BVH_JOINTS)
    successful, failed = 0, []

    for idx, file_name in enumerate(piece_list):
        print(f"\n[INFO] File {idx + 1}/{len(piece_list)}: {file_name}")

        file_data = process_single_file(
            file_name=file_name,
            modes=modes,
            bvh_dir=bvh_dir,
            base_path_cycles=base_path_cycles,
            debug=debug,
        )

        if file_data is None:
            failed.append(file_name)
            continue

        # Merge metadata lists
        for key in [
            "file_name", "dmode_name", "dmode_seg_idx", "dmode_start", "dmode_end",
            "cycle_idx", "cycle_start", "cycle_end", "cycle_frame_times",
            "location", "ensemble", "day", "rec_no", "piece", "dancer_id",
            "hand_clap_onsets", "both_feet_onsets", "left_foot_onsets", "right_foot_onsets",
        ]:
            cluster_data[key].extend(file_data[key])

        # Merge per-joint trajectories
        for jnt in BVH_JOINTS:
            for key in [
                "position", "velocity", "acceleration",
                "velocity_magnitude", "acceleration_magnitude", "distance_from_com",
            ]:
                cluster_data["body_parts"][jnt][key].extend(
                    file_data["body_parts"][jnt][key]
                )

        # Merge COM
        for key in ["position", "velocity", "acceleration"]:
            cluster_data["center_of_mass"][key].extend(
                file_data["center_of_mass"][key]
            )

        successful += 1

    print(f"\n[INFO] Done: {successful}/{len(piece_list)} file(s) OK")
    if failed:
        print(f"[WARNING] Failed: {failed}")
    print(f"[INFO] Total cycles extracted: {len(cluster_data['file_name'])}")
    return cluster_data
