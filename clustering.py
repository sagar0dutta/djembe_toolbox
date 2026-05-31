"""
Clustering data extraction functions.

Extracts position, velocity, acceleration, magnitudes, and center of mass data
for all body parts from motion capture data, organized by dance mode and cycle.
"""

import os
import pickle
import numpy as np
import pandas as pd


def initialize_cluster_data(body_part_names):
    """
    Initialize the cluster_data dictionary structure.
    
    Parameters:
    -----------
    body_part_names : list
        List of body part names from motion_data["segments"].keys()
    
    Returns:
    --------
    cluster_data : dict
        Initialized dictionary with metadata and nested body parts structure
    """
    cluster_data = {
        # === METADATA ===
        "file_name": [],           # string
        "dmode_name": [],          # string
        "dmode_seg_idx": [],       # int
        "dmode_start": [],         # float
        "dmode_end": [],           # float
        "cycle_idx": [],           # int
        "cycle_start": [],         # float
        "cycle_end": [],           # float
        "cycle_frame_times": [],  # np.ndarray of shape (n_frames,) absolute times in seconds
        "location": [],            # string
        "ensemble": [],            # string
        "day": [],                 # string
        "rec_no": [],              # string
        "piece": [],               # string
        
        # === BODY PARTS DATA ===
        "body_parts": {}
    }
    
    # Initialize structure for each body part
    for part_name in body_part_names:
        cluster_data["body_parts"][part_name] = {
            "position": [],              # List of arrays: (n_frames, 3) for x,y,z
            "velocity": [],             # List of arrays: (n_frames, 3) for vx,vy,vz
            "acceleration": [],         # List of arrays: (n_frames, 3) for ax,ay,az
            "velocity_magnitude": [],   # List of arrays: (n_frames,) scalar
            "acceleration_magnitude": [], # List of arrays: (n_frames,) scalar
            "distance_from_com": []     # List of arrays: (n_frames,) scalar
        }
    
    # === CENTER OF MASS ===
    cluster_data["center_of_mass"] = {
        "position": [],              # List of arrays: (n_frames, 3) for x,y,z
        "velocity": [],             # List of arrays: (n_frames, 3) for vx,vy,vz
        "acceleration": [],         # List of arrays: (n_frames, 3) for ax,ay,az
    }
    
    print(f"[DEBUG] Initialized cluster_data with {len(body_part_names)} body parts: {body_part_names}")
    return cluster_data


def compute_magnitude(vector_array):
    """
    Compute magnitude (L2 norm) for each frame in a vector array.
    
    Parameters:
    -----------
    vector_array : np.ndarray
        Array of shape (n_frames, 3) or (n_frames,)
    
    Returns:
    --------
    magnitude : np.ndarray
        Array of shape (n_frames,) with magnitude for each frame
    """
    if vector_array.ndim == 1:
        return np.abs(vector_array)
    return np.linalg.norm(vector_array, axis=1)


def compute_distance_from_com(body_part_pos, com_pos):
    """
    Compute distance from center of mass for each frame.
    
    Parameters:
    -----------
    body_part_pos : np.ndarray
        Body part position array of shape (n_frames, 3)
    com_pos : np.ndarray
        Center of mass position array of shape (n_frames, 3)
    
    Returns:
    --------
    distance : np.ndarray
        Array of shape (n_frames,) with distance for each frame
    """
    return np.linalg.norm(body_part_pos - com_pos, axis=1)


def process_single_file(
    file_name,
    modes,
    mocap_fps=240,
    motion_data_dir="data/motion_data_pkl",
    base_path_cycles="data/virtual_cycles",
    debug=True
):
    """
    Process a single file and extract clustering data for all dance modes.
    
    Parameters:
    -----------
    file_name : str
        Name of the file to process
    modes : list
        List of dance modes to process ["group", "individual", "audience"]
    mocap_fps : float
        Motion capture frame rate (default: 240)
    motion_data_dir : str
        Directory containing motion data pickle files
    base_path_cycles : str
        Base path for cycle CSV files
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    file_data : dict
        Dictionary with same structure as cluster_data, containing data for this file only
    """
    if debug:
        print(f"\n[DEBUG] Processing file: {file_name}")
    
    # Parse file name
    parts = file_name.split("_")
    if len(parts) < 5:
        print(f"[WARNING] Unexpected file name format: {file_name}, skipping")
        return None
    
    location = parts[0]      # BKO
    ensemble = parts[1]       # E1
    day = parts[2]           # D1
    recording_no = parts[3]  # 01
    piece = "_".join(parts[4:])  # Suku (handles multi-word pieces)
    
    if debug:
        print(f"[DEBUG] Parsed: location={location}, ensemble={ensemble}, day={day}, rec_no={recording_no}, piece={piece}")
    
    # Build file paths
    cycles_csv = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    mpkl_path = os.path.join(motion_data_dir, f"{file_name}_T.pkl")
    
    
    ############################### TEMPORARY ######################################
    bvh_path = os.path.join("extracted_mocap_csv", f"{file_name}_T_worldpos.csv")
    
    bvh_df = pd.read_csv(bvh_path)
    bvh_frame_times = bvh_df["Time"].values  # in seconds
    
    bvh_hip_joint = {
        "left_hip_x_pos": np.array(bvh_df["LeftHip.X"].values),
        "left_hip_y_pos": np.array(bvh_df["LeftHip.Y"].values),
        "left_hip_z_pos": np.array(bvh_df["LeftHip.Z"].values),
        "right_hip_x_pos": np.array(bvh_df["RightHip.X"].values),
        "right_hip_y_pos": np.array(bvh_df["RightHip.Y"].values),
        "right_hip_z_pos": np.array(bvh_df["RightHip.Z"].values),
        
        }
    
    hip_part_names = list(bvh_hip_joint.keys())
    ############################### TEMPORARY ######################################
    
    # Check if files exist
    if not os.path.exists(cycles_csv):
        print(f"[WARNING] Cycles CSV not found: {cycles_csv}, skipping")
        return None
    
    if not os.path.exists(mpkl_path):
        print(f"[WARNING] Motion data not found: {mpkl_path}, skipping")
        return None
    
    # Load motion data
    try:
        with open(mpkl_path, 'rb') as f:
            motion_data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load motion data for {file_name}: {e}")
        return None
    
    # Get body part names from motion_data
    if "segments" not in motion_data:
        print(f"[ERROR] No 'segments' key in motion_data for {file_name}")
        return None
    
    body_part_names = list(motion_data["segments"].keys())
    # --------------------------DEBUG PRINT----------------------------
    if debug:
        print(f"[DEBUG] Found {len(body_part_names)} body parts: {body_part_names}")
    
    # Check for center of mass
    has_com = "center_of_mass" in motion_data
    # --------------------------DEBUG PRINT----------------------------
    if debug:
        print(f"[DEBUG] Center of mass available: {has_com}")
        if has_com:
            print(f"[DEBUG] COM keys: {list(motion_data['center_of_mass'].keys())}")
    # -------------------------------------------------------------------        
    
    ######################################## Initialize data structure for this file
    
    combined_body_part_names = body_part_names + hip_part_names
    file_data = initialize_cluster_data(combined_body_part_names)
    
    # Load cycles
    try:
        cyc_df = pd.read_csv(cycles_csv)
        onsets = cyc_df["Virtual Onset"].values  # in seconds
        if debug:
            print(f"[DEBUG] Loaded {len(onsets)} cycle onsets")
    except Exception as e:
        print(f"[ERROR] Failed to load cycles CSV for {file_name}: {e}")
        return None
    
    # Get time arrays for all body parts (they should all have same length)
    first_part = body_part_names[0]
    n_frames = len(motion_data["segments"][first_part]["position"])
    times = np.arange(n_frames) / mocap_fps
    
    
    ################################
    # print("n_frames mvnx:", n_frames)
    # print("n_frames bvh:", len(bvh_frame_times))
    #################################

    if debug:
        print(f"[DEBUG] Total frames: {n_frames}, duration: {times[-1]:.2f}s")
    
    # Process each dance mode
    for dance_mode in modes:
        dance_mode_path = f"data/dance_modes_ts/{file_name}_{dance_mode}.pkl"
        
        if not os.path.exists(dance_mode_path):
            if debug:
                print(f"[DEBUG] {file_name} {dance_mode} does not exist, skipping")
            continue
        
        # Load dance mode time segments
        try:
            with open(dance_mode_path, "rb") as f:
                dmode_ts = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load dance mode segments for {file_name} {dance_mode}: {e}")
            continue
        
        if debug:
            print(f"[DEBUG] Processing {dance_mode} mode with {len(dmode_ts)} segments")
        
        # Process each dance mode segment
        for dmode_idx, dmode in enumerate(dmode_ts):
            dmode_start, dmode_end = dmode
            
            if debug:
                print(f"[DEBUG]  Segment {dmode_idx+1}: {dmode_start:.2f}s - {dmode_end:.2f}s")
            
            # Filter onsets to get only those within the dance mode time segment
            mode_mask = (onsets >= dmode_start) & (onsets <= dmode_end)
            mode_onsets = onsets[mode_mask]
            
            if len(mode_onsets) < 2:
                if debug:
                    print(f"[DEBUG]    Only {len(mode_onsets)} onsets in segment, skipping")
                continue
            
            # Create list of tuples with start and end times of cycles within the dance mode segment
            cycle_times = [(round(mode_onsets[i], 3), round(mode_onsets[i+1], 3)) 
                          for i in range(len(mode_onsets)-1)]
            
            if debug:
                print(f"[DEBUG]    Found {len(cycle_times)} cycles in segment")
            
            # Process each cycle
            for c_idx, (c_start, c_end) in enumerate(cycle_times):
                if debug and c_idx % 10 == 0:
                    print(f"[DEBUG]      Processing cycle {c_idx+1}/{len(cycle_times)}: {c_start:.2f}s - {c_end:.2f}s")
                
                # Window mask for this cycle
                win_mask = (times >= c_start) & (times <= c_end)        # in mocap time
                n_win_frames = np.sum(win_mask)
                
                # Time vector for frames inside this cycle
                cycle_times_frames = times[win_mask]   # absolute time in seconds 13 Dec 2025
                # print(cycle_times_frames)
                
                if n_win_frames == 0:
                    if debug:
                        print(f"[DEBUG]        No frames in window, skipping")
                    continue
                
                # === PROCESS CENTER OF MASS ===
                if has_com:
                    com_pos_win = motion_data["center_of_mass"]["position"][win_mask]  # (n_frames, 3)
                    com_vel_win = motion_data["center_of_mass"]["velocity"][win_mask]  # (n_frames, 3)
                    com_acc_win = motion_data["center_of_mass"]["acceleration"][win_mask]  # (n_frames, 3)
                    
                    file_data["center_of_mass"]["position"].append(com_pos_win)
                    file_data["center_of_mass"]["velocity"].append(com_vel_win)
                    file_data["center_of_mass"]["acceleration"].append(com_acc_win)

                else:
                    # If no COM, append None or empty arrays
                    file_data["center_of_mass"]["position"].append(None)
                    file_data["center_of_mass"]["velocity"].append(None)
                    file_data["center_of_mass"]["acceleration"].append(None)
                
                # === PROCESS EACH BODY PART ===
                for part_name in body_part_names:
                    # Get position, velocity, acceleration for this cycle window
                    part_pos_win = motion_data["segments"][part_name]["position"][win_mask]  # (n_frames, 3)
                    part_vel_win = motion_data["segments"][part_name]["velocity"][win_mask]  # (n_frames, 3)
                    part_acc_win = motion_data["segments"][part_name]["acceleration"][win_mask]  # (n_frames, 3)
                    
                    # Compute magnitudes
                    part_vel_mag = compute_magnitude(part_vel_win)
                    part_acc_mag = compute_magnitude(part_acc_win)
                    
                    # Compute distance from center of mass
                    if has_com:
                        part_dist_from_com = compute_distance_from_com(part_pos_win, com_pos_win)
                    else:
                        part_dist_from_com = np.full(n_win_frames, np.nan)
                    
                    # Store data
                    file_data["body_parts"][part_name]["position"].append(part_pos_win)
                    file_data["body_parts"][part_name]["velocity"].append(part_vel_win)
                    file_data["body_parts"][part_name]["acceleration"].append(part_acc_win)
                    file_data["body_parts"][part_name]["velocity_magnitude"].append(part_vel_mag)
                    file_data["body_parts"][part_name]["acceleration_magnitude"].append(part_acc_mag)
                    file_data["body_parts"][part_name]["distance_from_com"].append(part_dist_from_com)
                
                for hip_part in hip_part_names:
                    hip_part_win = bvh_hip_joint[hip_part][win_mask]  # (n_frames,)
                    file_data["body_parts"][hip_part]["position"].append(hip_part_win)
                
                
                
                # === STORE METADATA ===
                file_data["file_name"].append(file_name)
                file_data["dmode_name"].append(dance_mode)
                file_data["dmode_seg_idx"].append(dmode_idx + 1)
                file_data["dmode_start"].append(dmode_start)
                file_data["dmode_end"].append(dmode_end)
                file_data["cycle_idx"].append(c_idx + 1)
                file_data["cycle_start"].append(c_start)
                file_data["cycle_end"].append(c_end)
                file_data["cycle_frame_times"].append(cycle_times_frames)
                file_data["location"].append(location)
                file_data["ensemble"].append(ensemble)
                file_data["day"].append(day)
                file_data["rec_no"].append(recording_no)
                file_data["piece"].append(piece)
    
    if debug:
        n_cycles = len(file_data["file_name"])
        print(f"[DEBUG] Completed {file_name}: {n_cycles} cycles extracted")
    
    return file_data


def process_all_files(
    piece_list,
    modes=["group", "individual", "audience"],
    mocap_fps=240,
    motion_data_dir="data/motion_data_pkl",
    base_path_cycles="data/virtual_cycles",
    debug=True
):
    """
    Process all files in piece_list and combine into a single cluster_data dictionary.
    
    Parameters:
    -----------
    piece_list : list
        List of file names to process
    modes : list
        List of dance modes to process
    mocap_fps : float
        Motion capture frame rate
    motion_data_dir : str
        Directory containing motion data pickle files
    base_path_cycles : str
        Base path for cycle CSV files
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    cluster_data : dict
        Combined cluster_data dictionary with all files
    """
    print(f"[INFO] Starting processing of {len(piece_list)} files")
    print(f"[INFO] Dance modes: {modes}")
    
    # Initialize with body parts from first file
    # We'll need to load one file first to get body part names
    first_file = piece_list[0]
    mpkl_path = os.path.join(motion_data_dir, f"{first_file}_T.pkl")
    
    if not os.path.exists(mpkl_path):
        print(f"[ERROR] Cannot find first file to initialize: {mpkl_path}")
        return None
    
    with open(mpkl_path, 'rb') as f:
        first_motion_data = pickle.load(f)
    
    body_part_names = list(first_motion_data["segments"].keys())
    cluster_data = initialize_cluster_data(body_part_names)
    
    # Process each file
    successful_files = 0
    failed_files = []
    
    for file_idx, file_name in enumerate(piece_list):
        print(f"\n[INFO] Processing file {file_idx+1}/{len(piece_list)}: {file_name}")
        
        file_data = process_single_file(
            file_name=file_name,
            modes=modes,
            mocap_fps=mocap_fps,
            motion_data_dir=motion_data_dir,
            base_path_cycles=base_path_cycles,
            debug=debug
        )
        
        if file_data is None:
            failed_files.append(file_name)
            continue
        
        # Merge file_data into cluster_data
        # Metadata
        for key in ["file_name", "dmode_name", "dmode_seg_idx", "dmode_start", "dmode_end",
                   "cycle_idx", "cycle_start", "cycle_end", "location", "ensemble", 
                   "day", "rec_no", "piece"]:
            cluster_data[key].extend(file_data[key])
        
        # Body parts
        for part_name in body_part_names:
            for data_type in ["position", "velocity", "acceleration", 
                            "velocity_magnitude", "acceleration_magnitude", "distance_from_com"]:
                cluster_data["body_parts"][part_name][data_type].extend(
                    file_data["body_parts"][part_name][data_type]
                )
        
        # Center of mass
        for data_type in ["position", "velocity", "acceleration"]:
            cluster_data["center_of_mass"][data_type].extend(
                file_data["center_of_mass"][data_type]
            )
        
        successful_files += 1
    
    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Successful files: {successful_files}/{len(piece_list)}")
    if failed_files:
        print(f"[WARNING] Failed files: {failed_files}")
    
    total_cycles = len(cluster_data["file_name"])
    print(f"[INFO] Total cycles extracted: {total_cycles}")
    
    return cluster_data

