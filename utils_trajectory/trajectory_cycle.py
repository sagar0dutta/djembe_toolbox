import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.lines import Line2D
from scipy.interpolate import interp1d


def plot_cycles_trajectories(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,
    time_segments: list = None,  # List of (start, end) tuples
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    figsize: tuple = (12, 6),
    dpi: int = 200,
    show_trajectories: bool = True,  # Control trajectory lines
    show_vlines: bool = True,        # Control vertical lines
    show_gray_plots: bool = True     # Control gray trajectory plots
):
    """
    Plot all foot trajectories in a single plot with grand average.
    Shows beat and subdivision lines with colors.
    X-axis shows beats 1-4 directly.
    """
    # Use default window if no segments provided
    if time_segments is None:
        time_segments = [(0, 10)]

    # build file paths
    cycles_csv = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    # logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T", "onset_info")
    
    # left_onsets_csv  = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_onsets.csv")
    # right_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_onsets.csv")
    # left_zpos_csv    = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_zpos.csv")
    # right_zpos_csv   = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_zpos.csv")
    
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T")
    onset_data_path = os.path.join(logs_onset_dir, f"{file_name}_T_feet_onset.pkl")
    
    with open(onset_data_path, 'rb') as f:
        onset_data = pickle.load(f)

    
    

    # load data
    # Lz = pd.read_csv(left_zpos_csv)["zpos"].values
    # Rz = pd.read_csv(right_zpos_csv)["zpos"].values
    
    Lz = onset_data["left_foot_zPos"]
    Rz = onset_data["right_foot_zPos"]
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate

    # interpolation functions
    L_interp = interp1d(times, Lz, bounds_error=False, fill_value="extrapolate")
    R_interp = interp1d(times, Rz, bounds_error=False, fill_value="extrapolate")

    # Get overall time range for color mapping
    total_start = min(seg[0] for seg in time_segments)
    total_end = max(seg[1] for seg in time_segments)
    t_range = total_end - total_start

    # Calculate average cycle duration from all segments
    all_onsets = []
    for seg_start, seg_end in time_segments:
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not cyc_df.empty:
            all_onsets.extend(cyc_df["Virtual Onset"].values[:-1])
    
    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")
    
    all_onsets = np.sort(all_onsets)
    
    # Calculate average cycle duration
    durations = np.diff(sorted(all_onsets))
    avg_cycle = durations.mean()

    # Calculate beat and subdivision lengths
    # beat_len = avg_cycle / n_beats_per_cycle
    # subdiv_len = beat_len / n_subdiv_per_beat

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap('cool')

    # Define subdivision color mapping
    def get_subdiv_color(subdiv):
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        group = ((subdiv - 1) % 3) + 1
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'

    # Process each time segment
    all_L_trajectories = []
    all_R_trajectories = []
    all_times = []

    for seg_start, seg_end in time_segments:
        # trim to window
        win_mask = (times >= seg_start) & (times <= seg_end)
        t_win = times[win_mask]
        L_win = Lz[win_mask]
        R_win = Rz[win_mask]

        # cycles (downbeats)
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        onsets = cyc_df["Virtual Onset"].values[:-1]

        # foot onsets
        # left_df  = pd.read_csv(left_onsets_csv) #---------------------------------------
        # right_df = pd.read_csv(right_onsets_csv)
        # left_times  = left_df[(left_df["time_sec"]>=seg_start)&(left_df["time_sec"]<=seg_end)]["time_sec"].values
        # right_times = right_df[(right_df["time_sec"]>=seg_start)&(right_df["time_sec"]<=seg_end)]["time_sec"].values
        
        left_df = onset_data["left_foot_zOnsets"]
        right_df = onset_data["right_foot_zOnsets"]
        
        mask_left = (left_df['time_sec'] >= seg_start) & (left_df['time_sec'] <= seg_end)
        mask_right = (right_df['time_sec'] >= seg_start) & (right_df['time_sec'] <= seg_end)

        left_times = left_df['time_sec'][mask_left]
        right_times = right_df['time_sec'][mask_right]
        

        # Plot trajectories for each cycle
        # for c in onsets:
        for i, c in enumerate(onsets):
            # Convert time to beat position (1-4)
            # cycle_start = c
            # cycle_end = c + avg_cycle
            # m = (t_win >= cycle_start) & (t_win <= cycle_end)
            # tr = (t_win[m] - cycle_start) / beat_len  # Convert to beat positions 1-4
            
            cycle_start = c
            cycle_end = onsets[i + 1] if i < len(onsets) - 1 else c + avg_cycle  # fallback to avg only for last cycle
            cycle_duration = cycle_end - cycle_start
            beat_len = cycle_duration / n_beats_per_cycle
            
            # Convert time to beat position using actual cycle duration
            m = (t_win >= cycle_start) & (t_win <= cycle_end)
            tr = (t_win[m] - cycle_start) / (cycle_duration / n_beats_per_cycle)  # normalize to 0-4 beats
  
            
            if show_trajectories:
                col = cmap((c-total_start)/t_range)
                
                # # Check if this cycle has any onsets
                # has_left_onsets = any(cycle_start <= lt <= cycle_end for lt in left_times)
                # has_right_onsets = any(cycle_start <= rt <= cycle_end for rt in right_times)
                
                # # Plot gray trajectories only for cycles without onsets
                # if show_gray_plots:
                #     if not has_left_onsets:
                #         ax.plot(tr, L_win[m], '-', color='black', alpha=0.3)
                #     if not has_right_onsets:
                #         ax.plot(tr, R_win[m], '--', color='black', alpha=0.3)
                
                # # Plot colored trajectories for cycles with onsets
                # if has_left_onsets:
                #     ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                # if has_right_onsets:
                #     ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                
                # Check if this cycle has any onsets at all
                has_any_onsets = any(cycle_start <= t <= cycle_end for t in left_times) or any(cycle_start <= t <= cycle_end for t in right_times)
                
                # Plot gray trajectories only for cycles without any onsets
                if show_gray_plots and not has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color='gray', alpha=0.9)
                    ax.plot(tr, R_win[m], '--', color='gray', alpha=0.9)
                
                # Plot colored trajectories for cycles with any onsets
                if has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                    ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                            
                # Store ALL trajectories for grand average (removed the condition)
                all_L_trajectories.append(L_win[m])
                all_R_trajectories.append(R_win[m])
                all_times.append(tr)

                # Plot markers for foot onsets
                for lt in left_times:
                    if cycle_start <= lt <= cycle_end:
                        rel = (lt - cycle_start) / beat_len
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
                        ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)

                for rt in right_times:
                    if cycle_start <= rt <= cycle_end:
                        rel = (rt - cycle_start) / beat_len
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
                        ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)

    # Calculate and plot grand average
    if all_L_trajectories and all_R_trajectories:
        # Interpolate all trajectories to the same time points
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        
        L_avg = np.zeros(len(common_times))
        R_avg = np.zeros(len(common_times))
        count = 0
        
        for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
            if len(t_traj) > 1:  # Only use trajectories with more than one point
                L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
                R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
                L_avg += L_interp(common_times)
                R_avg += R_interp(common_times)
                count += 1
        
        if count > 0:
            L_avg /= count
            R_avg /= count
            ax.plot(common_times, L_avg, '-', color='blue', linewidth=3, label='Left Foot Average')
            ax.plot(common_times, R_avg, '--', color='red', linewidth=3, label='Right Foot Average')

    # # Calculate and plot grand average combined left and right
    # if all_L_trajectories and all_R_trajectories:
    #     # Interpolate all trajectories to the same time points
    #     common_times = np.linspace(0, n_beats_per_cycle, 100)
        
    #     # Single array for combined average
    #     combined_avg = np.zeros(len(common_times))
    #     count = 0
        
    #     for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
    #         if len(t_traj) > 1:  # Only use trajectories with more than one point
    #             L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
    #             R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
    #             # Combine left and right trajectories
    #             combined_avg += (L_interp(common_times) + R_interp(common_times)) / 2
    #             count += 1
        
    #     if count > 0:
    #         combined_avg /= count
    #         ax.plot(common_times, combined_avg, '-', color='purple', linewidth=3, label='Combined Foot Average')
        
    # Add vertical line at position 0 (will display as 1)
    ax.axvline(0, color='black', linewidth=2, alpha=0.8)
    
    # Add beat and subdivision lines
    for beat in range(1, n_beats_per_cycle + 1):
        ax.axvline(beat, color='black', linewidth=2, alpha=0.8)
        # Add subdivision lines
        for subdiv in range(1, n_subdiv_per_beat):
            subdiv_pos = beat - 1 + subdiv/n_subdiv_per_beat
            subdiv_num = (beat - 1) * n_subdiv_per_beat + subdiv + 1
            
            grid_color = get_subdiv_color(subdiv_num)
            ax.axvline(subdiv_pos, color=grid_color, alpha=0.8, linewidth=1.5)
            
            

    ax.set_xlabel("Cycle Span")
    ax.set_ylabel("Foot Y Position")
    ax.set_title(f"All Cycles Trajectories with Grand Average\n{file_name} | {mode}")
    # ax.grid(True, alpha=0.3)
    xticks = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0])
    ax.set_xlim(0.0, 4.0)
    # Add all legends together outside the plot
    custom = [
        # Main trajectory and onset markers
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],marker='x', color='red', ms=8),
        Line2D([0],[0],color='blue', lw=3),
        Line2D([0],[0],color='red', lw=3, linestyle='--'),
        # Line2D([0],[0],color='purple', lw=3),  # For combined average
        # Subdivision lines
        Line2D([0],[0],color='gray', lw=1.5),
        Line2D([0],[0],color='black', lw=1.5),
        Line2D([0],[0],color='green', lw=1.5),
        Line2D([0],[0],color='red', lw=1.5)
    ]
    labels = [
        "Left Onset", 
        "Right Onset", 
        "Left Foot Average", 
        "Right Foot Average", 
        # "Combined Average",
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)", 
        "Subdivision 2 (2,5,8,11)", 
        "Subdivision 3 (3,6,9,12)"
    ]
    fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), framealpha=0.3)
    

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time in recording (s)')

    plt.tight_layout()
    return fig, ax


def plot_trajectories_downbeat_window(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/logs_v1_may",
    frame_rate: float = 240,
    W_start: float = 170.0,
    W_end: float = 185.0,
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 12,
    nn: int = 8,
    figsize: tuple = (12, 6),
    dpi: int = 100,
    use_cycles: bool = True,  # New parameter to control x-axis units
    show_gray_plots: bool = True        # show non detected foot onsets trajectories
):
    """
    Plot left- and right-foot Y-position trajectories ±window around each downbeat,
    marking foot-onset times for cycles that have an onset in the window.
    Also plots trajectories for cycles without onsets in gray.

    Parameters
    ----------
    file_name : str
        Base name (e.g. "BKO_E1_D2_03_Suku")
    mode : str
        Dance mode (e.g. "group", "individual", "audience")
    base_path_cycles : str
        Directory containing your virtual_cycles CSVs
    base_path_logs : str
        Directory containing your logs_v1_may/<file>_T/onset_info
    frame_rate : float
        Frame rate of the motion capture data
    W_start : float
        Start time (in seconds) of the window to plot trajectories
    W_end : float
        End time (in seconds) of the window to plot trajectories
    n_beats_per_cycle : int
        Beats per cycle (e.g. 4)
    n_subdiv_per_beat : int
        Subdivisions per beat (e.g. 12)
    nn : int
        Half-width in subdivisions: window_size_subdiv = 2*n_subdivs_per_side
    figsize : tuple
        Matplotlib figure size tuple
    dpi : int
        Matplotlib figure DPI
    use_cycles : bool
        If True, x-axis shows cycles (0 = downbeat, 1 = next downbeat)
        If False, x-axis shows seconds relative to downbeat
    show_gray_plots : bool
        If True, show trajectories for cycles without detected onsets in gray

    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes objects
    """
    # build file paths
    cycles_csv = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T", "onset_info")
    left_onsets_csv  = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_onsets.csv")
    right_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_onsets.csv")
    left_zpos_csv    = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_zpos.csv")
    right_zpos_csv   = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_zpos.csv")

    # load data
    Lz = pd.read_csv(left_zpos_csv)["zpos"].values
    Rz = pd.read_csv(right_zpos_csv)["zpos"].values
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate

    # interpolation functions
    L_interp = interp1d(times, Lz, bounds_error=False, fill_value="extrapolate")
    R_interp = interp1d(times, Rz, bounds_error=False, fill_value="extrapolate")

    # trim to window
    win_mask = (times >= W_start) & (times <= W_end)
    t_win = times[win_mask]
    L_win = Lz[win_mask]
    R_win = Rz[win_mask]

    # cycles (downbeats)
    cyc_df = pd.read_csv(cycles_csv)
    cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= W_start) & (cyc_df["Virtual Onset"] <= W_end)]
    onsets = cyc_df["Virtual Onset"].values[:-1]
    durations = np.diff(cyc_df["Virtual Onset"].values)
    avg_cycle = durations.mean()

    # foot onsets
    left_df  = pd.read_csv(left_onsets_csv)
    right_df = pd.read_csv(right_onsets_csv)
    left_times  = left_df[ (left_df["time_sec"]>=W_start)&(left_df["time_sec"]<=W_end) ]["time_sec"].values
    right_times = right_df[(right_df["time_sec"]>=W_start)&(right_df["time_sec"]<=W_end)]["time_sec"].values

    # window half-width in seconds
    beat_len   = avg_cycle / n_beats_per_cycle
    subdiv_len = beat_len / n_subdiv_per_beat
    half_win   = subdiv_len * nn

    # collect cycles that have foot onsets
    cyc_L, L_near = [], {}
    for c in onsets:
        hits = left_times[(left_times>=c-half_win)&(left_times<=c+half_win)]
        if len(hits):
            cyc_L.append(c); L_near[c] = hits

    cyc_R, R_near = [], {}
    for c in onsets:
        hits = right_times[(right_times>=c-half_win)&(right_times<=c+half_win)]
        if len(hits):
            cyc_R.append(c); R_near[c] = hits

    # plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap('cool')
    t_range = W_end - W_start

    # Plot all cycles first (gray/transparent)
    if show_gray_plots:
        for c in onsets:
            m = (t_win>=c-half_win)&(t_win<=c+half_win)
            tr = t_win[m] - c
            if use_cycles:
                tr = tr / avg_cycle  # Convert to cycles
            # Plot left foot
            ax.plot(tr, L_win[m], '-', color='gray', alpha=0.5)
            # Plot right foot
            ax.plot(tr, R_win[m], '--', color='gray', alpha=0.5)

    # left foot (with onsets)
    for i, c in enumerate(cyc_L):
        col = cmap((c-W_start)/t_range)
        m = (t_win>=c-half_win)&(t_win<=c+half_win)
        tr = t_win[m] - c
        if use_cycles:
            tr = tr / avg_cycle  # Convert to cycles
        ax.plot(tr, L_win[m], '-', color=col, alpha=0.3,
                label="Left Foot" if i==0 else "")
        for lt in L_near[c]:
            rel = lt - c
            if use_cycles:
                rel = rel / avg_cycle  # Convert to cycles
            ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
            ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', alpha=0.8)

    # right foot (with onsets)
    for i, c in enumerate(cyc_R):
        col = cmap((c-W_start)/t_range)
        m = (t_win>=c-half_win)&(t_win<=c+half_win)
        tr = t_win[m] - c
        if use_cycles:
            tr = tr / avg_cycle  # Convert to cycles
        ax.plot(tr, R_win[m], '--', color=col, alpha=0.3,
                label="Right Foot" if i==0 else "")
        for rt in R_near[c]:
            rel = rt - c
            if use_cycles:
                rel = rel / avg_cycle  # Convert to cycles
            ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
            ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='k', alpha=0.8)

    # decorations
    ax.axvline(0, color='k', linewidth=1.5, label="Downbeat (t=0)")
    # subdivision lines
    for j in range(-nn, nn+1):
        if j!=0:
            pos = j*subdiv_len
            if use_cycles:
                pos = pos / avg_cycle  # Convert to cycles
            ax.axvline(pos, color='gray', linestyle=':', alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(W_start, W_end))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time in recording (s)')

    custom = [
        Line2D([0],[0],color='blue',linestyle='-', lw=2),
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],color='blue',linestyle='--', lw=2),
        Line2D([0],[0],marker='x', color='w', markeredgecolor='blue', ms=8),
        Line2D([0],[0],color='k', lw=2)
    ]
    labels = ["Left Trajectory","Left Onset","Right Trajectory",
              "Right Onset","Downbeat (t=0)"]
    ax.legend(custom, labels, loc='upper left', framealpha=0.3)

    xlabel = "Cycles relative to downbeat" if use_cycles else "Time relative to downbeat (s)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Foot Y Position")
    ax.set_title(
        f"Foot Trajectories ±{2*nn/n_subdiv_per_beat/ n_beats_per_cycle:.1f} beats around downbeats\n"
        f"{file_name} | window {W_start}-{W_end}s | {mode}",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_cycles_trajectories_by_threshold(
    file_name: str,
    mode: str,
    pelvis_zpos: np.ndarray,  # New parameter
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,
    time_segments: list = None,  # List of (start, end) tuples
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    traj_threshold=0.15,
    figsize: tuple = (12, 12),  # Increased height for two plots
    dpi: int = 200,
    show_trajectories: bool = True,  # Control trajectory lines
    show_vlines: bool = True,        # Control vertical lines
    show_gray_plots: bool = True     # Control gray trajectory plots
):
    """
    Plot all foot trajectories in a single plot with grand average.
    Shows beat and subdivision lines with colors.
    X-axis shows 1.5 cycles (e.g., 0-6 for 4-beat cycles).
    Top subplot shows pelvis trajectories for included cycles.
    """
    print(time_segments)
    # Use default window if no segments provided
    if time_segments is None:
        time_segments = [(0, 10)]

    # build file paths
    cycles_csv = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T", "onset_info")
    left_onsets_csv  = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_onsets.csv")
    right_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_onsets.csv")
    left_zpos_csv    = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_zpos.csv")
    right_zpos_csv   = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_zpos.csv")

    # load data
    Lz = pd.read_csv(left_zpos_csv)["zpos"].values
    Rz = pd.read_csv(right_zpos_csv)["zpos"].values
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate

    # interpolation functions
    L_interp = interp1d(times, Lz, bounds_error=False, fill_value="extrapolate")
    R_interp = interp1d(times, Rz, bounds_error=False, fill_value="extrapolate")

    # Get overall time range for color mapping
    total_start = min(seg[0] for seg in time_segments)
    total_end = max(seg[1] for seg in time_segments)
    t_range = total_end - total_start

    # Get all onsets
    all_onsets = []
    for seg_start, seg_end in time_segments:
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not cyc_df.empty:
            all_onsets.extend(cyc_df["Virtual Onset"].values[:-1])
    
    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")
    
    # Sort onsets
    all_onsets = np.sort(all_onsets)

    # Create figure with two subplots
    fig, (ax_pelvis, ax_feet) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap('cool')

    # Define subdivision color mapping
    def get_subdiv_color(subdiv):
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        group = ((subdiv - 1) % 3) + 1
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'

    # Process each time segment
    all_L_trajectories = []
    all_R_trajectories = []
    all_times = []
    included_cycles = []  # New list to store cycle start times
    excluded_cycles = []    # New list to store cycle end times
    all_pelvis_trajectories = [] 
    
    for seg_start, seg_end in time_segments:
        # trim to window
        win_mask = (times >= seg_start) & (times <= seg_end)
        t_win = times[win_mask]
        L_win = Lz[win_mask]
        R_win = Rz[win_mask]
        pelvis_win = pelvis_zpos[win_mask]  # Get pelvis data for this window

        # cycles (downbeats)
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        onsets = cyc_df["Virtual Onset"].values[:-1]

        # foot onsets
        left_df  = pd.read_csv(left_onsets_csv)
        right_df = pd.read_csv(right_onsets_csv)
        left_times  = left_df[(left_df["time_sec"]>=seg_start)&(left_df["time_sec"]<=seg_end)]["time_sec"].values
        right_times = right_df[(right_df["time_sec"]>=seg_start)&(right_df["time_sec"]<=seg_end)]["time_sec"].values

        # Plot trajectories for each cycle
        for i, c in enumerate(onsets):
            # Get cycle start and end from actual onsets
            cycle_start = c
            cycle_end = onsets[i + 1] if i < len(onsets) - 1 else c + (onsets[1] - onsets[0])  # fallback to first cycle duration
            cycle_duration = cycle_end - cycle_start
            beat_len = cycle_duration / n_beats_per_cycle
            
            m = (t_win >= cycle_start) & (t_win <= cycle_end)
            tr = (t_win[m] - cycle_start) / beat_len  # normalize to 0-4 beats

            if show_trajectories:
                col = cmap((c-total_start)/t_range)
                has_any_onsets = any(cycle_start <= t <= cycle_end for t in left_times) or any(cycle_start <= t <= cycle_end for t in right_times)
                
                # Plot gray trajectories only for cycles without any onsets
                if show_gray_plots and not has_any_onsets:
                    ax_feet.plot(tr, L_win[m], '-', color='gray', alpha=0.9)
                    ax_feet.plot(tr, R_win[m], '--', color='gray', alpha=0.9)
                    # Also plot shifted for 1.5 cycles
                    ax_feet.plot(tr + 4, L_win[m], '-', color='gray', alpha=0.9)
                    ax_feet.plot(tr + 4, R_win[m], '--', color='gray', alpha=0.9)
                
                # Plot colored trajectories for cycles with any onsets
                if has_any_onsets:
                    ax_feet.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                    ax_feet.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                    # Also plot shifted for 1.5 cycles
                    ax_feet.plot(tr + 4, L_win[m], '-', color=col, alpha=0.3)
                    ax_feet.plot(tr + 4, R_win[m], '--', color=col, alpha=0.3)
                
                # Apply thresholding when storing trajectories for averaging
                traj_max = max(np.nanmax(L_win[m]), np.nanmax(R_win[m]))
                if traj_max >= traj_threshold:
                    all_L_trajectories.append(L_win[m])
                    all_R_trajectories.append(R_win[m])
                    all_pelvis_trajectories.append(pelvis_win[m])  # Add pelvis trajectory
                    all_times.append(tr)
                    # Store the cycle timing information for included trajectories
                    included_cycles.append((cycle_start, cycle_end))
                    # Plot pelvis trajectory for included cycles
                    ax_pelvis.plot(tr, pelvis_win[m], '-', color=col, alpha=0.3)
                    ax_pelvis.plot(tr + 4, pelvis_win[m], '-', color=col, alpha=0.3)
                else:
                    excluded_cycles.append((cycle_start, cycle_end))

                # Plot markers for foot onsets
                for lt in left_times:
                    if cycle_start <= lt <= cycle_end:
                        rel = (lt - cycle_start) / beat_len
                        if show_vlines:
                            ax_feet.axvline(rel, color=col, linestyle='-', alpha=0.5)
                        ax_feet.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)
                        # Also plot shifted for 1.5 cycles
                        if show_vlines:
                            ax_feet.axvline(rel + 4, color=col, linestyle='-', alpha=0.5)
                        ax_feet.plot(rel + 4, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)

                for rt in right_times:
                    if cycle_start <= rt <= cycle_end:
                        rel = (rt - cycle_start) / beat_len
                        if show_vlines:
                            ax_feet.axvline(rel, color=col, linestyle='--', alpha=0.5)
                        ax_feet.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)
                        # Also plot shifted for 1.5 cycles
                        if show_vlines:
                            ax_feet.axvline(rel + 4, color=col, linestyle='--', alpha=0.5)
                        ax_feet.plot(rel + 4, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)
    
    print("count of included trajectories: ", len(included_cycles))
    print("count of excluded trajectories: ", len(excluded_cycles))
    # print(included_cycles)
    
    pickle_dir = "traj_files"
    # Save the dictionaries
    with open(os.path.join(pickle_dir, f'{file_name}_included_{traj_threshold}.pkl'), 'wb') as f:
        pickle.dump(included_cycles, f)
    
    with open(os.path.join(pickle_dir, f'{file_name}_excluded_{traj_threshold}.pkl'), 'wb') as f:
        pickle.dump(excluded_cycles, f)
    
    # Calculate and plot grand average for feet
    if all_L_trajectories and all_R_trajectories and all_pelvis_trajectories:
        # Interpolate all trajectories to the same time points
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        L_avg = np.zeros(len(common_times))
        R_avg = np.zeros(len(common_times))
        pelvis_avg = np.zeros(len(common_times))  # New array for pelvis average
        count = 0
        
        for L_traj, R_traj, P_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_pelvis_trajectories, all_times):
            L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
            R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
            P_interp = interp1d(t_traj, P_traj, bounds_error=False, fill_value="extrapolate")  # New interpolation
            L_avg += L_interp(common_times)
            R_avg += R_interp(common_times)
            pelvis_avg += P_interp(common_times)  # Add to pelvis average
            count += 1
        
        if count > 0:
            L_avg /= count
            R_avg /= count
            pelvis_avg /= count  # Normalize pelvis average
            
            # Plot foot averages
            ax_feet.plot(common_times, L_avg, '-', color='blue', linewidth=3, label='Left Foot Average')
            ax_feet.plot(common_times, R_avg, '--', color='red', linewidth=3, label='Right Foot Average')
            ax_feet.plot(common_times + 4, L_avg, '-', color='blue', linewidth=3)
            ax_feet.plot(common_times + 4, R_avg, '--', color='red', linewidth=3)
            
            # Plot pelvis average
            ax_pelvis.plot(common_times, pelvis_avg, '-', color='purple', linewidth=3, label='Pelvis Average')
            ax_pelvis.plot(common_times + 4, pelvis_avg, '-', color='purple', linewidth=3)

    # Add vertical lines and grid for both subplots
    for ax in [ax_pelvis, ax_feet]:
        # Add vertical line at position 0 (will display as 1)
        ax.axvline(0, color='black', linewidth=2, alpha=0.8)
        ax.axvline(4, color='black', linewidth=2, alpha=0.8)  # Start of second cycle

        # Add beat and subdivision lines for 1.5 cycles
        for cycle in range(2):  # 0 and 1
            for beat in range(1, n_beats_per_cycle + 1):
                pos = cycle * n_beats_per_cycle + beat
                ax.axvline(pos, color='black', linewidth=2, alpha=0.8)
                # Add subdivision lines
                for subdiv in range(1, n_subdiv_per_beat):
                    subdiv_pos = cycle * n_beats_per_cycle + (beat - 1) + subdiv / n_subdiv_per_beat
                    subdiv_num = (beat - 1) * n_subdiv_per_beat + subdiv + 1
                    grid_color = get_subdiv_color(subdiv_num)
                    ax.axvline(subdiv_pos, color=grid_color, alpha=0.8, linewidth=1.5)

        # Set x-axis for 1.5 cycles
        xticks = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0, 5.33, 5.67, 6.0]
        xticklabels = ['1.00', '1.33', '1.67', '2.00', '2.33', '2.67', '3.00', '3.33', '3.67', '4.00', '4.33', '4.67', '5.00', '5.33', '5.67', '6.00', '6.33', '6.67', '7.00']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)  # Label as 1-based
        ax.set_xlim(0.0, 6.0)
        ax.grid(True, alpha=0.3)

    # Set labels and titles
    ax_pelvis.set_ylabel("Pelvis Y Position")
    ax_feet.set_ylabel("Foot Y Position")
    ax_feet.set_xlabel("Cycle Span")
    
    # Set titles
    ax_pelvis.set_title("Pelvis Trajectories")
    ax_feet.set_title(f"Foot Trajectories with Grand Average | {mode} | Threshold: {traj_threshold}")
    
    # Add all legends together outside the plot
    custom = [
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],marker='x', color='red', ms=8),
        Line2D([0],[0],color='blue', lw=3),
        Line2D([0],[0],color='red', lw=3, linestyle='--'),
        Line2D([0],[0],color='purple', lw=3),  # Add pelvis average to legend
        Line2D([0],[0],color='gray', lw=1.5),
        Line2D([0],[0],color='black', lw=1.5),
        Line2D([0],[0],color='green', lw=1.5),
        Line2D([0],[0],color='red', lw=1.5)
    ]
    labels = [
        "Left Onset", 
        "Right Onset", 
        "Left Foot Average", 
        "Right Foot Average", 
        "Pelvis Average",  # Add pelvis average label
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)", 
        "Subdivision 2 (2,5,8,11)", 
        "Subdivision 3 (3,6,9,12)"
    ]
    # fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), framealpha=0.3)

    # # Add colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    # sm.set_array([])
    # plt.colorbar(sm, ax=[ax_pelvis, ax_feet], label='Time in recording (s)')

    # plt.tight_layout()
    # return fig, (ax_pelvis, ax_feet)

    # Create a new axes for the legend
    legend_ax = fig.add_axes([1.02, 0.1, 0.1, 0.8])
    legend_ax.axis('off')
    legend_ax.legend(custom, labels, loc='center left', framealpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
    plt.colorbar(sm, cax=cbar_ax, label='Time in recording (s)')

    # Adjust layout to make room for legend and colorbar
    fig.subplots_adjust(right=0.85)
    
    return fig, (ax_pelvis, ax_feet)



########################################################
############## HANDS TRAJECTORIES PLOT #################
########################################################

def plot_hand_cycles_trajectories(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/output-nov13",
    frame_rate: float = 240,
    time_segments: list = None,  # List of (start, end) tuples
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    figsize: tuple = (12, 6),
    dpi: int = 200,
    show_trajectories: bool = True,  # Control trajectory lines
    show_vlines: bool = True,        # Control vertical lines
    show_gray_plots: bool = True     # Control gray trajectory plots
):
    """
    Plot all foot trajectories in a single plot with grand average.
    Shows beat and subdivision lines with colors.
    X-axis shows beats 1-4 directly.
    """
    # Use default window if no segments provided
    if time_segments is None:
        time_segments = [(0, 10)]

    # build file paths
    cycles_csv_path = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T")
    onset_data_path = os.path.join(logs_onset_dir, f"{file_name}_T_hands_onset.pkl")
    
    with open(onset_data_path, 'rb') as f:
        onset_data = pickle.load(f)

    mocap_dir = "data/motion_data_pkl"
    mocap_path = os.path.join(mocap_dir, f"{file_name}_T.pkl")
    with open(mocap_path, 'rb') as f:
        mocap_data = pickle.load(f)
    

    
    Lz = mocap_data['segments']['LeftHand']['position'][:,2]
    Rz = mocap_data['segments']['RightHand']['position'][:,2]
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate

    # interpolation functions
    L_interp = interp1d(times, Lz, bounds_error=False, fill_value="extrapolate")
    R_interp = interp1d(times, Rz, bounds_error=False, fill_value="extrapolate")

    # Get overall time range for color mapping
    total_start = min(seg[0] for seg in time_segments)
    total_end = max(seg[1] for seg in time_segments)
    t_range = total_end - total_start

    # Calculate average cycle duration from all segments
    all_onsets = []
    cyc_df = pd.read_csv(cycles_csv_path)
    for seg_start, seg_end in time_segments:
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not cyc_df.empty:
            all_onsets.extend(cyc_df["Virtual Onset"].values[:-1])
    
    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")
    
    all_onsets = np.sort(all_onsets)
    
    # Calculate average cycle duration
    durations = np.diff(sorted(all_onsets))
    avg_cycle = durations.mean()

    # Calculate beat and subdivision lengths
    # beat_len = avg_cycle / n_beats_per_cycle
    # subdiv_len = beat_len / n_subdiv_per_beat

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap('cool')

    # Define subdivision color mapping
    def get_subdiv_color(subdiv):
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        group = ((subdiv - 1) % 3) + 1
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'

    # Process each time segment
    all_L_trajectories = []
    all_R_trajectories = []
    all_times = []

    for seg_start, seg_end in time_segments:
        # trim to window
        win_mask = (times >= seg_start) & (times <= seg_end)
        t_win = times[win_mask]
        L_win = Lz[win_mask]
        R_win = Rz[win_mask]

        # cycles (downbeats)
        # cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        onsets = cyc_df["Virtual Onset"].values[:-1]
        
        left_df = onset_data["left_hand"]
        right_df = onset_data["right_hand"]
        
        mask_left = (left_df['time_sec'] >= seg_start) & (left_df['time_sec'] <= seg_end)
        mask_right = (right_df['time_sec'] >= seg_start) & (right_df['time_sec'] <= seg_end)

        left_times = left_df['time_sec'][mask_left]
        right_times = right_df['time_sec'][mask_right]
        

        # Plot trajectories for each cycle
        # for c in onsets:
        for i, c in enumerate(onsets):
            # Convert time to beat position (1-4)
            # cycle_start = c
            # cycle_end = c + avg_cycle
            # m = (t_win >= cycle_start) & (t_win <= cycle_end)
            # tr = (t_win[m] - cycle_start) / beat_len  # Convert to beat positions 1-4
            
            cycle_start = c
            cycle_end = onsets[i + 1] if i < len(onsets) - 1 else c + avg_cycle  # fallback to avg only for last cycle
            cycle_duration = cycle_end - cycle_start
            beat_len = cycle_duration / n_beats_per_cycle
            
            # Convert time to beat position using actual cycle duration
            m = (t_win >= cycle_start) & (t_win <= cycle_end)
            tr = (t_win[m] - cycle_start) / (cycle_duration / n_beats_per_cycle)  # normalize to 0-4 beats
  
            
            if show_trajectories:
                col = cmap((c-total_start)/t_range)
                
                # # Check if this cycle has any onsets
                # has_left_onsets = any(cycle_start <= lt <= cycle_end for lt in left_times)
                # has_right_onsets = any(cycle_start <= rt <= cycle_end for rt in right_times)
                
                # # Plot gray trajectories only for cycles without onsets
                # if show_gray_plots:
                #     if not has_left_onsets:
                #         ax.plot(tr, L_win[m], '-', color='black', alpha=0.3)
                #     if not has_right_onsets:
                #         ax.plot(tr, R_win[m], '--', color='black', alpha=0.3)
                
                # # Plot colored trajectories for cycles with onsets
                # if has_left_onsets:
                #     ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                # if has_right_onsets:
                #     ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                
                # Check if this cycle has any onsets at all
                has_any_onsets = any(cycle_start <= t <= cycle_end for t in left_times) or any(cycle_start <= t <= cycle_end for t in right_times)
                
                # Plot gray trajectories only for cycles without any onsets
                if show_gray_plots and not has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color='gray', alpha=0.9)
                    ax.plot(tr, R_win[m], '--', color='gray', alpha=0.9)
                
                # Plot colored trajectories for cycles with any onsets
                if has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Hand" if c==onsets[0] else "")
                    ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Hand" if c==onsets[0] else "")
                            
                # Store ALL trajectories for grand average (removed the condition)
                all_L_trajectories.append(L_win[m])
                all_R_trajectories.append(R_win[m])
                all_times.append(tr)

                # Plot markers for hand onsets
                # for lt in left_times:
                #     if cycle_start <= lt <= cycle_end:
                #         rel = (lt - cycle_start) / beat_len
                #         if show_vlines:
                #             ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
                #         ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                #                 markerfacecolor='blue', alpha=0.8)

                # for rt in right_times:
                #     if cycle_start <= rt <= cycle_end:
                #         rel = (rt - cycle_start) / beat_len
                #         if show_vlines:
                #             ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
                #         ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                #                 color='red', alpha=0.8)

    # Calculate and plot grand average
    if all_L_trajectories and all_R_trajectories:
        # Interpolate all trajectories to the same time points
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        
        L_avg = np.zeros(len(common_times))
        R_avg = np.zeros(len(common_times))
        count = 0
        
        for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
            if len(t_traj) > 1:  # Only use trajectories with more than one point
                L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
                R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
                L_avg += L_interp(common_times)
                R_avg += R_interp(common_times)
                count += 1
        
        if count > 0:
            L_avg /= count
            R_avg /= count
            ax.plot(common_times, L_avg, '-', color='blue', linewidth=3, label='Left Hand Average')
            ax.plot(common_times, R_avg, '--', color='red', linewidth=3, label='Right Hand Average')

    # # Calculate and plot grand average combined left and right
    # if all_L_trajectories and all_R_trajectories:
    #     # Interpolate all trajectories to the same time points
    #     common_times = np.linspace(0, n_beats_per_cycle, 100)
        
    #     # Single array for combined average
    #     combined_avg = np.zeros(len(common_times))
    #     count = 0
        
    #     for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
    #         if len(t_traj) > 1:  # Only use trajectories with more than one point
    #             L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
    #             R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
    #             # Combine left and right trajectories
    #             combined_avg += (L_interp(common_times) + R_interp(common_times)) / 2
    #             count += 1
        
    #     if count > 0:
    #         combined_avg /= count
    #         ax.plot(common_times, combined_avg, '-', color='purple', linewidth=3, label='Combined Foot Average')
        
    # Add vertical line at position 0 (will display as 1)
    ax.axvline(0, color='black', linewidth=2, alpha=0.8)
    
    # Add beat and subdivision lines
    for beat in range(1, n_beats_per_cycle + 1):
        ax.axvline(beat, color='black', linewidth=2, alpha=0.8)
        # Add subdivision lines
        for subdiv in range(1, n_subdiv_per_beat):
            subdiv_pos = beat - 1 + subdiv/n_subdiv_per_beat
            subdiv_num = (beat - 1) * n_subdiv_per_beat + subdiv + 1
            
            grid_color = get_subdiv_color(subdiv_num)
            ax.axvline(subdiv_pos, color=grid_color, alpha=0.8, linewidth=1.5)
            
            

    ax.set_xlabel("Cycle Span")
    ax.set_ylabel("Hand Y Position")
    ax.set_title(f"All Cycles Trajectories with Grand Average\n{file_name} | {mode}")
    # ax.grid(True, alpha=0.3)
    xticks = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0])
    ax.set_xlim(0.0, 4.0)
    # Add all legends together outside the plot
    custom = [
        # Main trajectory and onset markers
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],marker='x', color='red', ms=8),
        Line2D([0],[0],color='blue', lw=3),
        Line2D([0],[0],color='red', lw=3, linestyle='--'),
        
        Line2D([0],[0],color=cmap(0.5), lw=2, alpha=0.3),  # Individual Left Hand trajectories (colored by time)
        Line2D([0],[0],color=cmap(0.5), lw=2, linestyle='--', alpha=0.3),  # Individual Right Hand trajectories (colored by time)
        
        # Line2D([0],[0],color='purple', lw=3),  # For combined average
        
        
        
        # Subdivision lines
        Line2D([0],[0],color='gray', lw=1.5),
        Line2D([0],[0],color='black', lw=1.5),
        Line2D([0],[0],color='green', lw=1.5),
        Line2D([0],[0],color='red', lw=1.5)
    ]
    labels = [
        "Left Onset", 
        "Right Onset", 
        "Left Hand Average", 
        "Right Hand Average", 
        "Left Hand",
        "Right Hand",
        # "Combined Average",
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)", 
        "Subdivision 2 (2,5,8,11)", 
        "Subdivision 3 (3,6,9,12)"
    ]
    fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), framealpha=0.3)
    

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time in recording (s)')

    plt.tight_layout()
    return fig, ax


######## Function Hand Distance and relative velocity norm

def plot_hand_norms_trajectories(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    frame_rate: float = 240,
    time_segments: list = None,
    select_axis: int = None,   # None: 3D, 0: x, 1: y, 2: z
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    figsize: tuple = (12, 10),
    dpi: int = 200,
    show_trajectories: bool = True,
    show_vlines: bool = True,
    show_gray_plots: bool = True  # retained for signature compatibility, unused
):
    """
    Plot per-cycle trajectories for:
      - Euclidean distance between left/right hands.
      - Relative velocity norm ||V_left - V_right||.
    Uses cycle timing from the virtual cycles CSV; no onset data is used.
    """
    if time_segments is None:
        time_segments = [(0, 10)]

    cycles_csv_path = os.path.join(base_path_cycles, f"{file_name}_C.csv")

    # Load mocap positions
    mocap_dir = "data/motion_data_pkl"
    mocap_path = os.path.join(mocap_dir, f"{file_name}_T.pkl")
    with open(mocap_path, 'rb') as f:
        mocap_data = pickle.load(f)

    # 3D positions (or set select_axis=None to keep full vectors)
    Lpos_full = mocap_data['segments']['LeftHand']['position']
    Rpos_full = mocap_data['segments']['RightHand']['position']
    if select_axis is None:
        Lpos, Rpos = Lpos_full, Rpos_full                     # shape (N,3)
    else:
        Lpos, Rpos = Lpos_full[:, select_axis], Rpos_full[:, select_axis]  # shape (N,)
    
    n_frames = len(Lpos)
    times = np.arange(n_frames) / frame_rate

    # Distance, relative velocity, and per-hand acceleration (elementwise)
    if select_axis is None:
        hand_distance = np.linalg.norm(Lpos - Rpos, axis=1)
        L_vel = np.gradient(Lpos, 1.0/frame_rate, axis=0)
        R_vel = np.gradient(Rpos, 1.0/frame_rate, axis=0)
        rel_speed = np.linalg.norm(L_vel - R_vel, axis=1)
        L_acc = np.gradient(L_vel, 1.0/frame_rate, axis=0)
        R_acc = np.gradient(R_vel, 1.0/frame_rate, axis=0)
    else:
        hand_distance = np.abs(Lpos - Rpos)
        L_vel = np.gradient(Lpos, 1.0/frame_rate)
        R_vel = np.gradient(Rpos, 1.0/frame_rate)
        rel_speed = np.abs(L_vel - R_vel)
        L_acc = np.gradient(L_vel, 1.0/frame_rate)
        R_acc = np.gradient(R_vel, 1.0/frame_rate)

    # Interpolators (used for averaging)
    dist_interp_full = interp1d(times, hand_distance, bounds_error=False, fill_value="extrapolate")
    speed_interp_full = interp1d(times, rel_speed, bounds_error=False, fill_value="extrapolate")
    L_acc_interp_full = interp1d(times, L_acc, bounds_error=False, fill_value="extrapolate")
    R_acc_interp_full = interp1d(times, R_acc, bounds_error=False, fill_value="extrapolate")

    total_start = min(seg[0] for seg in time_segments)
    total_end = max(seg[1] for seg in time_segments)
    t_range = total_end - total_start

    # Average cycle duration
    all_onsets = []
    cyc_df = pd.read_csv(cycles_csv_path)
    for seg_start, seg_end in time_segments:
        seg_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not seg_df.empty:
            all_onsets.extend(seg_df["Virtual Onset"].values[:-1])

    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")

    all_onsets = np.sort(all_onsets)
    durations = np.diff(sorted(all_onsets))
    avg_cycle = durations.mean()

    fig, (ax_dist, ax_vel, ax_acc) = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    cmap = plt.get_cmap('cool')

    def get_subdiv_color(subdiv):
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        group = ((subdiv - 1) % 3) + 1
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'

    all_dist_traj, all_speed_traj, all_L_acc_traj, all_R_acc_traj, all_times = [], [], [], [], []

    for seg_start, seg_end in time_segments:
        win_mask = (times >= seg_start) & (times <= seg_end)
        t_win = times[win_mask]
        dist_win = hand_distance[win_mask]
        speed_win = rel_speed[win_mask]
        L_acc_win = L_acc[win_mask]
        R_acc_win = R_acc[win_mask]

        seg_cycles = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        onsets = seg_cycles["Virtual Onset"].values[:-1]

        for i, c in enumerate(onsets):
            cycle_start = c
            cycle_end = onsets[i + 1] if i < len(onsets) - 1 else c + avg_cycle
            cycle_duration = cycle_end - cycle_start
            beat_len = cycle_duration / n_beats_per_cycle

            m = (t_win >= cycle_start) & (t_win <= cycle_end)
            tr = (t_win[m] - cycle_start) / beat_len  # normalized to beats (0-4)

            if show_trajectories:
                col = cmap((c - total_start) / t_range)
                ax_dist.plot(tr, dist_win[m], '-', color=col, alpha=0.35,
                             label="Distance (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")
                ax_vel.plot(tr, speed_win[m], '--', color=col, alpha=0.35,
                            label="Rel speed (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")
                ax_acc.plot(tr, L_acc_win[m], ':', color=col, alpha=0.35,
                            label="Left accel (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")
                ax_acc.plot(tr, R_acc_win[m], '-.', color=col, alpha=0.35,
                            label="Right accel (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")

                all_dist_traj.append(dist_win[m])
                all_speed_traj.append(speed_win[m])
                all_L_acc_traj.append(L_acc_win[m])
                all_R_acc_traj.append(R_acc_win[m])
                all_times.append(tr)

    # Grand averages
    if all_dist_traj and all_speed_traj and all_L_acc_traj and all_R_acc_traj:
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        dist_avg = np.zeros_like(common_times)
        speed_avg = np.zeros_like(common_times)
        L_acc_avg = np.zeros_like(common_times)
        R_acc_avg = np.zeros_like(common_times)
        count = 0
        for d_traj, s_traj, La_traj, Ra_traj, t_traj in zip(all_dist_traj, all_speed_traj, all_L_acc_traj, all_R_acc_traj, all_times):
            if len(t_traj) > 1:
                d_interp = interp1d(t_traj, d_traj, bounds_error=False, fill_value="extrapolate")
                s_interp = interp1d(t_traj, s_traj, bounds_error=False, fill_value="extrapolate")
                La_interp = interp1d(t_traj, La_traj, bounds_error=False, fill_value="extrapolate")
                Ra_interp = interp1d(t_traj, Ra_traj, bounds_error=False, fill_value="extrapolate")
                dist_avg += d_interp(common_times)
                speed_avg += s_interp(common_times)
                L_acc_avg += La_interp(common_times)
                R_acc_avg += Ra_interp(common_times)
                count += 1
        if count > 0:
            dist_avg /= count
            speed_avg /= count
            L_acc_avg /= count
            R_acc_avg /= count
            ax_dist.plot(common_times, dist_avg, color='blue', linewidth=3, label='Distance average')
            ax_vel.plot(common_times, speed_avg, color='red', linewidth=3, label='Rel speed average')
            ax_acc.plot(common_times, L_acc_avg, color='blue', linewidth=3, label='Left accel average')
            ax_acc.plot(common_times, R_acc_avg, color='red', linewidth=3, label='Right accel average')

    # Vertical lines (beats + subdivisions)
    for ax in (ax_dist, ax_vel, ax_acc):
        ax.axvline(0, color='black', linewidth=2, alpha=0.8)
        for beat in range(1, n_beats_per_cycle + 1):
            ax.axvline(beat, color='black', linewidth=2, alpha=0.8)
            for subdiv in range(1, n_subdiv_per_beat):
                subdiv_pos = beat - 1 + subdiv / n_subdiv_per_beat
                grid_color = get_subdiv_color((beat - 1) * n_subdiv_per_beat + subdiv + 1)
                ax.axvline(subdiv_pos, color=grid_color, alpha=0.8, linewidth=1.5)

    ax_acc.set_xlabel("Cycle Span (beats)")
    ax_dist.set_ylabel("Hand distance")
    ax_vel.set_ylabel("‖V_left - V_right‖")
    ax_acc.set_ylabel("Hand acceleration")

    xticks = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0]
    ax_vel.set_xticks(xticks)
    ax_vel.set_xticklabels([1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0])
    ax_vel.set_xlim(0.0, 4.0)

    ax_dist.set_title(f"Hand Distance per Cycle with Grand Average\n{file_name} | {mode}")
    ax_vel.set_title(f"Relative Velocity Norm per Cycle\n{file_name} | {mode}")
    ax_acc.set_title(f"Left/Right Acceleration per Cycle\n{file_name} | {mode}")

    custom = [
        Line2D([0], [0], color=cmap(0.5), lw=2, alpha=0.35),
        Line2D([0], [0], color=cmap(0.5), lw=2, linestyle='--', alpha=0.35),
        Line2D([0], [0], color=cmap(0.5), lw=2, linestyle=':', alpha=0.35),
        Line2D([0], [0], color=cmap(0.5), lw=2, linestyle='-.', alpha=0.35),
        Line2D([0], [0], color='blue', lw=3),
        Line2D([0], [0], color='red', lw=3),
        Line2D([0], [0], color='blue', lw=3, linestyle=':'),
        Line2D([0], [0], color='red', lw=3, linestyle='-.'),
        Line2D([0], [0], color='gray', lw=1.5),
        Line2D([0], [0], color='black', lw=1.5),
        Line2D([0], [0], color='green', lw=1.5),
        Line2D([0], [0], color='red', lw=1.5)
    ]
    labels = [
        "Distance (cycles)",
        "Rel speed (cycles)",
        "Left accel (cycles)",
        "Right accel (cycles)",
        "Distance average",
        "Rel speed average",
        "Left accel average",
        "Right accel average",
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)",
        "Subdivision 2 (2,5,8,11)",
        "Subdivision 3 (3,6,9,12)"
    ]
    fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), framealpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    plt.colorbar(sm, ax=[ax_dist, ax_vel, ax_acc], label='Time in recording (s)')

    # plt.tight_layout()
    return fig, (ax_dist, ax_vel, ax_acc)


from scipy.signal import argrelmin, argrelmax

def plot_rel_hand_norms_trajectories(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    frame_rate: float = 240,
    time_segments: list = None,
    select_axis: int = None,   # None: 3D, 0: x, 1: y, 2: z
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    figsize: tuple = (12, 10),
    dpi: int = 200,
    show_trajectories: bool = True,
    min_gap_sec: float = 0.3,
    slope_min: float = 0.3,
    min_thresh: float = 0.19,
):
    """
    Plot per-cycle trajectories for:
      - Euclidean distance between left/right hands.
      - Relative velocity norm ||V_left - V_right||.
      - Relative acceleration norm ||A_left - A_right||.
    Uses cycle timing from the virtual cycles CSV; no onset data is used.
    """
    if time_segments is None:
        time_segments = [(0, 10)]

    cycles_csv_path = os.path.join(base_path_cycles, f"{file_name}_C.csv")

    # # Load mocap positions
    # mocap_dir = "data/motion_data_pkl"
    # mocap_path = os.path.join(mocap_dir, f"{file_name}_T.pkl")
    
    # with open(mocap_path, 'rb') as f:
    #     mocap_data = pickle.load(f)

    # # 3D positions (or set select_axis=None to keep full vectors)
    # Lpos_full = mocap_data['segments']['LeftHand']['position']
    # Rpos_full = mocap_data['segments']['RightHand']['position']
    
    # if select_axis is None:
    #     Lpos, Rpos = Lpos_full, Rpos_full                     # shape (N,3)
    # else:
    #     Lpos, Rpos = Lpos_full[:, select_axis], Rpos_full[:, select_axis]  # shape (N,)
    
    # n_frames = len(Lpos)
    # times = np.arange(n_frames) / frame_rate

    # # Distance, relative velocity, and relative acceleration (elementwise)
    # if select_axis is None:
    # # --- Relative position (Euclidean distance) ---
    #     hand_distance = np.sqrt(
    #         (Lpos[:, 0] - Rpos[:, 0])**2 +
    #         (Lpos[:, 1] - Rpos[:, 1])**2 +
    #         (Lpos[:, 2] - Rpos[:, 2])**2
    #     )

    #     # --- Relative velocity ---
    #     L_vel = np.gradient(Lpos, 1.0 / frame_rate, axis=0)
    #     R_vel = np.gradient(Rpos, 1.0 / frame_rate, axis=0)

    #     rel_speed = np.sqrt(
    #         (L_vel[:, 0] - R_vel[:, 0])**2 +
    #         (L_vel[:, 1] - R_vel[:, 1])**2 +
    #         (L_vel[:, 2] - R_vel[:, 2])**2
    #     )

    #     # --- Relative acceleration ---
    #     L_acc = np.gradient(L_vel, 1.0 / frame_rate, axis=0)
    #     R_acc = np.gradient(R_vel, 1.0 / frame_rate, axis=0)

    #     rel_acc = np.sqrt(
    #         (L_acc[:, 0] - R_acc[:, 0])**2 +
    #         (L_acc[:, 1] - R_acc[:, 1])**2 +
    #         (L_acc[:, 2] - R_acc[:, 2])**2
    #     )
    # else:
        
    #     hand_distance = np.abs(Lpos - Rpos)
    #     L_vel = np.gradient(Lpos, 1.0/frame_rate)
    #     R_vel = np.gradient(Rpos, 1.0/frame_rate)
    #     rel_speed = np.abs(L_vel - R_vel)
    #     L_acc = np.gradient(L_vel, 1.0/frame_rate)
    #     R_acc = np.gradient(R_vel, 1.0/frame_rate)
    #     rel_acc = np.abs(L_acc - R_acc)
    
    # Load joint position data for both wrists
    dir_csv = "extracted_mocap_csv"
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    worldpos_file = os.path.join(dir_csv, f"{base_name}_T_worldpos.csv")
    
    try:
        world_positions = pd.read_csv(worldpos_file)
        print(f"Successfully loaded CSV with {len(world_positions)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    # Get time column and position data for both wrists
    time_column = world_positions.columns[0]  # First column is time
    times = world_positions[time_column].values
    
    # Get positions for both wrists along the chosen axis
    L_pos_x = world_positions[f"LeftWrist.X"].values
    R_pos_x = world_positions[f"RightWrist.X"].values
    
    L_pos_y = world_positions[f"LeftWrist.Y"].values
    R_pos_y = world_positions[f"RightWrist.Y"].values
    
    L_pos_z = world_positions[f"LeftWrist.Z"].values
    R_pos_z = world_positions[f"RightWrist.Z"].values
    
    
    # Calculate velocities
    dt = 1.0 / frame_rate
    L_vel_x = np.gradient(L_pos_x, dt)
    R_vel_x = np.gradient(R_pos_x, dt)
    
    L_vel_y = np.gradient(L_pos_y, dt)
    R_vel_y = np.gradient(R_pos_y, dt)
    
    L_vel_z = np.gradient(L_pos_z, dt)
    R_vel_z = np.gradient(R_pos_z, dt)

    rel_speed = np.sqrt(
        (L_vel_x - R_vel_x)**2 +
        (L_vel_y - R_vel_y)**2 +
        (L_vel_z - R_vel_z)**2
    )
    
    # Calculate accelerations
    L_acc_x = np.gradient(L_vel_x, dt)
    R_acc_x = np.gradient(R_vel_x, dt)
    
    L_acc_y = np.gradient(L_vel_y, dt)
    R_acc_y = np.gradient(R_vel_y, dt)
    
    L_acc_z = np.gradient(L_vel_z, dt)
    R_acc_z = np.gradient(R_vel_z, dt)
    
    rel_acc = np.sqrt(
        (L_acc_x - R_acc_x)**2 +
        (L_acc_y - R_acc_y)**2 +
        (L_acc_z - R_acc_z)**2
    )
    
    
    hand_distance = np.sqrt(
        (L_pos_x - R_pos_x)**2 +
        (L_pos_y - R_pos_y)**2 +
        (L_pos_z - R_pos_z)**2
    )
        
    # Find onsets of hand contacts
    contacts = detect_hand_contacts_from_distance(
                    hand_distance,
                    frame_rate=frame_rate,
                    min_gap_sec=min_gap_sec,
                    min_distance_thresh=min_thresh*100,
                    slope_min= slope_min
                )
    
    print(f"Detected {len(contacts)} hand contacts")
    # refined_contacts = contacts
    
    # Choose  minima around the contacts using window from rel_speed or rel_acc
    refined_contacts = []
    refined_window = 0.1
    win = int(refined_window * frame_rate)   # ± ms window

    file_parts = file_name.split("_")
    ensmble = file_parts[1]
    
    if ensmble == "E1" or ensmble == "E2":
        acc_threshold = 3000
    
    if ensmble == "E3":
        acc_threshold = 1500   


    for c in contacts:
        # for rel_acc
        start = max(0, c - win)
        end = min(len(rel_acc), c + int(0.03* frame_rate))
        seg = rel_acc[start:end]
        
        # find local maxima in the window
        peaks, _ = find_peaks(seg, prominence=0.1* np.max(seg))
        if len(peaks) == 0:
            continue

        # convert to absolute indices
        abs_peaks = start + peaks
        peak_vals = seg[peaks]
        
        # keep only peaks above threshold
        mask = peak_vals > acc_threshold
        peak_vals = peak_vals[mask]
        abs_peaks = abs_peaks[mask]
        
        # sort peaks by magnitude (descending)
        order = np.argsort(peak_vals)[::-1]     # order = np.argsort(peak_vals)[::-1]
        abs_peaks = abs_peaks[order]
        peak_vals = peak_vals[order]
        
        # ---- NEW LOGIC ----
        if len(peak_vals) == 0:
            continue  # or skip this window safely

        elif len(peak_vals) == 1:
            local_max = abs_peaks[0]

        else:
            # identify earlier and later in time
            if abs_peaks[0] <= abs_peaks[1]:
                earlier_idx, later_idx = 0, 1
            else:
                earlier_idx, later_idx = 1, 0

            p_earlier = peak_vals[earlier_idx]
            p_later = peak_vals[later_idx]

            # symmetric relative difference
            rel_diff = abs(p_earlier - p_later) / max(p_earlier, p_later)

            # Rule 1: earlier peak stronger → choose earlier
            if p_earlier >= p_later:
                local_max = abs_peaks[earlier_idx]

            # Rule 2: later stronger, but earlier within 30% → choose earlier
            elif rel_diff < 0.30:
                local_max = abs_peaks[earlier_idx]

            # Rule 3: otherwise choose strongest
            else:
                # local_max = abs_peaks[later_idx]
                if p_earlier > p_later:
                    local_max = abs_peaks[earlier_idx]
                else:
                    local_max = abs_peaks[later_idx]
# 
        refined_contacts.append(local_max)
        
    print(f"Detected {len(refined_contacts)} refined hand contacts")
    
    ########## Save hand contacts as pickle
    contacts_path = os.path.join("data/hand_contacts", f"{file_name}_contacts_{round(time_segments[0][0])}_{round(time_segments[0][1])}.pkl")
    os.makedirs(os.path.dirname(contacts_path), exist_ok=True)
    with open(contacts_path, 'wb') as f:
        pickle.dump(refined_contacts, f)

    

    # Interpolators (used for averaging)
    # dist_interp_full = interp1d(times, hand_distance, bounds_error=False, fill_value="extrapolate")
    # speed_interp_full = interp1d(times, rel_speed, bounds_error=False, fill_value="extrapolate")
    # acc_interp_full = interp1d(times, rel_acc, bounds_error=False, fill_value="extrapolate")

    total_start = min(seg[0] for seg in time_segments)
    total_end = max(seg[1] for seg in time_segments)
    t_range = total_end - total_start

    # Average cycle duration
    all_onsets = []
    cyc_df = pd.read_csv(cycles_csv_path)
    for seg_start, seg_end in time_segments:
        seg_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not seg_df.empty:
            all_onsets.extend(seg_df["Virtual Onset"].values[:-1])

    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")

    all_onsets = np.sort(all_onsets)
    durations = np.diff(sorted(all_onsets))
    avg_cycle = durations.mean()

    fig, (ax_dist, ax_vel, ax_acc) = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    cmap = plt.get_cmap('cool')

    def get_subdiv_color(subdiv):
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        group = ((subdiv - 1) % 3) + 1
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'

    all_dist_traj, all_speed_traj, all_acc_traj, all_times = [], [], [], []

    for seg_start, seg_end in time_segments:
        win_mask = (times >= seg_start) & (times <= seg_end)
        t_win = times[win_mask]
        dist_win = hand_distance[win_mask]
        speed_win = rel_speed[win_mask]
        acc_win = rel_acc[win_mask]

        seg_cycles = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        onsets = seg_cycles["Virtual Onset"].values[:-1]

        for i, c in enumerate(onsets):
            cycle_start = c
            cycle_end = onsets[i + 1] if i < len(onsets) - 1 else c + avg_cycle
            cycle_duration = cycle_end - cycle_start
            beat_len = cycle_duration / n_beats_per_cycle

            m = (t_win >= cycle_start) & (t_win <= cycle_end)
            tr = (t_win[m] - cycle_start) / beat_len  # normalized to beats (0-4)

            
            # Convert contact frame indices to time values
            contact_times = np.array(contacts) / frame_rate
            refined_contacts_times = np.array(refined_contacts) / frame_rate
            
            # Find contacts within this cycle
            cycle_contacts = contact_times[(contact_times >= cycle_start) & (contact_times <= cycle_end)]
            cycle_refined_contacts = refined_contacts_times[(refined_contacts_times >= cycle_start) & (refined_contacts_times <= cycle_end)]
            
            # if len(cycle_contacts) != len(cycle_refined_contacts):
            #     print(f"Warning: Mismatch in contact counts for cycle starting at {cycle_start:.2f}s")
            
            if len(cycle_contacts) > 0:
                # Convert contact times to normalized beat positions
                contact_tr = (cycle_contacts - cycle_start) / beat_len
                refined_contact_tr = (cycle_refined_contacts - cycle_start) / beat_len
                
                # Get corresponding distance values for markers
                contact_dist_values = []
                contact_speed_values = []
                contact_acc_values = []
                
                # for ct in cycle_contacts:
                #     # Find closest time point in t_win[m]
                #     closest_idx = np.argmin(np.abs(t_win[m] - ct))
                #     contact_dist_values.append(dist_win[m][closest_idx])
                    
                    # closest_ridx = np.argmin(np.abs(t_win[m] - rct))
                    # contact_speed_values.append(speed_win[m][closest_ridx])
                
                for rct in cycle_refined_contacts:
                    closest_ridx = np.argmin(np.abs(t_win[m] - rct))
                    
                    contact_dist_values.append(dist_win[m][closest_ridx])
                    
                    contact_speed_values.append(speed_win[m][closest_ridx])
                    
                    contact_acc_values.append(acc_win[m][closest_ridx])
            
            has_contact = len(cycle_contacts) > 0
            ls = '-' if has_contact else '--'
            col = cmap((c - total_start) / t_range)
            new_col = col if has_contact else 'gray'
            
            if show_trajectories:
                ax_dist.plot(tr, dist_win[m], ls, color=new_col, alpha=0.35,
                             label="Distance (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")
                ax_vel.plot(tr, speed_win[m], ls, color=new_col, alpha=0.35,
                            label="Rel speed (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")
                ax_acc.plot(tr, acc_win[m], ls, color=new_col, alpha=0.35,
                            label="Rel accel (cycle)" if (i == 0 and seg_start == time_segments[0][0]) else "")

                
                # Plot hand contact markers on distance plot
                
                # Plot contact markers
                if has_contact:
                    ax_dist.plot(refined_contact_tr, contact_dist_values, 'o', 
                                color=col, ms=4, alpha=0.8, markeredgecolor='k', 
                                markeredgewidth=1.5, zorder=5,
                                label="Hand contacts" if (i == 0 and seg_start == time_segments[0][0]) else "")
                    
                    ax_vel.plot(refined_contact_tr, contact_speed_values, 'o', 
                                color=col, ms=4, alpha=0.8, markeredgecolor='k', 
                                markeredgewidth=1.5, zorder=5,
                                label="Hand contacts" if (i == 0 and seg_start == time_segments[0][0]) else "")
                    
                    ax_acc.plot(refined_contact_tr, contact_acc_values, 'o', 
                                color=col, ms=4, alpha=0.8, markeredgecolor='k', 
                                markeredgewidth=1.5, zorder=5,
                                label="Hand contacts" if (i == 0 and seg_start == time_segments[0][0]) else "")


                all_dist_traj.append(dist_win[m])
                all_speed_traj.append(speed_win[m])
                all_acc_traj.append(acc_win[m])
                all_times.append(tr)

    # Grand averages
    if all_dist_traj and all_speed_traj and all_acc_traj:
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        dist_avg = np.zeros_like(common_times)
        speed_avg = np.zeros_like(common_times)
        acc_avg = np.zeros_like(common_times)
        count = 0
        for d_traj, s_traj, a_traj, t_traj in zip(all_dist_traj, all_speed_traj, all_acc_traj, all_times):
            if len(t_traj) > 1:
                d_interp = interp1d(t_traj, d_traj, bounds_error=False, fill_value="extrapolate")
                s_interp = interp1d(t_traj, s_traj, bounds_error=False, fill_value="extrapolate")
                a_interp = interp1d(t_traj, a_traj, bounds_error=False, fill_value="extrapolate")
                dist_avg += d_interp(common_times)
                speed_avg += s_interp(common_times)
                acc_avg += a_interp(common_times)
                count += 1
        if count > 0:
            dist_avg /= count
            speed_avg /= count
            acc_avg /= count
            ax_dist.plot(common_times, dist_avg, color='blue', linewidth=3, label='Distance average')
            ax_vel.plot(common_times, speed_avg, color='red', linewidth=3, label='Rel speed average')
            ax_acc.plot(common_times, acc_avg, color='purple', linewidth=3, label='Rel accel average')

    # Vertical lines (beats + subdivisions)
    for ax in (ax_dist, ax_vel, ax_acc):
        ax.axvline(0, color='black', linewidth=2, alpha=0.8)
        for beat in range(1, n_beats_per_cycle + 1):
            ax.axvline(beat, color='black', linewidth=2, alpha=0.8)
            for subdiv in range(1, n_subdiv_per_beat):
                subdiv_pos = beat - 1 + subdiv / n_subdiv_per_beat
                grid_color = get_subdiv_color((beat - 1) * n_subdiv_per_beat + subdiv + 1)
                ax.axvline(subdiv_pos, color=grid_color, alpha=0.8, linewidth=1.5)

    ax_acc.set_xlabel("Cycle Span (beats)")
    ax_dist.set_ylabel("Hand distance")
    ax_vel.set_ylabel("‖V_left - V_right‖")
    ax_acc.set_ylabel("‖A_left - A_right‖")

    xticks = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0]
    ax_vel.set_xticks(xticks)
    ax_vel.set_xticklabels([1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0])
    ax_vel.set_xlim(0.0, 4.0)

    ax_dist.set_title(f"Hand Distance per Cycle with Grand Average\n{file_name} |{time_segments[0]} | {mode}")
    ax_vel.set_title(f"Relative Velocity Norm per Cycle\n{file_name} | {mode}")
    ax_acc.set_title(f"Relative Acceleration Norm per Cycle\n{file_name} | {mode}")

    custom = [
        Line2D([0], [0], color=cmap(0.5), lw=2, alpha=0.35),
        Line2D([0], [0], color=cmap(0.5), lw=2, linestyle='-', alpha=0.35),
        Line2D([0], [0], color=cmap(0.5), lw=2, linestyle='-', alpha=0.35),
        
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.5), 
               ms=8, markeredgecolor='k', markeredgewidth=1.5, alpha=0.8),  # Add this line
        
        Line2D([0], [0], color='blue', lw=3),
        Line2D([0], [0], color='red', lw=3),
        Line2D([0], [0], color='purple', lw=3),
        Line2D([0], [0], color='gray', lw=1.5, linestyle='--'),
        Line2D([0], [0], color='black', lw=1.5),
        Line2D([0], [0], color='green', lw=1.5),
        Line2D([0], [0], color='red', lw=1.5)
    ]
    labels = [
        "Distance (cycles)",
        "Rel speed (cycles)",
        "Rel accel (cycles)",
        "Hand contacts",
        "Distance average",
        "Rel speed average",
        "Rel accel average",
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)",
        "Subdivision 2 (2,5,8,11)",
        "Subdivision 3 (3,6,9,12)"
    ]
    fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), framealpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    plt.colorbar(sm, ax=[ax_dist, ax_vel, ax_acc], label='Time in recording (s)')

    # plt.tight_layout()
    return fig, (ax_dist, ax_vel, ax_acc)


import numpy as np
from scipy.signal import find_peaks

def detect_hand_contacts_from_distance(
    hand_distance,
    frame_rate,
    min_gap_sec=0.3,
    min_distance_thresh=0.19,
    prominence=0.01,
    slope_min=0.3
):
    """
    Detect hand-contact events using only inter-hand distance.

    Parameters
    ----------
    hand_distance : np.ndarray
        1D array of inter-hand distance over time
    frame_rate : float
        Sampling rate (frames per second)
    min_gap_sec : float, optional
        Minimum temporal gap between extrema (seconds)
    min_distance_thresh : float, optional
        Keep only minima where hand_distance <= this value
    prominence : float, optional
        Peak prominence for maxima/minima detection
    slope_min : float, optional
        Minimum descent slope (distance units per second)

    Returns
    -------
    contacts : list of int
        Frame indices of detected hand contacts
    """

    min_gap_samples = int(min_gap_sec * frame_rate)

    # --- find maxima ---
    max_idx, _ = find_peaks(
        hand_distance,
        distance=min_gap_samples,
        prominence=prominence
    )

    # --- find minima (invert signal) ---
    min_idx, _ = find_peaks(
        -hand_distance,
        distance=min_gap_samples,
        prominence=prominence,
        height=-min_distance_thresh
    )

    contacts = []

    for t_min in min_idx:
        prev_max = max_idx[max_idx < t_min]
        if len(prev_max) == 0:
            continue

        t_max = prev_max[-1]

        delta_d = hand_distance[t_max] - hand_distance[t_min]
        delta_t = (t_min - t_max) / frame_rate

        if delta_t <= 0:
            continue

        slope = delta_d / delta_t

        if slope >= slope_min:
            contacts.append(t_min)
            
            # --- New heuristic: find contact onset before minimum ---
            # T = 0.1  # 1 cm (adjust units)

            # min_val = hand_distance[t_min]

            # pre_segment = hand_distance[:t_min]
            # valid_idx = np.where(pre_segment >= min_val + T)[0]

            # if len(valid_idx) == 0:
            #     continue  # reject this minimum

            # t_min_new = valid_idx[-1]  # closest point BEFORE minimum
            # contacts.append(t_min_new)
            

    return contacts
