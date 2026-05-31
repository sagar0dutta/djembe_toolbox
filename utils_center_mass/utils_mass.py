import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from bvh_converter import bvh_mod
# from scipy.signal import savgol_filter

from matplotlib.lines import Line2D
from scipy.interpolate import interp1d




def plot_all_cycles_trajectories(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,
    time_segments: list = None,  # List of (start, end) tuples
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    traj_threshold=0.15,
    figsize: tuple = (12, 6),
    dpi: int = 200,
    show_trajectories: bool = True,  # Control trajectory lines
    show_vlines: bool = True,        # Control vertical lines
    show_gray_plots: bool = True     # Control gray trajectory plots
):
    """
    Plot all foot trajectories in a single plot with grand average.
    Shows beat and subdivision lines with colors.
    X-axis shows 1.5 cycles (e.g., 0-6 for 4-beat cycles).
    """

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
    included_cycles = []  # New list to store cycle start times
    excluded_cycles = []    # New list to store cycle end times

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
                    ax.plot(tr, L_win[m], '-', color='gray', alpha=0.9)
                    ax.plot(tr, R_win[m], '--', color='gray', alpha=0.9)
                    # Also plot shifted for 1.5 cycles
                    ax.plot(tr + 4, L_win[m], '-', color='gray', alpha=0.9)
                    ax.plot(tr + 4, R_win[m], '--', color='gray', alpha=0.9)
                
                # Plot colored trajectories for cycles with any onsets
                if has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                    ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                    # Also plot shifted for 1.5 cycles
                    ax.plot(tr + 4, L_win[m], '-', color=col, alpha=0.3)
                    ax.plot(tr + 4, R_win[m], '--', color=col, alpha=0.3)
                
                # Apply thresholding when storing trajectories for averaging
                traj_max = max(np.nanmax(L_win[m]), np.nanmax(R_win[m]))
                if traj_max >= traj_threshold:
                    all_L_trajectories.append(L_win[m])
                    all_R_trajectories.append(R_win[m])
                    all_times.append(tr)
                    # Store the cycle timing information for included trajectories
                    # This tuple (cycle_start, cycle_end) represents the time window of this cycle
                    included_cycles.append((cycle_start, cycle_end))

                else:
                    # Store the cycle timing information for excluded trajectories
                    # These cycles didn't meet the threshold criteria
                    excluded_cycles.append((cycle_start, cycle_end))

                # Plot markers for foot onsets
                for lt in left_times:
                    if cycle_start <= lt <= cycle_end:
                        rel = (lt - cycle_start) / beat_len
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
                        ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)
                        # Also plot shifted for 1.5 cycles
                        if show_vlines:
                            ax.axvline(rel + 4, color=col, linestyle='-', alpha=0.5)
                        ax.plot(rel + 4, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)

                for rt in right_times:
                    if cycle_start <= rt <= cycle_end:
                        rel = (rt - cycle_start) / beat_len
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
                        ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)
                        # Also plot shifted for 1.5 cycles
                        if show_vlines:
                            ax.axvline(rel + 4, color=col, linestyle='--', alpha=0.5)
                        ax.plot(rel + 4, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)
    
    print("count of included trajectories: ", len(included_cycles))
    print("count of excluded trajectories: ", len(excluded_cycles))
    print(included_cycles)
    
    pickle_dir = "traj_files"
    # Save the dictionaries
    with open(os.path.join(pickle_dir, f'{file_name}_included_{traj_threshold}.pkl'), 'wb') as f:
        pickle.dump(included_cycles, f)
    
    with open(os.path.join(pickle_dir, f'{file_name}_excluded_{traj_threshold}.pkl'), 'wb') as f:
        pickle.dump(excluded_cycles, f)
    
    # Calculate and plot grand average
    if all_L_trajectories and all_R_trajectories:
        # Interpolate all trajectories to the same time points
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        L_avg = np.zeros(len(common_times))
        R_avg = np.zeros(len(common_times))
        count = 0
        
        # No need to check threshold again since trajectories are already filtered
        for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
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
            # Also plot shifted for 1.5 cycles
            ax.plot(common_times + 4, L_avg, '-', color='blue', linewidth=3)
            ax.plot(common_times + 4, R_avg, '--', color='red', linewidth=3)

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

    ax.set_xlabel("Cycle Span")
    ax.set_ylabel("Foot Y Position")
    ax.set_title(f"All Cycles Trajectories with Grand Average\n{file_name} | {mode}")

    # Add all legends together outside the plot
    custom = [
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],marker='x', color='red', ms=8),
        Line2D([0],[0],color='blue', lw=3),
        Line2D([0],[0],color='red', lw=3, linestyle='--'),
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


########################################

# helper to extract all (start, end) tuples for a mode
def get_segments(df, name):
    if df.empty:
        print(f"⚠️  No rows for mode '{name}', skipping.")
        return None
    return [(row["Start (in sec)"], row["End (in sec)"]) for _, row in df.iterrows()]


def get_tsegment_for(mode_name, mode_value, suffix):
    """
    Run get_segments on one mode, and return a one-entry dict
    iff it isn’t None.
    """
    seg = get_segments(mode_value, suffix)
    return mode_name, seg if seg is not None else {}