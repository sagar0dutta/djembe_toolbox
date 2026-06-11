import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.interpolate import interp1d


def plot_trajectories_by_beat(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/dance_onsets_v4_0.007_foot_jun3",
    output_dir: str = "output_trajectory_plot",
    frame_rate: float = 240,
    time_segments: list = None,  # List of (start, end) tuples
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 2,
    figsize: tuple = (12, 6),
    dpi: int = 200,
    use_cycles: bool = True,
    show_gray_plots: bool = True,
    show_trajectories: bool = True,  # New parameter to control trajectory lines
    show_vlines: bool = True        # New parameter to control vertical lines
):
    """
    Plot left- and right-foot Y-position trajectories ±window around each beat (1,2,3,4),
    marking foot-onset times for cycles that have an onset in the window.
    Optionally plots trajectories for cycles without onsets in gray.

    Parameters
    ----------
    [previous parameters remain the same]
    time_segments : list of tuples
        List of (start, end) time segments to plot. If None, uses default window.
    show_trajectories : bool
        Whether to show the continuous trajectory lines
    show_vlines : bool
        Whether to show vertical lines at onset times
    """
    # Use default window if no segments provided
    if time_segments is None:
        time_segments = [(0, 10)]
        print("No time segments provided, using default window (0, 10)")

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

    # Calculate average cycle duration from all segments
    all_onsets = []
    for seg_start, seg_end in time_segments:
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not cyc_df.empty:
            all_onsets.extend(cyc_df["Virtual Onset"].values[:-1])
    
    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")
    
    # Calculate average cycle duration
    durations = np.diff(sorted(all_onsets))
    avg_cycle = durations.mean()

    # Calculate beat and subdivision lengths
    beat_len = avg_cycle / n_beats_per_cycle
    subdiv_len = beat_len / n_subdiv_per_beat
    half_win = subdiv_len * nn

    # Create figure with subplots for each beat
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    cmap = plt.get_cmap('cool')

    # Define subdivision color mapping
    def get_subdiv_color(subdiv):
        # First, normalize the subdivision to be 1-based and within the valid range
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        

        group = ((subdiv - 1) % 3) + 1  # This gives us 1,2,3 for the three groups
        
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'  # fallback color

    # For each beat position (1,2,3,4)
    for beat_idx, ax in enumerate(axes):
        beat_offset = beat_idx * beat_len  # Time offset for this beat
        # To:
        # beat_offset = (beat_idx + 1) * beat_len
        
        
        # Process each time segment
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
            total_cycles = len(onsets) - 1
            
            # foot onsets
            left_df  = pd.read_csv(left_onsets_csv)
            right_df = pd.read_csv(right_onsets_csv)
            left_times  = left_df[ (left_df["time_sec"]>=seg_start)&(left_df["time_sec"]<=seg_end) ]["time_sec"].values
            right_times = right_df[(right_df["time_sec"]>=seg_start)&(right_df["time_sec"]<=seg_end)]["time_sec"].values

            # Plot gray trajectories if enabled
            if show_gray_plots and show_trajectories:
                for c in onsets:
                    beat_time = c + beat_offset  # Time of this beat in this cycle
                    m = (t_win>=beat_time-half_win)&(t_win<=beat_time+half_win)
                    tr = t_win[m] - beat_time
                    if use_cycles:
                        tr = tr / beat_len  # Changed from avg_cycle to beat_len
                    ax.plot(tr, L_win[m], '-', color='gray', alpha=0.3)
                    ax.plot(tr, R_win[m], '--', color='gray', alpha=0.3)

            # Collect cycles that have foot onsets near this beat
            cyc_L, L_near = [], {}
            cyc_R, R_near = [], {}
            
            for c in onsets:
                beat_time = c + beat_offset
                
                # Left foot
                hits = left_times[(left_times>=beat_time-half_win)&(left_times<=beat_time+half_win)]
                if len(hits):
                    cyc_L.append(c)
                    L_near[c] = hits
                
                
                # Right foot
                hits = right_times[(right_times>=beat_time-half_win)&(right_times<=beat_time+half_win)]
                if len(hits):
                    cyc_R.append(c)
                    R_near[c] = hits
                    
            # counter for total onsets in window:
            left_onsets_in_window = sum(len(hits) for hits in L_near.values())
            right_onsets_in_window = sum(len(hits) for hits in R_near.values())
            # total_onsets_in_window = left_onsets_in_window + right_onsets_in_window

            # Plot left foot trajectories with onsets
            for i, c in enumerate(cyc_L):
                col = cmap((c-total_start)/t_range)
                beat_time = c + beat_offset
                m = (t_win>=beat_time-half_win)&(t_win<=beat_time+half_win)
                tr = t_win[m] - beat_time
                if use_cycles:
                    tr = tr / beat_len  # Changed from avg_cycle to beat_len
                if show_trajectories:
                    ax.plot(tr, L_win[m], '-', color=col, alpha=0.3,
                            label="Left Foot" if i==0 else "")
                for lt in L_near[c]:
                    rel = lt - beat_time
                    if use_cycles:
                        rel = rel / beat_len  # Changed from avg_cycle to beat_len
                    if show_vlines:
                        ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
                    ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                            markerfacecolor='blue', alpha=0.8)

            # Plot right foot trajectories with onsets
            for i, c in enumerate(cyc_R):
                col = cmap((c-total_start)/t_range)
                beat_time = c + beat_offset
                m = (t_win>=beat_time-half_win)&(t_win<=beat_time+half_win)
                tr = t_win[m] - beat_time
                if use_cycles:
                    tr = tr / beat_len  # Changed from avg_cycle to beat_len
                if show_trajectories:
                    ax.plot(tr, R_win[m], '--', color=col, alpha=0.3,
                            label="Right Foot" if i==0 else "")
                for rt in R_near[c]:
                    rel = rt - beat_time
                    if use_cycles:
                        rel = rel / beat_len  # Changed from avg_cycle to beat_len
                    if show_vlines:
                        ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
                    ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                            color='red', alpha=0.8)

            # Add trajectories from beat 4 to beat 1's plot
            if beat_idx == 0:  # Only for beat 1
                # Get trajectories from beat 4 (beat_idx == 3)
                beat4_offset = 3 * beat_len  # Time offset for beat 4
                
                # Plot gray trajectories from beat 4 if enabled
                if show_gray_plots and show_trajectories:
                    for c in onsets:
                        beat_time = c + beat4_offset  # Time of beat 4 in this cycle
                        m = (t_win>=beat_time-half_win)&(t_win<=beat_time+half_win)
                        tr = t_win[m] - beat_time
                        if use_cycles:
                            tr = tr / beat_len
                        # Shift by one cycle to the left
                        tr = tr - 1  # Subtract one cycle
                        ax.plot(tr, L_win[m], '-', color='gray', alpha=0.3)
                        ax.plot(tr, R_win[m], '--', color='gray', alpha=0.3)

                # Collect cycles that have foot onsets near beat 4
                cyc_L4, L_near4 = [], {}
                cyc_R4, R_near4 = [], {}
                
                for c in onsets:
                    beat_time = c + beat4_offset
                    # Left foot
                    hits = left_times[(left_times>=beat_time-half_win)&(left_times<=beat_time+half_win)]
                    if len(hits):
                        cyc_L4.append(c)
                        L_near4[c] = hits
                    # Right foot
                    hits = right_times[(right_times>=beat_time-half_win)&(right_times<=beat_time+half_win)]
                    if len(hits):
                        cyc_R4.append(c)
                        R_near4[c] = hits

                # Plot left foot trajectories with onsets from beat 4
                for i, c in enumerate(cyc_L4):
                    col = cmap((c-total_start)/t_range)
                    beat_time = c + beat4_offset
                    m = (t_win>=beat_time-half_win)&(t_win<=beat_time+half_win)
                    tr = t_win[m] - beat_time
                    if use_cycles:
                        tr = tr / beat_len
                    # Shift by one cycle to the left
                    tr = tr - 1  # Subtract one cycle
                    if show_trajectories:
                        ax.plot(tr, L_win[m], '-', color=col, alpha=0.3,
                                label="Left Foot" if i==0 else "")
                    for lt in L_near4[c]:
                        rel = lt - beat_time
                        if use_cycles:
                            rel = rel / beat_len
                        # Shift by one cycle to the left
                        rel = rel - 1  # Subtract one cycle
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
                        ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)

                # Plot right foot trajectories with onsets from beat 4
                for i, c in enumerate(cyc_R4):
                    col = cmap((c-total_start)/t_range)
                    beat_time = c + beat4_offset
                    m = (t_win>=beat_time-half_win)&(t_win<=beat_time+half_win)
                    tr = t_win[m] - beat_time
                    if use_cycles:
                        tr = tr / beat_len
                    # Shift by one cycle to the left
                    tr = tr - 1  # Subtract one cycle
                    if show_trajectories:
                        ax.plot(tr, R_win[m], '--', color=col, alpha=0.3,
                                label="Right Foot" if i==0 else "")
                    for rt in R_near4[c]:
                        rel = rt - beat_time
                        if use_cycles:
                            rel = rel / beat_len
                        # Shift by one cycle to the left
                        rel = rel - 1  # Subtract one cycle
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
                        ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)

        # Add yellow glow effect at t=0
        ax.axvline(0, color='yellow', linewidth=6, alpha=0.3, zorder=0)  # Glow effect
        ax.axvline(0, color='black', linewidth=1.5, label="Beat (t=0)", zorder=1)  # Main line

        # Add grid lines for other subdivisions
        for j in range(-nn, nn+1):
            if j!=0:
                pos = j*subdiv_len
                if use_cycles:
                    pos = pos / beat_len  # Changed from avg_cycle to beat_len
                # Calculate the subdivision number relative to the current beat
                subdiv_num = beat_idx * n_subdiv_per_beat + j + 1
                # Normalize to be within the valid range
                subdiv_num = ((subdiv_num - 1) % (n_beats_per_cycle * n_subdiv_per_beat)) + 1
                grid_color = get_subdiv_color(subdiv_num)
                ax.axvline(pos, color=grid_color, alpha=0.8, linewidth=1.5)

        xlabel = "Beats relative to beat" if use_cycles else "Time relative to beat (s)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Foot Y Position")
        ax.set_title(f"Beat {beat_idx + 1} \n R:{right_onsets_in_window} ({right_onsets_in_window*100/total_cycles:.0f}%) |  L:{left_onsets_in_window} ({left_onsets_in_window*100/total_cycles:.0f}%) |  C:{total_cycles}")
        ax.grid(True, alpha=0.3)  
        
        if use_cycles:
            xlim = nn * subdiv_len / beat_len +0.01  # Convert time to beats
        else:
            xlim = nn * subdiv_len  # Keep in seconds

        ax.set_xlim(-xlim, xlim)


    # Add colorbar to the figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist())
    cbar.set_label('Time in recording (s)')

    custom = [
        # Main trajectory and onset markers
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],marker='x', color='red', ms=8),
        # Line2D([0],[0],color='blue', lw=3),
        # Line2D([0],[0],color='red', lw=3, linestyle='--'),
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
        # "Left Foot Average", 
        # "Right Foot Average", 
        # "Combined Average",
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)", 
        "Subdivision 2 (2,5,8,11)", 
        "Subdivision 3 (3,6,9,12)"
    ]
    fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.20, 0.5), framealpha=0.3)
    
    plt.suptitle(
        f"Foot Trajectories ±{nn/n_subdiv_per_beat:.2f} beats around each beat\n"
        f"{file_name} | segments: all | {mode} | {time_segments[0][0]:.1f} - {time_segments[0][1]:.1f}",
        fontsize=10
    )
    
    plt.subplots_adjust(top=0.85)
    fig.set_constrained_layout(True)
    
    #save figure
    save_dir = os.path.join(output_dir, "beatwise_plots", file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{file_name}_{time_segments[0][0]:.1f}_{time_segments[0][1]:.1f}_beatwise.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    return fig, axes