import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils_subdivision.gen_distribution_single_plots import find_cycle_phases, kde_estimate

def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'

def plot_foot_onsets_stacked(file_name,
                             dance_mode,
                             cycles_csv_path,
                             left_onsets,
                             right_onsets,
                             dance_mode_time_segments,
                             figsize=(10, 3),
                             dpi=200,
                             use_window=True,
                             legend_flag=True):
    """Plot left and right foot onsets with stacked scatter and combined KDE, using robust phase and KDE calculation."""
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cycles = pd.read_csv(cycles_csv_path)["Virtual Onset"].values

    vertical_ranges = {
        'left': (1, 6),
        'right': (8, 13),                    # 'right': (1, 6),       # 'right': (8, 13)
    }

    combined_phases = []
    dance_phases_kde = {"phases": {}, "y_scaled": {}, "kde_h": {}, "kde_xx": {}, "window_positions": {}, }
    
    for foot_type, onsets, color in [('left', left_onsets, '#1f77b4'), ('right', right_onsets, '#d62728')]:
        # Filter onsets by time segments
        if use_window:
            window_mask = np.zeros(len(onsets), dtype=bool)
            for W_start, W_end in dance_mode_time_segments:
                segment_mask = (onsets >= W_start) & (onsets <= W_end)
                window_mask |= segment_mask
            filtered_onsets = onsets[window_mask]
        else:
            filtered_onsets = onsets

        if len(filtered_onsets) == 0:
            continue

        # Use robust phase calculation
        cycle_indices, phases, valid_onsets = find_cycle_phases(filtered_onsets, cycles)
        if len(phases) == 0:
            continue

        # Collect for combined KDE
        combined_phases.extend(phases)

        # Calculate window positions --------------------------------------------------------
        window_positions = []
        if use_window:
            for onset in valid_onsets:
                for seg_idx, (W_start, W_end) in enumerate(dance_mode_time_segments):
                    if W_start <= onset <= W_end:
                        segment_duration = W_end - W_start
                        relative_pos = (onset - W_start) / segment_duration
                        # Normalize to 0-1 range by dividing by total number of segments
                        window_pos = relative_pos / len(dance_mode_time_segments)
                        window_positions.append(window_pos)
                        break
        else:
            window_positions = np.zeros_like(valid_onsets)

        window_positions = np.array(window_positions)
        
        # Scale to vertical range
        y0, y1 = vertical_ranges[foot_type]
        y_scaled = y0 + (window_positions * (y1 - y0))
        

        # Plot scatter
        ax.scatter(phases * 400, y_scaled, s=5, alpha=0.6, color=color, label=f'{foot_type.capitalize()} Foot')

        # Collect for combined KDE
        dance_phases_kde["phases"][foot_type] = phases
        dance_phases_kde["window_positions"][foot_type] = window_positions
        dance_phases_kde["y_scaled"][foot_type] = y_scaled
    
        

    # Combined KDE at bottom using kde_estimate ----------------------------------------------
    if len(combined_phases) > 0:
        
        kde_xx, kde_h = kde_estimate(np.array(combined_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

        # Collect for combined KDE
        dance_phases_kde["kde_h"] = kde_h_plot
        dance_phases_kde["kde_xx"] = kde_xx_plot
    
    
    # Subdivision lines --------------------------------------------------------------------
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling ------------------------------------------------------------------------------
    # xtick = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367, 400]
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 13.5)
    ax.set_yticks([3, 10])
    ax.set_yticklabels(['LF', 'RF'])
    ax.set_ylabel('Foot')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'File: {file_name} | Dance Mode: {dance_mode}'
    title += f' | Segments: {len(dance_mode_time_segments)}' if use_window else ' | Full Recording'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax, dance_phases_kde


def plot_combined_foot_stacked(piece_type, 
                               dance_mode, 
                               dance_phases_kde_all, 
                               figsize=(10, 3), 
                               dpi=200, 
                               legend_flag=True):
    """Create a single plot showing combined foot analysis for all pieces of a type"""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    vertical_ranges = {
        'left': (1, 6),
        'right': (8, 13),
    }

    combined_phases = []
    
    # Combine data for each foot
    for foot_type, color in [('left', '#1f77b4'), ('right', '#d62728')]:
        # Combine phases and y_scaled from all pieces
        all_phases = []
        all_y_scaled = []
        segment_kde_h = None
        segment_kde_xx = None

        # Loop through all pieces' data
        for piece_data in dance_phases_kde_all:
            if foot_type in piece_data["phases"]:
                # Combine phases and y_scaled
                all_phases.extend(piece_data["phases"][foot_type])
                all_y_scaled.extend(piece_data["y_scaled"][foot_type])
                combined_phases.extend(piece_data["phases"][foot_type])

        if not all_phases:  # Skip if no data found
            continue

        # Convert lists to numpy arrays
        phases = np.array(all_phases)
        y_scaled = np.array(all_y_scaled)
        
        # Normalize y_scaled values to fit within the vertical range
        y_min, y_max = vertical_ranges[foot_type]
        y_scaled = y_min + (y_scaled - np.min(y_scaled)) * (y_max - y_min) / (np.max(y_scaled) - np.min(y_scaled))
            
        # Plot scatter with single color for each foot
        ax.scatter(phases * 400,
                   y_scaled,
                   s=5, alpha=0.6,
                   color=color,
                   label=f'{foot_type.capitalize()} Foot')

    # Combined KDE at bottom using kde_estimate
    if len(combined_phases) > 0:
        kde_xx, kde_h = kde_estimate(np.array(combined_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

    # Subdivision lines
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 13.5)
    ax.set_yticks([3, 10])
    ax.set_yticklabels(['LF', 'RF'])
    ax.set_ylabel('Foot')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'Piece: {piece_type} | Dance Mode: {dance_mode}'
    title += f' | Combined from {len(dance_phases_kde_all)} pieces'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax



def plot_foot_onsets_stacked_all_modes(
    file_name,
    cycles_csv_path,
    left_onset_path,
    right_onset_path,
    dance_mode_time_segments_all,  # Dictionary containing time segments for all modes
    figsize=(10, 3),
    dpi=200,
    use_window=True,
    legend_flag=True
):
    """Create a single plot showing combined foot onset analysis for all modes for a single file."""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    vertical_ranges = {
        'left': (1, 6),
        'right': (8, 13)
    }

    # Fixed colors for feet
    foot_colors = {
        'left': '#1f77b4',   # blue
        'right': '#d62728'   # red
    }

    # Initialize combined data structure
    foot_types = ['left', 'right']
    foot_phases_kde = {foot_type: {"phases": [], "y_scaled": []} for foot_type in foot_types}
    
    # Load foot onset data
    left_onsets = pd.read_csv(left_onset_path)["time_sec"].values
    right_onsets = pd.read_csv(right_onset_path)["time_sec"].values
    
    # Process each mode
    for mode, dance_mode_time_segments in dance_mode_time_segments_all.items():
        # Get cycles data
        cycles_df = pd.read_csv(cycles_csv_path)
        cycle_times = cycles_df['Virtual Onset'].values
        
        # Process each foot type
        for foot_type, color in foot_colors.items():
            # Get onset times for this foot
            onset_times = left_onsets if foot_type == 'left' else right_onsets
            
            # Process each time segment
            for segment in dance_mode_time_segments:
                start_time, end_time = segment  # Unpack the tuple
                
                # Get cycles in this segment
                segment_cycles = cycle_times[
                    (cycle_times >= start_time) & 
                    (cycle_times <= end_time)
                ]
                
                if len(segment_cycles) == 0:
                    continue
                
                # Get onsets in this segment
                segment_onsets = onset_times[
                    (onset_times >= start_time) & 
                    (onset_times <= end_time)
                ]
                
                if len(segment_onsets) == 0:
                    continue
                
                # Calculate phases for each onset
                for onset_time in segment_onsets:
                    # Find the cycle containing this onset
                    cycle_mask = (cycle_times <= onset_time)
                    if not any(cycle_mask):
                        continue
                    
                    cycle_start = cycle_times[cycle_mask][-1]  # Last cycle before onset
                    
                    # Find the next cycle
                    next_cycle_mask = (cycle_times > onset_time)
                    if any(next_cycle_mask):
                        cycle_end = cycle_times[next_cycle_mask][0]
                    else:
                        cycle_end = cycle_times[-1]
                    
                    # Skip if cycle_start equals cycle_end
                    if cycle_end == cycle_start:
                        continue
                    
                    # Calculate phase
                    phase = (onset_time - cycle_start) / (cycle_end - cycle_start)
                    
                    # Handle cycle wrapping
                    # if phase > 0.5:  # If phase is closer to next cycle
                    #     phase = phase - 1.0
                    
                    # Add to combined data
                    foot_phases_kde[foot_type]["phases"].append(phase)
                    foot_phases_kde[foot_type]["y_scaled"].append(len(foot_phases_kde[foot_type]["phases"]))

    # Plot the combined data
    for foot_type, color in foot_colors.items():
        phases = np.array(foot_phases_kde[foot_type]["phases"])
        y_scaled = np.array(foot_phases_kde[foot_type]["y_scaled"])
        
        if len(phases) == 0:
            continue
            
        # Normalize y_scaled values to fit within the vertical range
        y_min, y_max = vertical_ranges[foot_type]
        y_scaled = y_min + (y_scaled - np.min(y_scaled)) * (y_max - y_min) / (np.max(y_scaled) - np.min(y_scaled))
    
        # Plot scatter
        ax.scatter(phases * 400,
                  y_scaled,
                  s=5, alpha=0.4,
                  color=color,
                  label=f'{foot_type.capitalize()} Foot')

    # Calculate and plot combined KDE
    all_phases = []
    for foot_type in foot_types:
        all_phases.extend(foot_phases_kde[foot_type]["phases"])
    
    if len(all_phases) > 0:
        kde_xx, kde_h = kde_estimate(np.array(all_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

    # Subdivision lines
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 13.5)
    ax.set_yticks([3, 10])
    ax.set_yticklabels(['LF', 'RF'])
    ax.set_ylabel('Foot')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'File: {file_name} | All Modes Combined'
    # title += f' | Combined from {len(dance_mode_time_segments_all)} modes'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax, foot_phases_kde


def plot_foot_onsets_stacked_by_piece_type(
    piece_type,
    piece_dance_phases_kde,  # Dictionary containing all modes' data
    figsize=(10, 3),
    dpi=200,
    legend_flag=True
):
    """Create a single plot showing combined foot analysis for all pieces and all modes."""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    vertical_ranges = {
        'left': (1, 6),
        'right': (8, 13),
    }

    # Fixed colors for feet
    foot_colors = {
        'left': '#1f77b4',   # blue
        'right': '#d62728'   # red
    }

    combined_phases = []
    
    # Process each mode
    for mode in piece_dance_phases_kde.keys():
        if piece_type not in piece_dance_phases_kde[mode]:
            continue
            
        # Get the list of data for this piece type in this mode
        piece_data_list = piece_dance_phases_kde[mode][piece_type]
        
        # Combine data for each foot
        for foot_type, color in foot_colors.items():
            # Combine phases and y_scaled from all pieces
            all_phases = []
            all_y_scaled = []

            # Loop through all pieces' data
            for piece_data in piece_data_list:
                if foot_type in piece_data["phases"]:
                    # Combine phases and y_scaled
                    all_phases.extend(piece_data["phases"][foot_type])
                    all_y_scaled.extend(piece_data["y_scaled"][foot_type])
                    combined_phases.extend(piece_data["phases"][foot_type])

            if not all_phases:  # Skip if no data found
                continue

            # Convert lists to numpy arrays
            phases = np.array(all_phases)
            y_scaled = np.array(all_y_scaled)
            
            # Normalize y_scaled values to fit within the vertical range
            y_min, y_max = vertical_ranges[foot_type]
            y_scaled = y_min + (y_scaled - np.min(y_scaled)) * (y_max - y_min) / (np.max(y_scaled) - np.min(y_scaled))
                
            # Plot scatter with foot-specific color
            ax.scatter(phases * 400,
                      y_scaled,
                      s=5, alpha=0.4,  # Reduced alpha for better visibility of overlapping points
                      color=color,
                      label=f'{foot_type.capitalize()} Foot')

    # Rest of the function remains the same...
    # Combined KDE at bottom using kde_estimate
    if len(combined_phases) > 0:
        kde_xx, kde_h = kde_estimate(np.array(combined_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

    # Subdivision lines
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 13.5)
    ax.set_yticks([3, 10])
    ax.set_yticklabels(['LF', 'RF'])
    ax.set_ylabel('Foot')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'Piece: {piece_type} | All Modes Combined'
    # title += f' | Combined from {len(piece_dance_phases_kde)} modes'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax


def plot_hand_onsets_kde(file_name,
                             dance_mode,
                             cycles_csv_path,
                             hand_onsets,
                             dance_mode_time_segments,
                             figsize=(10, 3),
                             dpi=200,
                             use_window=True,
                             legend_flag=True):
    """Plot left and right foot onsets with stacked scatter and combined KDE, using robust phase and KDE calculation."""
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cycles = pd.read_csv(cycles_csv_path)["Virtual Onset"].values

    vertical_ranges = {
        'hand': (1, 6),
        # 'right': (8, 13),                    # 'right': (1, 6),       # 'right': (8, 13)
    }

    combined_phases = []
    dance_phases_kde = {"phases": {}, "y_scaled": {}, "kde_h": {}, "kde_xx": {} }
    
    for foot_type, onsets, color in [('hand', hand_onsets, '#1f77b4')]:
        # Filter onsets by time segments
        if use_window:
            window_mask = np.zeros(len(onsets), dtype=bool)
            for W_start, W_end in dance_mode_time_segments:
                segment_mask = (onsets >= W_start) & (onsets <= W_end)
                window_mask |= segment_mask
            filtered_onsets = onsets[window_mask]
        else:
            filtered_onsets = onsets

        if len(filtered_onsets) == 0:
            continue

        # Use robust phase calculation
        cycle_indices, phases, valid_onsets = find_cycle_phases(filtered_onsets, cycles)
        if len(phases) == 0:
            continue

        # Collect for combined KDE
        combined_phases.extend(phases)

        # Calculate window positions --------------------------------------------------------
        window_positions = []
        if use_window:
            for onset in valid_onsets:
                for seg_idx, (W_start, W_end) in enumerate(dance_mode_time_segments):
                    if W_start <= onset <= W_end:
                        segment_duration = W_end - W_start
                        relative_pos = (onset - W_start) / segment_duration
                        # Normalize to 0-1 range by dividing by total number of segments
                        window_pos = relative_pos / len(dance_mode_time_segments)
                        window_positions.append(window_pos)
                        break
        else:
            window_positions = np.zeros_like(valid_onsets)

        window_positions = np.array(window_positions)
        
        # Scale to vertical range
        y0, y1 = vertical_ranges[foot_type]
        y_scaled = y0 + (window_positions * (y1 - y0))
        

        # Plot scatter
        ax.scatter(phases * 400, y_scaled, s=5, alpha=0.6, color=color, label=f'{foot_type.capitalize()} Hand')

        # Collect for combined KDE
        dance_phases_kde["phases"][foot_type] = phases
        dance_phases_kde["y_scaled"][foot_type] = y_scaled
    
        

    # Combined KDE at bottom using kde_estimate ----------------------------------------------
    if len(combined_phases) > 0:
        
        kde_xx, kde_h = kde_estimate(np.array(combined_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

        # Collect for combined KDE
        dance_phases_kde["kde_h"] = kde_h_plot
        dance_phases_kde["kde_xx"] = kde_xx_plot
    
    
    # Subdivision lines --------------------------------------------------------------------
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling ------------------------------------------------------------------------------
    # xtick = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367, 400]
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 6.5)
    ax.set_yticks([])
    # ax.set_yticklabels(['Onsets'])
    ax.set_ylabel('Hand Clap')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'File: {file_name} | Dance Mode: {dance_mode}'
    # title += f' | Segments: {len(dance_mode_time_segments)}' if use_window else ' | Full Recording'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax, dance_phases_kde




def plot_hand_onsets_kde_piece(
        file_name,
        dance_mode,
        phases,
        window_positions,
        figsize=(10, 3),
        dpi=200,
        legend_flag=True,
        debug=True):
    """
    Plot hand onset phases and KDE computed from the SAME phases.
    This matches plot_combined_foot_stacked() logic.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if debug:
        print("\n========== PLOT HAND ONSETS DEBUG ==========")
        print(f"File: {file_name} | Mode: {dance_mode}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # -----------------------------
    # Sanity + reshape
    # -----------------------------
    phases = np.atleast_1d(phases)
    window_positions = np.atleast_1d(window_positions)

    if debug:
        print(f"[DEBUG] phases shape: {phases.shape}")
        print(f"[DEBUG] window_positions shape: {window_positions.shape}")
        if phases.size:
            print(f"[DEBUG] phase min/max: {phases.min():.3f} / {phases.max():.3f}")

    # -----------------------------
    # Scatter (hand onsets)
    # -----------------------------
    if phases.size > 0:
        y0, y1 = 1, 6
        y_scaled = y0 + window_positions * (y1 - y0)

        ax.scatter(
            phases * 400,
            y_scaled,
            s=5,
            alpha=0.6,
            color='#1f77b4',
            label='Hand claps'
        )
    else:
        if debug:
            print("[DEBUG] No phases → skipping scatter")

    # -----------------------------
    # KDE from SAME phases (KEY FIX)
    # -----------------------------
    if phases.size > 1:
        kde_xx, kde_h = kde_estimate(phases, SIG=0.01)

        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]

        if debug:
            print(f"[DEBUG] KDE size: {kde_h_plot.size}")
            print(f"[DEBUG] KDE max: {np.max(kde_h_plot):.6f}")

        if kde_h_plot.size > 0 and np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))

            ax.fill_between(
                kde_xx_plot * 400,
                -5,
                kde_scaled,
                alpha=0.3,
                color='purple',
                label='Phase KDE'
            )
        else:
            if debug:
                print("[DEBUG] KDE empty after masking")
    else:
        if debug:
            print("[DEBUG] Not enough phases for KDE")

    # -----------------------------
    # Subdivision lines
    # -----------------------------
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv - 1) * 400) / 12

        ax.vlines(
            x_pos,
            -5.5,
            20.5,
            color=color,
            linestyle='-' if subdiv in [1, 4, 7, 10] else '--',
            linewidth=1.5 if subdiv in [1, 4, 7, 10] else 1,
            alpha=0.7 if subdiv in [1, 4, 7, 10] else 0.3
        )

    # -----------------------------
    # Styling
    # -----------------------------
    ax.set_xlim(-33, 400)
    ax.set_ylim(-5.5, 6.5)
    ax.set_yticks([])
    ax.set_ylabel('Hand Clap')

    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_xlabel('Beat span')

    ax.grid(True, alpha=0.3)

    ax.set_title(f'{file_name} | Dance Mode: {dance_mode}', pad=10)

    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    if debug:
        print("========== END DEBUG ==========\n")

    return fig, ax



import numpy as np
from scipy.interpolate import interp1d

def combine_phases_and_average_kde(phase_kde_dicts, num_points=400):
    phases_all = []
    window_positions_all = []

    kde_list = []
    xx_list = []

    for d in phase_kde_dicts:
        # ---- phases ----
        p = np.atleast_1d(d["phases"])
        w = np.atleast_1d(d["window_positions"])

        if p.size == 0 or w.size == 0:
            continue

        phases_all.append(p)
        window_positions_all.append(w)

        # ---- KDE ----
        kde_list.append(np.atleast_1d(d["kde"]))
        xx_list.append(np.atleast_1d(d["kde_xx"]))

    # ---- concatenate safely ----
    phases_all = np.concatenate(phases_all)
    window_positions_all = np.concatenate(window_positions_all)

    # ---- average KDEs ----
    kde_xx_common = np.linspace(0, 1, num_points)
    kde_interp = []

    for kde, xx in zip(kde_list, xx_list):
        f = interp1d(xx, kde, bounds_error=False, fill_value=0.0)
        kde_interp.append(f(kde_xx_common))

    kde_avg = np.mean(np.vstack(kde_interp), axis=0)

    return phases_all, window_positions_all, kde_xx_common, kde_avg








def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'