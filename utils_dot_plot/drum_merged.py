import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


from utils_dot_plot.drum_single import analyze_phases
# from utils_dot_plot.drum_merged import plot_merged_per_mode

from utils_subdivision.gen_distribution_subplot import analyze_single_type, kde_estimate




def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'

def drum_plot_merged_stacked(file_name,
                       dance_mode,
                       cycles_csv_path,
                       onsets_csv_path,
                       dance_mode_time_segments,
                       W_start=None,
                       W_end=None,
                       figsize=(10, 12),
                       dpi=100,
                       use_window=True,
                       legend_flag=True):
    """Create a single plot showing merged analysis for Dun, J1, and J2
    with stacked, colored scatter and combined KDE."""

    
    # 2) set up single merged axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    onset_types = ['Dun', 'J1', 'J2']
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    
    # vertical stacking ranges for each type
    vertical_ranges = {
        'Dun': (1, 6),    # y from 1–6
        'J1':  (8, 13),   # y from 8–13
        'J2':  (15, 20),  # y from 15–20
    }

    combined_h = None
    combined_xx = None

    # 3) loop onset types, plot colored scatter
    drum_phases_kde = { onset_type: {"phases": {}, "y_scaled": {}, "kde_h": {}, "kde_xx": {}} for onset_type in onset_types }
    
    for onset_type, color in zip(onset_types, colors):
        # For each time segment, analyze and combine the results
        all_phases = []
        all_window_pos = []
        segment_kde_h = None
        segment_kde_xx = None

        for start_time, end_time in dance_mode_time_segments:
            phases, window_pos, kde_xx, kde_h = analyze_single_type(
                cycles_csv_path, onsets_csv_path,
                onset_type, start_time, end_time,  # Use segment times
                use_window=True  # Force window mode for segments
            )
            
            if phases is not None:
                all_phases.extend(phases)
                all_window_pos.extend(window_pos)
                
                # Accumulate KDE
                if segment_kde_h is None:
                    segment_kde_h = kde_h.copy()
                    segment_kde_xx = kde_xx.copy()
                else:
                    segment_kde_h += kde_h

        
        
        if not all_phases:  # Skip if no data found
            continue

        # Convert lists to numpy arrays
        phases = np.array(all_phases)
        window_pos = np.array(all_window_pos)

        # stack y
        y0, y1 = vertical_ranges[onset_type]
        y_scaled = y0 + (window_pos * (y1 - y0))
            
        # Plot scatter with single color for each instrument
        ax.scatter(phases * 400,
                   y_scaled,
                   s=5, alpha=0.6,
                   color=color,
                   label=onset_type)

        # accumulate KDE
        if combined_h is None:
            combined_h = segment_kde_h.copy()
            combined_xx = segment_kde_xx.copy()
        else:
            combined_h += segment_kde_h
            
        drum_phases_kde[onset_type]["phases"] = phases
        drum_phases_kde[onset_type]["y_scaled"] = y_scaled
        drum_phases_kde[onset_type]["kde_h"] = segment_kde_h
        drum_phases_kde[onset_type]["kde_xx"] = segment_kde_xx

    # 4) draw combined KDE at bottom (-5 to 0)
    if combined_h is not None:
        # normalize to 0–1
        combined_h = combined_h / np.max(combined_h)
        kde_scaled = -5 + (5 * combined_h)
        
        ax.fill_between(combined_xx * 400,
                        -5, kde_scaled,
                        alpha=0.3, color='purple',
                        label='Combined KDE')

    # 5) Add subdivision lines
    for subdiv in range(1, 13):  # 12 subdivisions
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    

    # 5) styling & axes
    # xtick = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367, 400]
    # xtick_labels = [1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0]
    
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xlim(0, 400)
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 20.5)
    ax.set_yticks([3, 10, 17])
    ax.set_yticklabels(['Dun', 'J1', 'J2'])
    
    ax.set_ylabel('Instrument')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-33, 400)

    # 6) title & legend
    title = f'File: {file_name} | Dance Mode: {dance_mode}'
    title += f' | Segments: {len(dance_mode_time_segments)}' if use_window else ' | Full Recording'
    ax.set_title(title, pad=10)
    
    # Add legend
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax, drum_phases_kde


def drum_plot_combined_merged_stacked(piece_type, dance_mode, drum_phases_kde_all, 
                                 figsize=(10, 3), dpi=200, legend_flag=True):
    """Create a single plot showing combined merged analysis for all pieces of a type"""
    
    # Set up single merged axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    onset_types = ['Dun', 'J1', 'J2']
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    
    # vertical stacking ranges for each type
    vertical_ranges = {
        'Dun': (1, 6),    # y from 1–6
        'J1':  (8, 13),   # y from 8–13
        'J2':  (15, 20),  # y from 15–20
    }

    combined_h = None
    combined_xx = None

    # Combine data for each onset type
    for onset_type, color in zip(onset_types, colors):
        # Combine phases and y_scaled from all pieces
        all_phases = []
        all_y_scaled = []
        segment_kde_h = None
        segment_kde_xx = None

        # Loop through all pieces' data
        for piece_data in drum_phases_kde_all:
            if onset_type in piece_data:
                # Combine phases and y_scaled
                all_phases.extend(piece_data[onset_type]["phases"])
                all_y_scaled.extend(piece_data[onset_type]["y_scaled"])
                
                # Accumulate KDE
                if segment_kde_h is None:
                    segment_kde_h = piece_data[onset_type]["kde_h"].copy()
                    segment_kde_xx = piece_data[onset_type]["kde_xx"].copy()
                else:
                    segment_kde_h += piece_data[onset_type]["kde_h"]

        if not all_phases:  # Skip if no data found
            continue

        # Convert lists to numpy arrays
        phases = np.array(all_phases)
        y_scaled = np.array(all_y_scaled)
            
        # Plot scatter with single color for each instrument
        ax.scatter(phases * 400,
                   y_scaled,
                   s=5, alpha=0.6,
                   color=color,
                   label=onset_type)

        # accumulate KDE
        if combined_h is None:
            combined_h = segment_kde_h.copy()
            combined_xx = segment_kde_xx.copy()
        else:
            combined_h += segment_kde_h

    # Draw combined KDE at bottom (-5 to 0)
    if combined_h is not None:
        # normalize to 0–1
        combined_h = combined_h / np.max(combined_h)
        kde_scaled = -5 + (5 * combined_h)
        
        ax.fill_between(combined_xx * 400,
                        -5, kde_scaled,
                        alpha=0.3, color='purple',
                        label='Combined KDE')

    # Add subdivision lines
    for subdiv in range(1, 13):  # 12 subdivisions
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling & axes
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xlim(0, 400)
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 20.5)
    ax.set_yticks([3, 10, 17])
    ax.set_yticklabels(['Dun', 'J1', 'J2'])
    
    ax.set_ylabel('Instrument')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-33, 400)

    # Title & legend
    title = f'Piece Type: {piece_type} | Dance Mode: {dance_mode}'
    title += f' | Combined from {len(drum_phases_kde_all)} pieces'
    ax.set_title(title, pad=10)
    
    # Add legend
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax


def drum_plot_merged_stacked_all_modes(
    file_name,
    cycles_csv_path,
    onsets_csv_path,
    dance_mode_time_segments_all,  # Dictionary containing time segments for all modes
    figsize=(10, 3),
    dpi=200,
    use_window=True,
    legend_flag=True
):
    """Create a single plot showing combined drum onset analysis for all modes for a single file."""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    vertical_ranges = {
        'Dun': (1, 6),
        'J1': (8, 13),
        'J2': (15, 20)
    }

    # Fixed colors for each drum type
    drum_colors = {
        'Dun': '#1f77b4',   # blue
        'J1': '#d62728',    # red
        'J2': '#2ca02c'     # green
    }

    # Initialize combined data structure
    onset_types = ['Dun', 'J1', 'J2']
    drum_phases_kde = {onset_type: {"phases": [], "y_scaled": []} for onset_type in onset_types}
    
    # Process each mode
    for mode, dance_mode_time_segments in dance_mode_time_segments_all.items():
        # Get cycles data
        cycles_df = pd.read_csv(cycles_csv_path)
        cycle_times = cycles_df['Virtual Onset'].values
        
        # Get onsets data
        onsets_df = pd.read_csv(onsets_csv_path)
        
        # Process each onset type
        for onset_type in onset_types:
            # Get onset times for this type
            onset_times = onsets_df[onset_type].dropna().values  # Remove NaN values
            
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
                    
                    # Calculate phase
                    phase = (onset_time - cycle_start) / (cycle_end - cycle_start)
                    
                    # Handle cycle wrapping
                    if phase > 0.98:  # If phase is closer to next cycle
                        phase = phase - 1.0
                    
                    # Add to combined data
                    drum_phases_kde[onset_type]["phases"].append(phase)
                    drum_phases_kde[onset_type]["y_scaled"].append(len(drum_phases_kde[onset_type]["phases"]))

    # Plot the combined data
    for onset_type, color in drum_colors.items():
        phases = np.array(drum_phases_kde[onset_type]["phases"])
        y_scaled = np.array(drum_phases_kde[onset_type]["y_scaled"])
        
        if len(phases) == 0:
            continue
            
        # Normalize y_scaled values to fit within the vertical range
        y_min, y_max = vertical_ranges[onset_type]
        y_scaled = y_min + (y_scaled - np.min(y_scaled)) * (y_max - y_min) / (np.max(y_scaled) - np.min(y_scaled))
        
        # Plot scatter
        ax.scatter(phases * 400,
                  y_scaled,
                  s=5, alpha=0.4,
                  color=color,
                  label=f'{onset_type}')

    # Calculate and plot combined KDE
    all_phases = []
    for onset_type in onset_types:
        all_phases.extend(drum_phases_kde[onset_type]["phases"])
    
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
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 20.5)
    ax.set_yticks([3, 10, 17])
    ax.set_yticklabels(['Dun', 'J1', 'J2'])
    ax.set_ylabel('Drum')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'File: {file_name} | All Modes Combined'
    # title += f' | Combined from {len(dance_mode_time_segments_all)} modes'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax, drum_phases_kde



def plot_combined_drum_stacked_all_modes(
    piece_type,
    piece_drum_phases_kde,  # Dictionary containing all modes' data
    figsize=(10, 3),
    dpi=300,
    legend_flag=True
):
    """Create a single plot showing combined drum onset analysis for all pieces and all modes."""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    vertical_ranges = {
        'Dun': (1, 6),
        'J1': (8, 13),
        'J2': (15, 20)
    }

    # Fixed colors for each drum type
    drum_colors = {
        'Dun': '#1f77b4',   # blue
        'J1': '#d62728',    # red
        'J2': '#2ca02c'     # green
    }

    combined_phases = []
    
    # Process each mode
    for mode in piece_drum_phases_kde.keys():
        if piece_type not in piece_drum_phases_kde[mode]:
            continue
            
        # Get the list of data for this piece type in this mode
        piece_data_list = piece_drum_phases_kde[mode][piece_type]
        
        # Combine data for each drum type
        for drum_type, color in drum_colors.items():
            # Combine phases and y_scaled from all pieces
            all_phases = []
            all_y_scaled = []

            # Loop through all pieces' data
            for piece_data in piece_data_list:
                if drum_type in piece_data:
                    # Combine phases and y_scaled
                    all_phases.extend(piece_data[drum_type]["phases"])
                    all_y_scaled.extend(piece_data[drum_type]["y_scaled"])
                    combined_phases.extend(piece_data[drum_type]["phases"])

            if not all_phases:  # Skip if no data found
                continue

            # Convert lists to numpy arrays
            phases = np.array(all_phases)
            y_scaled = np.array(all_y_scaled)
            
            # Normalize y_scaled values to fit within the vertical range
            y_min, y_max = vertical_ranges[drum_type]
            y_scaled = y_min + (y_scaled - np.min(y_scaled)) * (y_max - y_min) / (np.max(y_scaled) - np.min(y_scaled))
                
            # Plot scatter with drum-specific color
            ax.scatter(phases * 400,
                      y_scaled,
                      s=5, alpha=0.4,  # Reduced alpha for better visibility of overlapping points
                      color=color,
                      label=f'{drum_type}')

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
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 20.5)
    ax.set_yticks([3, 10, 17])
    ax.set_yticklabels(['Dun', 'J1', 'J2'])
    ax.set_ylabel('Drum')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'Piece: {piece_type} | All Modes Combined'
    # title += f' | Combined from {len(piece_drum_phases_kde)} modes'
    ax.set_title(title, pad=10)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax




def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'