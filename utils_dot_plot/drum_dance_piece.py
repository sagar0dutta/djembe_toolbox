import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils_subdivision.gen_distribution_subplot import analyze_single_type
from utils_dot_plot.kinematic_dot_plot import *
from utils_dot_plot.drum_merged import *



def plot_combined_drum_dance(piece_type, 
                             dance_mode, 
                             drum_phases_kde_all, 
                             dance_phases_kde_all, 
                             figsize=(10, 6), 
                             dpi=300,
                             save_dir=None,
                             legend_flag=True):
    """Create a single figure with two subplots: drum and dance"""
    
    # Create figure with two subplots
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    fig.tight_layout(pad=4.0)
    
    # Plot drum data (top subplot)
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
        ax1.scatter(phases * 400,
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
        
        ax1.fill_between(combined_xx * 400,
                        -5, kde_scaled,
                        alpha=0.3, color='purple',
                        label='Combined KDE')

    # Add subdivision lines for drum
    for subdiv in range(1, 13):  # 12 subdivisions
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax1.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax1.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling for drum plot
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax1.set_xlim(0, 400)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_xlabel('Beat span')
    
    ax1.set_ylim(-5.5, 20.5)
    ax1.set_yticks([3, 10, 17])
    ax1.set_yticklabels(['Dun', 'J1', 'J2'])
    
    ax1.set_ylabel('Instrument')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-33, 400)

    # Title for drum plot
    ax1.set_title(f'{piece_type} | {dance_mode} | Drum Onsets', pad=10)
    
    if legend_flag:
        ax1.legend(loc='upper left', framealpha=0.4, fontsize=6)

    # Plot dance data (bottom subplot)
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
        ax2.scatter(phases * 400,
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
            ax2.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

    # Add subdivision lines for dance
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax2.vlines(x_pos, -5.5, 13.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax2.vlines(x_pos, -5.5, 13.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling for dance plot
    ax2.set_xticks(xtick)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xlim(-33, 400)
    # ax2.set_xlabel('Beat span')
    
    ax2.set_ylim(-5.5, 13.5)
    ax2.set_yticks([3, 10])
    ax2.set_yticklabels(['LF', 'RF'])
    ax2.set_ylabel('Foot')
    ax2.grid(True, alpha=0.3)

    # Title for dance plot
    ax2.set_title(f'{piece_type} | {dance_mode} | Dance Onsets', pad=10)
    
    if legend_flag:
        ax2.legend(loc='upper left', framealpha=0.4, fontsize=6)

    # Save the figure
    save_mode_dir = os.path.join(save_dir, "drum_dance_kde_by_piece", dance_mode)
    os.makedirs(save_mode_dir, exist_ok=True)
    save_path = os.path.join(save_mode_dir, f"{piece_type}_{dance_mode}_combined.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()











def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'


