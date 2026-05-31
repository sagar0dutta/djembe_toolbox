import os
import numpy as np
import matplotlib.pyplot as plt
from .gen_distribution_single_plots import load_cycles, load_onsets, find_cycle_phases, kde_estimate

def analyze_single_type(cycles_csv_path, onsets_csv_path, onset_type, W_start=None, W_end=None, SIG=0.01, use_window=True):
    """Analyze a single onset type without plotting.
    
    This replicates the analysis part of phase_analysis.analyze_phases without the plotting.
    
    Args:
        cycles_csv_path: Path to cycles CSV file
        onsets_csv_path: Path to onsets CSV file
        onset_type: Type of onset to analyze ('Dun', 'J1', 'J2')
        W_start: Start of analysis window (optional if use_window=False)
        W_end: End of analysis window (optional if use_window=False)
        SIG: Standard deviation for KDE estimation
        use_window: Whether to filter by time window (True) or use all data (False)
    """
    # Load data
    cycles = load_cycles(cycles_csv_path)
    all_onsets = load_onsets(onsets_csv_path, onset_type)
    
    # Filter onsets by time window if use_window is True
    if use_window:
        if W_start is None or W_end is None:
            raise ValueError("W_start and W_end must be provided when use_window=True")
        window_mask = (all_onsets >= W_start) & (all_onsets <= W_end)
        onsets = all_onsets[window_mask]
        print(f"{onset_type} - Total onsets: {len(all_onsets)}")
        print(f"{onset_type} - Onsets in window {W_start}s - {W_end}s: {len(onsets)}")
    else:
        onsets = all_onsets
        print(f"{onset_type} - Total onsets: {len(onsets)}")
    
    # Validate input data
    if len(cycles) < 2:
        print(f"{onset_type} - Error: Need at least 2 cycle points")
        return None, None, None, None
    
    if len(onsets) == 0:
        print(f"{onset_type} - Error: No onset points found")
        return None, None, None, None
    
    # Step 1: Find cycles and compute phases for filtered onsets
    cycle_indices, phases, valid_onsets = find_cycle_phases(onsets, cycles)
    
    if len(phases) == 0:
        print(f"{onset_type} - Error: No valid phases computed")
        return None, None, None, None
    
    # Step 2: Compute window positions
    if use_window:
        window_positions = (valid_onsets - W_start) / (W_end - W_start)
    else:
        # When not using window, distribute points evenly
        window_positions = np.linspace(0, 1, len(valid_onsets))
    
    # Step 3: KDE estimation
    kde_xx, kde_h = kde_estimate(phases, SIG)
    
    return phases, window_positions, kde_xx, kde_h

def plot_combined(file_name, cycles_csv_path, onsets_csv_path, W_start=None, W_end=None, figsize=(10, 8), dpi=100, use_window=True):
    """Create a combined plot showing phase analysis for Dun, J1, and J2.
    
    Args:
        file_name: Base name of the file to analyze
        cycles_csv_path: Path to cycles CSV file
        onsets_csv_path: Path to onsets CSV file
        W_start: Start of analysis window (optional if use_window=False)
        W_end: End of analysis window (optional if use_window=False)
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for the figure
        use_window: Whether to filter by time window (True) or use all data (False)
    """
    # Create figure with three subplots sharing x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    axes = [ax1, ax2, ax3]
    onset_types = ['Dun', 'J1', 'J2']
    colors = ['blue', 'green', 'red']
    
    # Process each onset type
    for ax, onset_type, color in zip(axes, onset_types, colors):
        # Get phase analysis results
        phases, window_positions, kde_xx, kde_h = analyze_single_type(
            cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end, use_window=use_window
        )
        
        if phases is not None:
            # Plot scatter points (above zero)
            ax.scatter(phases, window_positions, alpha=0.5, color=color, s=5)
            
            # Scale KDE to be between -0.5 and 0, starting from bottom
            kde_scaled = -0.5 + (0.5 * kde_h / np.max(kde_h))
            ax.fill_between(kde_xx, -0.5, kde_scaled, alpha=0.3, color=color)
            
            # Set y-axis limits and ticks
            ax.set_ylim(-0.55, 1.0)
            yticks = np.arange(0, 1.1, 0.2)
            ax.set_yticks(yticks)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add onset type label
            ax.set_ylabel(f'{onset_type}\nRelative Position')
    
    # Set common x-label and title
    axes[-1].set_xlabel('Normalized metric cycle')
    if use_window:
        fig.suptitle(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s')
    else:
        fig.suptitle(f'File: {file_name} | Full Recording')
    
    plt.tight_layout()
    return fig, axes

if __name__ == "__main__":
    # Example usage
    file_name = "BKO_E1_D1_02_Maraka"
    W_start = 50
    W_end = 180
    
    # Set figure size and DPI
    figsize = (10, 9)  # Taller to accommodate three plots
    dpi = 100
    
    # Create save directory if it doesn't exist
    save_dir = "phase_analysis_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate and save the plot
    fig, _ = plot_combined(file_name, W_start, W_end, figsize=figsize, dpi=dpi)
    save_path = os.path.join(save_dir, f"{file_name}_{W_start}_{W_end}__combined_phase_analysis.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)  # Close the figure to free memory 