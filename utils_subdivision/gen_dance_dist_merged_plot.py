import os
import numpy as np
import matplotlib.pyplot as plt
from gen_distribution_single_plots import load_cycles, load_onsets, find_cycle_phases, kde_estimate

def analyze_single_type(cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end, SIG=0.01):
    """Analyze a single onset type without plotting.
    
    This replicates the analysis part of phase_analysis.analyze_phases without the plotting.
    """
    # Load data
    cycles = load_cycles(cycles_csv_path)
    all_onsets = load_onsets(onsets_csv_path, onset_type)
    
    # Filter onsets by time window first
    window_mask = (all_onsets >= W_start) & (all_onsets <= W_end)
    onsets = all_onsets[window_mask]
    
    # Print some stats
    print(f"{onset_type} - Total onsets: {len(all_onsets)}")
    print(f"{onset_type} - Onsets in window {W_start}s - {W_end}s: {len(onsets)}")
    
    # Validate input data
    if len(cycles) < 2:
        print(f"{onset_type} - Error: Need at least 2 cycle points")
        return None, None, None, None
    
    if len(onsets) == 0:
        print(f"{onset_type} - Error: No onset points found in the specified window")
        return None, None, None, None
    
    # Step 1: Find cycles and compute phases for filtered onsets
    cycle_indices, phases, valid_onsets = find_cycle_phases(onsets, cycles)
    
    if len(phases) == 0:
        print(f"{onset_type} - Error: No valid phases computed")
        return None, None, None, None
    
    # Step 2: Compute window positions
    window_positions = (valid_onsets - W_start) / (W_end - W_start)
    
    # Step 3: KDE estimation
    kde_xx, kde_h = kde_estimate(phases, SIG)
    
    return phases, window_positions, kde_xx, kde_h

def load_feet_data(dance_csv_path):
    """Load feet data from dance onsets CSV file."""
    return np.loadtxt(dance_csv_path, skiprows=1)

def plot_merged_results(file_name, W_start, W_end, cycles_csv_path, onsets_csv_path, dance_csv_path, figsize=(10, 6), dpi=100):
    """Create a single plot showing merged analysis for Dun, J1, J2, and feet.
    
    Args:
        file_name: Base name of the file to analyze
        W_start: Start of analysis window
        W_end: End of analysis window
        cycles_csv_path: Path to the cycles CSV file
        onsets_csv_path: Path to the onsets CSV file
        dance_csv_path: Path to the dance onsets CSV file
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for the figure
    """
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot each onset type
    onset_types = ['Dun', 'J1', 'J2']
    colors = ['blue', 'green', 'red']
    
    # Initialize combined KDE
    combined_h = None
    kde_xx = None
    
    # Process each onset type
    for onset_type, color in zip(onset_types, colors):
        # Get phase analysis results
        phases, window_positions, curr_kde_xx, curr_kde_h = analyze_single_type(
            cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end
        )
        
        if phases is not None:
            # Plot scatter points (above zero)
            ax.scatter(phases, window_positions, alpha=0.5, color=color, s=5, 
                      label=onset_type)
            
            # Add to combined KDE
            if combined_h is None:
                combined_h = curr_kde_h
                kde_xx = curr_kde_xx
            else:
                combined_h += curr_kde_h
    
    # Add feet data to the plot
    feet_data = load_feet_data(dance_csv_path)
    # Filter feet data by time window
    window_mask = (feet_data >= W_start) & (feet_data <= W_end)
    feet_data = feet_data[window_mask]
    
    if len(feet_data) > 0:
        # Find phases for feet data
        cycles = load_cycles(cycles_csv_path)
        cycle_indices, phases, valid_onsets = find_cycle_phases(feet_data, cycles)
        window_positions = (valid_onsets - W_start) / (W_end - W_start)
        
        # Plot feet data with black dots
        ax.scatter(phases, window_positions, alpha=0.5, color='black', s=5, 
                  label='Feet')
    
    # Plot combined KDE if we have data
    if combined_h is not None:
        # Normalize combined KDE
        if np.sum(combined_h) > 0:
            combined_h = combined_h / np.max(combined_h)
        
        # Scale KDE to be between -0.5 and 0, starting from bottom
        kde_scaled = -0.5 + (0.5 * combined_h)
        ax.fill_between(kde_xx, -0.5, kde_scaled, alpha=0.3, color='purple',
                       label='Combined density')
    
    # Set axis labels
    ax.set_xlabel('Normalized metric cycle')
    ax.set_ylabel('Relative Position in Window')
    
    # Set y-axis limits and ticks
    ax.set_ylim(-0.55, 1.0)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add title
    ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s')
    
    # Add legend inside the plot
    ax.legend(loc='upper right', framealpha=0.7, fontsize='xx-small')
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    # Example usage
    file_name = "BKO_E1_D5_01_Maraka"
    cycles_csv_path = f"virtual_cycles/{file_name}_C.csv"
    onsets_csv_path = f"drum_onsets/{file_name}.csv"
    dance_csv_path = f"dance_onsets/{file_name}_T_dance_onsets.csv"
    W_start = 120
    W_end = 150
    
    # Set figure size and DPI
    figsize = (10, 3)  # Can be smaller now that legend is inside
    dpi = 200
    
    # Create save directory if it doesn't exist
    save_dir = "phase_analysis_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate and save the plot
    fig, _ = plot_merged_results(file_name, W_start, W_end, cycles_csv_path, onsets_csv_path, dance_csv_path, figsize=figsize, dpi=dpi)
    save_path = os.path.join(save_dir, f"{file_name}_{W_start}_{W_end}_dance_merged.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)  # Close the figure to free memory 