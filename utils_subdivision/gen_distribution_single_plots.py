import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_cycles(cycles_csv_path):
    """Load cycle onsets from a CSV file.
    
    Args:
        cycles_csv_path: Path to the CSV file containing cycle onsets
        
    Returns:
        numpy array: List of cycle locations in seconds
    """
    df = pd.read_csv(cycles_csv_path)
    return np.sort(np.array(df['Virtual Onset'].values))   # 'Selected Onset'

def load_onsets(onsets_csv_path, onset_type):
    """Load drum onsets from a CSV file.
    
    Args:
        onsets_csv_path: Path to the CSV file containing drum onsets
        onset_type: Type of onset to load ('Dun', 'J1', or 'J2')
        
    Returns:
        numpy array: List of onset locations in seconds
    """
    df = pd.read_csv(onsets_csv_path)
    return np.sort(np.array(df[onset_type].values))

def find_cycle_phases(onsets, cycles):
    """Find the cycle and phase for each onset, handling circular nature of metric cycle.
    
    Args:
        onsets: Array of onset times
        cycles: Array of cycle times
        
    Returns:
        tuple: (cycle_indices, phases, valid_onsets)
    """
    cycle_indices = []
    phases = []
    valid_onsets = []
    
    for onset in onsets:
        # Find which cycle contains this onset
        idx = np.searchsorted(cycles, onset)
        
        # Skip if onset is before first cycle or after last cycle
        if idx == 0 or idx >= len(cycles):
            continue
            
        # Initially assign to the cycle that contains the onset
        c = idx - 1
        
        # Compute initial phase within this cycle
        L_c = cycles[c]
        L_c1 = cycles[c + 1]
        
        # Skip if cycle duration is zero or invalid
        if L_c1 <= L_c:
            continue
            
        # Compute initial phase
        f = (onset - L_c) / (L_c1 - L_c)
        
        # Handle phases near cycle boundaries
        if f > 0.95:  # Close to cycle end
            # For downbeat (first subdivision), include negative phase
            if c == 0:  # First cycle
                f = f - 1.0  # Make it negative
            else:
                # For other cycles, assign to next cycle
                c = c + 1
                if c + 1 < len(cycles):
                    L_c = cycles[c]
                    L_c1 = cycles[c + 1]
                    if L_c1 <= L_c:
                        continue
                    f = (onset - L_c) / (L_c1 - L_c)
        
        cycle_indices.append(c)
        phases.append(f)
        valid_onsets.append(onset)
    
    return np.array(cycle_indices), np.array(phases), np.array(valid_onsets)

def compute_window_positions(onsets, W_start, W_end):
    """Compute relative positions within the window.
    
    Args:
        onsets: Array of onset times
        W_start: Start of window
        W_end: End of window
        
    Returns:
        tuple: (positions, mask) - Relative positions y(i) and boolean mask of valid onsets
    """
    # Skip if window duration is zero or invalid
    if W_end <= W_start:
        return np.zeros_like(onsets), np.zeros_like(onsets, dtype=bool)
    
    # Create mask for onsets within window
    mask = (onsets >= W_start) & (onsets <= W_end)
    
    # Compute positions only for valid onsets
    positions = np.zeros_like(onsets)
    positions[mask] = (onsets[mask] - W_start) / (W_end - W_start)
    
    return positions, mask

def kde_estimate(data, SIG=0.01):
    """Compute KDE estimate handling negative phases.
    
    Args:
        data: Array of phase values
        SIG: Standard deviation for Gaussian kernel
        
    Returns:
        tuple: (xx, h) - grid points and density estimates
    """
    # Filter out invalid data points
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return np.array([]), np.array([])
    
    # Extend grid to include negative values
    xx = np.arange(-0.2, 1.001, 0.001)  # Include negative values
    h = np.zeros_like(xx)
    
    for d in valid_data:
        # Compute Gaussian distribution for current onset
        p = norm.pdf(xx, d, SIG)
        p_sum = np.sum(p)
        if p_sum > 0:  # Only normalize if sum is not zero
            p = p / p_sum
        h = h + p
    
    h_sum = np.sum(h)
    if h_sum > 0:  # Only normalize if sum is not zero
        h = h / h_sum
    
    return xx, h

def plot_results(phases, window_positions, kde_xx, kde_h, file_name, onset_type, W_start=None, W_end=None, save_path=None, figsize=(12, 8), dpi=100, use_window=True):
    """Create a single plot with scatter above zero and KDE below zero, sharing x-axis.
    
    Args:
        phases: Array of phase values
        window_positions: Array of window positions
        kde_xx: Grid points for KDE
        kde_h: Density estimates
        file_name: Name of the file being analyzed
        onset_type: Type of onset being analyzed
        W_start: Start of analysis window (optional if use_window=False)
        W_end: End of analysis window (optional if use_window=False)
        save_path: Path to save the plot (if None, plot will be displayed)
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for the figure
        use_window: Whether to filter by time window (True) or use all data (False)
    """
    # Skip plotting if no valid data
    if len(phases) == 0 or len(kde_xx) == 0:
        print("No valid data to plot")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot scatter points (above zero)
    ax.scatter(phases, window_positions, alpha=0.5, color='blue', s=5)
    
    # Scale KDE to be between -0.5 and 0, starting from bottom
    kde_scaled = -0.5 + (0.5 * kde_h / np.max(kde_h))
    ax.fill_between(kde_xx, -0.5, kde_scaled, alpha=0.3, color='orange')
    
    # Set axis labels
    ax.set_xlabel('Normalized metric cycle')
    ax.set_ylabel('Relative Position in Window')
    
    # Add title
    if use_window:
        ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s | Onset: {onset_type}')
    else:
        ax.set_title(f'File: {file_name} | Full Recording | Onset: {onset_type}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(-0.55, 1.0)
    
    # Set x-axis limits to show negative values
    ax.set_xlim(-0.2, 1.0)
    
    # Customize y-axis ticks to only show positive values
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def analyze_phases(cycles_csv_path, onsets_csv_path, onset_type, W_start=None, W_end=None, save_path=None, SIG=0.01, figsize=(12, 8), dpi=100, use_window=True):
    """Main function to perform phase analysis.
    
    Args:
        cycles_csv_path: Path to the CSV file containing cycle onsets
        onsets_csv_path: Path to the CSV file containing drum onsets
        onset_type: Type of onset to analyze ('Dun', 'J1', or 'J2')
        W_start: Start of analysis window (optional if use_window=False)
        W_end: End of analysis window (optional if use_window=False)
        save_path: Path to save the plot (if None, plot will be displayed)
        SIG: Standard deviation for KDE
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for the figure
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
        print(f"Total onsets: {len(all_onsets)}")
        print(f"Onsets in window {W_start}s - {W_end}s: {len(onsets)}")
    else:
        onsets = all_onsets
        print(f"Total onsets: {len(onsets)}")
    
    # Validate input data
    if len(cycles) < 2:
        print("Error: Need at least 2 cycle points")
        return None, None, None, None
    
    if len(onsets) == 0:
        print("Error: No onset points found")
        return None, None, None, None
    
    # Step 1: Find cycles and compute phases for filtered onsets
    cycle_indices, phases, valid_onsets = find_cycle_phases(onsets, cycles)
    
    if len(phases) == 0:
        print("Error: No valid phases computed")
        return None, None, None, None
    
    # Step 2: Compute window positions
    if use_window:
        window_positions = (valid_onsets - W_start) / (W_end - W_start)
    else:
        # When not using window, distribute points evenly
        window_positions = np.linspace(0, 1, len(valid_onsets))
    
    # Step 3: KDE estimation
    kde_xx, kde_h = kde_estimate(phases, SIG)
    
    # Step 4: Plot results
    file_name = os.path.basename(cycles_csv_path).replace('_C.csv', '')
    plot_results(phases, window_positions, kde_xx, kde_h, file_name, onset_type, W_start, W_end, save_path, figsize, dpi, use_window)
    
    return phases, window_positions, kde_xx, kde_h

if __name__ == "__main__":
    # Example usage
    file_name = "BKO_E1_D1_02_Maraka"
    cycles_csv_path = f"virtual_cycles/{file_name}_C.csv"
    onsets_csv_path = f"drum_onsets/{file_name}.csv"
    W_start = 70    
    W_end = 75
    
    # Set figure size (width, height) in inches and DPI
    figsize = (10, 3)  # Slightly smaller than default
    dpi = 100         # Higher resolution
    
    # Create save directory if it doesn't exist
    save_dir = "phase_analysis_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Process all onset types
    onset_types = ["Dun", "J1", "J2"]
    for onset_type in onset_types:
        # Create save path for this onset type
        save_path = os.path.join(save_dir, f"{file_name}_{onset_type}_{W_start}_{W_end}_phase_analysis.png")
        
        # Analyze phases and save plot
        analyze_phases(
            cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end,
            save_path=save_path, figsize=figsize, dpi=dpi
        )
        
        print(f"Saved plot for {onset_type} to {save_path}")