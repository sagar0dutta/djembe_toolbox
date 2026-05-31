import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dance_phase_analysis import load_cycles, load_onsets, find_cycle_phases, kde_estimate

def analyze_phases_no_plot(cycles_csv_path, onsets_csv_path, W_start, W_end, SIG=0.01):
    """Perform phase analysis without showing any plots.
    
    Args:
        cycles_csv_path: Path to the CSV file containing cycle onsets
        onsets_csv_path: Path to the CSV file containing drum onsets
        W_start: Start of analysis window
        W_end: End of analysis window
        SIG: Standard deviation for KDE
        
    Returns:
        tuple: (phases, window_positions, kde_xx, kde_h)
    """
    # Load data
    cycles = load_cycles(cycles_csv_path)
    all_onsets = load_onsets(onsets_csv_path)
    
    # Filter onsets by time window first
    window_mask = (all_onsets >= W_start) & (all_onsets <= W_end)
    onsets = all_onsets[window_mask]
    
    # Print some stats
    print(f"Total onsets: {len(all_onsets)}")
    print(f"Onsets in window {W_start}s - {W_end}s: {len(onsets)}")
    
    # Validate input data
    if len(cycles) < 2:
        print("Error: Need at least 2 cycle points")
        return None, None, None, None
    
    if len(onsets) == 0:
        print("Error: No onset points found in the specified window")
        return None, None, None, None
    
    # Step 1: Find cycles and compute phases for filtered onsets
    cycle_indices, phases, valid_onsets = find_cycle_phases(onsets, cycles)
    
    if len(phases) == 0:
        print("Error: No valid phases computed")
        return None, None, None, None
    
    # Step 2: Compute window positions
    window_positions = (valid_onsets - W_start) / (W_end - W_start)
    
    # Step 3: KDE estimation
    kde_xx, kde_h = kde_estimate(phases, SIG)
    
    return phases, window_positions, kde_xx, kde_h

def animate_dance_phase_analysis(file_name, W_start, W_end, cycles_csv_path, onsets_csv_path, figsize=(10, 3), dpi=100, save_path=None, save_dir=None):
    """Animate the dance phase analysis plot with moving playheads.
    
    Args:
        file_name: Base name of the file to analyze
        W_start: Start of analysis window
        W_end: End of analysis window
        cycles_csv_path: Path to the cycles CSV file
        onsets_csv_path: Path to the onsets CSV file
        figsize: Figure size in inches
        dpi: Dots per inch for the figure
        save_path: Path to save the animation (MP4 format)
        save_dir: Directory to save the animation in
    """
    # Get cycles for playhead animation
    cycles = load_cycles(cycles_csv_path)
    
    print(f"Loading data from:\n  {cycles_csv_path}\n  {onsets_csv_path}")
    
    # Get the phase analysis results
    phases, window_positions, kde_xx, kde_h = analyze_phases_no_plot(
        cycles_csv_path, onsets_csv_path, W_start, W_end
    )
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot scatter points (above zero)
    ax.scatter(phases, window_positions, alpha=0.5, color='blue', s=5)
    
    # Scale KDE to be between -0.5 and 0, starting from bottom
    kde_scaled = -0.5 + (0.5 * kde_h / np.max(kde_h))  # Start at -0.5 and go up
    ax.fill_between(kde_xx, -0.5, kde_scaled, alpha=0.3, color='orange')
    
    # Set axis labels
    ax.set_xlabel('Normalized metric cycle')
    ax.set_ylabel('Relative Position in Window')
    
    # Add title
    ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s | Onset: feet')
    
    # Set y-axis limits
    ax.set_ylim(-0.55, 1.0)  # Exact limits: -0.55 to 1.0
    
    # Customize y-axis ticks to only show positive values
    yticks = np.arange(0, 1.1, 0.2)  # Generate ticks from 0 to 1
    ax.set_yticks(yticks)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create vertical playhead line
    playhead, = ax.plot([0, 0], [-0.55, 1.0], 'k-', lw=1, alpha=0.7)
    # Create horizontal playhead line
    h_playhead, = ax.plot([0, 1], [0, 0], 'k-', lw=1, alpha=0.7)
    
    def find_phase(t):
        """Find the phase for a given time t."""
        idx = np.searchsorted(cycles, t)
        if idx == 0 or idx >= len(cycles):
            return None
        c = idx - 1
        L_c = cycles[c]
        L_c1 = cycles[c + 1]
        return (t - L_c) / (L_c1 - L_c)
    
    def update(frame):
        """Update function for animation."""
        phase = find_phase(frame)
        if phase is not None:
            # Update vertical playhead
            playhead.set_xdata([phase, phase])
            # Update horizontal playhead position (normalized to 0-1)
            y_pos = (frame - W_start) / (W_end - W_start)
            h_playhead.set_ydata([y_pos, y_pos])
            # Update title with current time
            ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s | Onset: feet | Time: {frame:.2f}s')
        return playhead, h_playhead,
    
    # Create animation
    print("\nCreating animation...")
    # Only create frames within the analysis window
    frames = np.arange(W_start, W_end, 0.05)  # 50ms steps
    print(f"Animation will have {len(frames)} frames")
    print(f"Time range: {frames[0]:.2f}s - {frames[-1]:.2f}s")
    
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=50, blit=True
    )
    
    # Apply tight_layout before saving
    plt.tight_layout()
    
    if save_path:
        # If save_dir is provided, use it to construct the full path
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_path)
        
        print(f"\nSaving animation to: {save_path}")
        try:
            # Save animation as MP4
            writer = animation.FFMpegWriter(fps=20, bitrate=2000)
            anim.save(save_path, writer=writer)
            plt.close(fig)  # Explicitly close the figure
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {str(e)}")
            plt.close(fig)  # Close figure even if there's an error
    else:
        print("Error: save_path must be provided")
        plt.close(fig)  # Close figure if no save path
    
    return anim

if __name__ == "__main__":
    # Example usage
    file_name = "BKO_E1_D1_01_Suku"
    cycles_csv_path = f"virtual_cycles/{file_name}_C.csv"
    onsets_csv_path = f"dance_onsets/{file_name}_T_dance_onsets.csv"
    W_start = 60    
    W_end = 80
    
    # Set figure size and DPI
    figsize = (10, 3)  # Slightly smaller than default
    dpi = 200         # Higher resolution
    
    # Save path for the animation
    save_dir = "phase_analysis_animations"  # New save directory
    save_path = f"{file_name}_{W_start}_{W_end}_feet.mp4"
    
    print(f"Starting dance phase analysis for {file_name}")
    print(f"Window: {W_start}s - {W_end}s")
    
    try:
        anim = animate_dance_phase_analysis(file_name, W_start, W_end, 
                                          cycles_csv_path, onsets_csv_path,
                                          figsize=figsize, dpi=dpi,
                                          save_path=save_path,
                                          save_dir=save_dir)
    except Exception as e:
        print(f"Error during animation: {str(e)}") 