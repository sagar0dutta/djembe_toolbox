import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils_subdivision.gen_distribution_single_plots import load_cycles, load_onsets, find_cycle_phases, kde_estimate

# Use system ffmpeg which has libx264 support
FFMPEG_PATH = "/itf-fi-ml/home/sagardu/bin/ffmpeg"

def analyze_phases_no_plot(cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end, SIG=0.01):
    """Perform phase analysis without showing any plots.
    
    Args:
        cycles_csv_path: Path to the CSV file containing cycle onsets
        onsets_csv_path: Path to the CSV file containing drum onsets
        onset_type: Type of onset to analyze ('Dun', 'J1', or 'J2')
        W_start: Start of analysis window
        W_end: End of analysis window
        SIG: Standard deviation for KDE
        
    Returns:
        tuple: (phases, window_positions, kde_xx, kde_h)
    """
    # Load data
    cycles = load_cycles(cycles_csv_path)
    all_onsets = load_onsets(onsets_csv_path, onset_type)
    
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


def animate_window_in_merged_phase_mode():
    pass


def animate_merged_phase_analysis(file_name, 
                                  W_start, 
                                  W_end, 
                                  cycles_csv_path, 
                                  onsets_csv_path, 
                                  figsize=(10, 6), 
                                  dpi=100, 
                                  save_fname=None, 
                                  save_dir=None, 
                                  legend_flag=True,
                                  fps=24,):
    """Animate the merged phase analysis plot with a moving playhead.
    
    Args:
        file_name: Base name of the file to analyze
        W_start: Start of analysis window
        W_end: End of analysis window
        cycles_csv_path: Path to the cycles CSV file
        onsets_csv_path: Path to the onsets CSV file
        figsize: Figure size in inches
        dpi: Dots per inch for the figure
        save_fname: Path to save the animation (MP4 format)
        save_dir: Directory to save the animation in
    """
    # Get cycles for playhead animation
    print(f"Generating animation for drum merged dot plot for {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s")
    cycles = load_cycles(cycles_csv_path)
    
    print(f"Loading data from:\n  {cycles_csv_path}\n  {onsets_csv_path}")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot each onset type
    onset_types = ['Dun', 'J1', 'J2']
    colors = ['blue', 'green', 'red']
    scatter_plots = []
    kde_plots = []
    
    # Initialize combined KDE
    combined_h = None
    kde_xx = None
    
    # Process each onset type
    for onset_type, color in zip(onset_types, colors):
        # Get phase analysis results
        phases, window_positions, curr_kde_xx, curr_kde_h = analyze_phases_no_plot(
            cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end
        )
        
        if phases is not None:
            # Plot scatter points (above zero)
            scatter = ax.scatter(phases, window_positions, alpha=0.5, color=color, s=5, 
                                label=onset_type)
            scatter_plots.append(scatter)
            
            # Add to combined KDE
            if combined_h is None:
                combined_h = curr_kde_h
                kde_xx = curr_kde_xx
            else:
                combined_h += curr_kde_h
    
    # Plot combined KDE if we have data
    if combined_h is not None:
        # Normalize combined KDE
        if np.sum(combined_h) > 0:
            combined_h = combined_h / np.max(combined_h)
        
        # Scale KDE to be between -0.5 and 0, starting from bottom
        kde_scaled = -0.6 + (0.5 * combined_h)
        kde_plot = ax.fill_between(kde_xx, -0.6, kde_scaled, alpha=0.3, color='purple',
                                 label='Combined density')
        kde_plots.append(kde_plot)
    
    # Set axis labels
    ax.set_xlabel('Beat Span')    # Normalized metric cycle
    ax.set_ylabel('Relative Position in Window')
    
    # Add title
    ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s')
    
    # Set x,y-axis limits and ticks
    ax.set_xlim(-0.1, 1.0)
    xticks = [0, 0.25, 0.5, 0.75, 1]        # 0, 0.25, 0.5, 0.75, 1
    ax.set_xticks(xticks) 
    ax.set_xticklabels([1, 2, 3, 4, 5])
    
    ax.set_ylim(-0.65, 1.0)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    
    # Draw vertical lines at each subdivision i/12
    ymin, ymax = ax.get_ylim()
    for subdiv  in range(1, 13):
        xpos = (subdiv - 1) / 12    # subdiv 1 → 0.0, subdiv 4 → 0.25, etc.
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), linestyle='--', linewidth=1, alpha=0.3)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if legend_flag:
        # Add legend inside the plot
        ax.legend(loc='upper left', framealpha=0.4, fontsize = 'xx-small')
    
    # Create playhead line
    playhead, = ax.plot([0, 0], [-0.65, 1.0], color='orange', lw=1.5, alpha=0.7, linestyle='-')
    # Create horizontal playhead line
    h_playhead, = ax.plot([0, 1], [0, 0], color='orange', lw=1.5, alpha=0.7, linestyle='-')
    
    def find_phase(t):
        """Find the phase for a given time t."""
        idx = np.searchsorted(cycles, t)
        if idx == 0 or idx >= len(cycles):
            return None
        c = idx - 1
        L_c = cycles[c]
        L_c1 = cycles[c + 1]
        return (t - L_c) / (L_c1 - L_c)
    
    # Update function for animation -----------------------------------------------------------------
    def update(frame):
        """Update function for animation."""
        phase = find_phase(frame)   # convert the current time (frame) into a phase value
                                    # The phase represents where in the musical cycle we are ( 0 to 1)
        
        if phase is not None:
            
            playhead.set_xdata([phase, phase])  # Updates the vertical playhead line's position
            
            # Update horizontal playhead position (normalized to 0-1)
            # Calculates the normalized position (0 to 1) of the current time within the analysis window
            # W_start and W_end define the time window being analyzed
            # creates a value between 0 and 1 representing progress through the window
            y_pos = (frame - W_start) / (W_end - W_start)
            h_playhead.set_ydata([y_pos, y_pos])            # Updates the horizontal playhead line's position
            
            
            # Update title with current time
            ax.set_title(f'File: {file_name} | Window: {W_start:.0f}s - {W_end:.0f}s | Onset: Merged Drums | Time: {frame:.0f}s')
        
        return playhead, h_playhead,
    
    # Create animation
    print("\nCreating animation...")
    # Only create frames within the analysis window
    frames = np.arange(W_start, W_end, 1/fps)  
    print(f"Animation will have {len(frames)} frames")
    print(f"Time range: {frames[0]:.2f}s - {frames[-1]:.2f}s")
    
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=50, blit=True
    )
    
    # Apply tight_layout before saving
    plt.tight_layout()
    
    if save_fname:
        # If save_dir is provided, use it to construct the full path
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_fname = os.path.join(save_dir, save_fname)
        
        print(f"\nSaving animation to: {save_fname}")
        try:
            # Temporarily modify PATH to use system ffmpeg
            original_path = os.environ.get('PATH', '')
            ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
            
            # Save animation as MP4
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(save_fname, writer=writer)
            plt.close(fig)  # Explicitly close the figure
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {str(e)}")
            plt.close(fig)  # Close figure even if there's an error
    else:
        print("Error: save_fname must be provided")
        plt.close(fig)  # Close figure if no save path
    
    return anim

def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'


########## Animate drum dot plot with a moving playhead within a user-defined window #########


def animate_merged_phase_analysis_with_user_window(
    file_name, 
    W_start,           # Full window start (e.g., 20)
    W_end,             # Full window end (e.g., 100)
    user_start,        # Animation start time
    user_end,          # Animation end time
    cycles_csv_path, 
    onsets_csv_path, 
    figsize=(10, 6), 
    dpi=100, 
    save_dir=None, 
    legend_flag=True,
    fps= 24,
):
    """Animate the merged phase analysis plot with a moving playhead within a user-defined window.
    
    Args:
        file_name: Base name of the file to analyze
        W_start: Start of full analysis window (e.g., 20)
        W_end: End of full analysis window (e.g., 100)
        user_start: Start time for animation playhead
        user_end: End time for animation playhead
        cycles_csv_path: Path to the cycles CSV file
        onsets_csv_path: Path to the onsets CSV file
        figsize: Figure size in inches
        dpi: Dots per inch for the figure
        save_fname: Path to save the animation (MP4 format)
        save_dir: Directory to save the animation in
        legend_flag: Whether to show the legend
    """
    # Validate user window is within full window
    if not (W_start <= user_start <= user_end <= W_end):
        raise ValueError("User window must be within the full window range")

    print(f"Generating animation for drum merged dot plot for {file_name}")
    print(f"Full window: {W_start:.1f}s - {W_end:.1f}s")
    print(f"Animation window: {user_start:.1f}s - {user_end:.1f}s")
    
    cycles = load_cycles(cycles_csv_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot each onset type
    onset_types = ['Dun', 'J1', 'J2']
    colors = ['blue', 'green', 'red']
    scatter_plots = []
    kde_plots = []
    
    # Initialize combined KDE
    combined_h = None
    kde_xx = None
    
    # Process each onset type for the FULL window
    for onset_type, color in zip(onset_types, colors):
        phases, window_positions, curr_kde_xx, curr_kde_h = analyze_phases_no_plot(
            cycles_csv_path, onsets_csv_path, onset_type, W_start, W_end
        )
        
        if phases is not None:
            scatter = ax.scatter(phases, window_positions, alpha=0.5,   # Scatter plot
                                 color=color, s=14, label=onset_type)
            scatter_plots.append(scatter)
            
            if combined_h is None:
                combined_h = curr_kde_h
                kde_xx = curr_kde_xx
            else:
                combined_h += curr_kde_h
    
    # Plot combined KDE
    if combined_h is not None:
        if np.sum(combined_h) > 0:
            combined_h = combined_h / np.max(combined_h)
        kde_scaled = -0.6 + (0.5 * combined_h)
        kde_plot = ax.fill_between(kde_xx, -0.6, kde_scaled, alpha=0.3, color='gray',
                                 label='Combined density')
        kde_plots.append(kde_plot)
    
    # Set up plot
    xticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_xlabel('4-beat cycle span')
    ax.set_xlim(-0.1, 1.0)
    
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])  # comment to show y-axis labels
    ax.set_ylabel('Relative Position in Window')
    ax.set_ylim(-0.65, 1.0)
    
    # Draw subdivision lines
    ymin, ymax = ax.get_ylim()
    for subdiv in range(1, 13):
        xpos = (subdiv - 1) / 12
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), 
                     linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), 
                     linestyle='--', linewidth=1, alpha=0.3)
    
    # ax.grid(True, alpha=0.3)
    
    if legend_flag:
        ax.legend(loc='upper left', framealpha=0.4, fontsize='xx-small')
    
    # Create playheads
    playhead, = ax.plot([0, 0], [-0.65, 1.0], color='orange', lw=1.5, alpha=0.7, linestyle='-')
    h_playhead, = ax.plot([0, 1], [0, 0], color='orange', lw=1.5, alpha=0.7, linestyle='-')
    
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
            
            # Update horizontal playhead (normalized to user window)
            y_pos = (frame - W_start) / (W_end - W_start)
            h_playhead.set_ydata([y_pos, y_pos])
            
            # Update title with both windows and current time
            ax.set_title( f'File: {file_name} | Window: {user_start:.0f}s - {user_end:.0f}s | Onset: Merged Drums | Time: {frame:.0f}s')
        
        return playhead, h_playhead,
    
    # Create animation frames for user window only
    print("\nCreating animation...")
    frames = np.arange(user_start, user_end, 1/fps)
    print(f"Animation will have {len(frames)} frames")
    print(f"Time range: {frames[0]:.2f}s - {frames[-1]:.2f}s")
    
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=50, blit=True
    )
    
    plt.tight_layout()
    
    # Save animation if requested

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_fname = os.path.join(save_dir, f"drum_dot_merged_{user_start:.2f}_{user_end:.2f}.mp4")
    
    print(f"\nSaving animation to: {save_fname}")
    try:
        # Temporarily modify PATH to use system ffmpeg
        original_path = os.environ.get('PATH', '')
        ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
        
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_fname, writer=writer)
        plt.close(fig)
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {str(e)}")
        plt.close(fig)

    
    return anim