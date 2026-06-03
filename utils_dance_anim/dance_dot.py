import os
import numpy as np
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.stats import norm

# Use system ffmpeg which has libx264 support
FFMPEG_PATH = "/itf-fi-ml/home/sagardu/bin/ffmpeg"

def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'

def load_cycles(cycles_csv_path):
    df = pd.read_csv(cycles_csv_path)
    return df["Virtual Onset"].values

def load_dance_onsets(dance_csv_path):
    df = pd.read_csv(dance_csv_path)
    return df["time_sec"].values

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

def analyze_dance_phases_no_plot(cycles_csv_path, dance_csv_path, W_start, W_end, SIG=0.01):
    # Load data
    cycles = load_cycles(cycles_csv_path)
    all_onsets = load_dance_onsets(dance_csv_path)
    # Filter onsets by time window
    window_mask = (all_onsets >= W_start) & (all_onsets <= W_end)
    onsets = all_onsets[window_mask]
    if len(cycles) < 2 or len(onsets) == 0:
        return None, None, None, None
    cycle_indices, phases, valid_onsets = find_cycle_phases(onsets, cycles)
    if len(phases) == 0:
        return None, None, None, None
    window_positions = (valid_onsets - W_start) / (W_end - W_start)
    kde_xx, kde_h = kde_estimate(phases, SIG)
    return phases, window_positions, kde_xx, kde_h

def animate_dance_phase_analysis(
    file_name, 
    W_start, 
    W_end, 
    cycles_csv_path, 
    dance_csv_path,
    figsize=(10, 3), 
    dpi=100, 
    save_dir=None
    ):
    """
    Animate the phase analysis plot for dance onsets with a moving playhead.
    """
    print(f"Generating animation for dance dot plot for {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s")
    
    cycles = load_cycles(cycles_csv_path)
    phases, window_positions, kde_xx, kde_h = analyze_dance_phases_no_plot(
        cycles_csv_path, dance_csv_path, W_start, W_end
    )
    if phases is None:
        print("Not enough data for animation.")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    color = 'purple'
    kde_color = 'orange'

    # Plot scatter points (above zero)
    ax.scatter(phases, window_positions, alpha=0.5, color=color, s=5)

    # Scale KDE to be between -0.5 and 0, starting from bottom
    kde_scaled = -0.6 + (0.5 * kde_h / np.max(kde_h))
    ax.fill_between(kde_xx, -0.6, kde_scaled, alpha=0.3, color=kde_color)

    ax.set_xlabel('Beat Span')    # Normalized metric cycle# 
    ax.set_ylabel('Relative Position in Window')
    ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s | Onset: Dance')

    ax.set_xlim(-0.1, 1)
    xticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(xticks)  
    ax.set_xticklabels([1, 2, 3, 4, 5])
    
    ax.set_ylim(-0.65, 1.0)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ax.grid(True, alpha=0.3)
    
    # Draw vertical lines at each subdivision i/12
    ymin, ymax = ax.get_ylim()
    for subdiv  in range(1, 13):
        xpos = (subdiv - 1) / 12    # subdiv 1 → 0.0, subdiv 4 → 0.25, etc.
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), linestyle='--', linewidth=1, alpha=0.3)

    playhead, = ax.plot([0, 0], [-0.65, 1.0], color='orange', lw=1.5, alpha=0.7, linestyle='-')
    h_playhead, = ax.plot([0, 1], [0, 0], color='orange', lw=1.5, alpha=0.7, linestyle='-')

    def find_phase(t):
        idx = np.searchsorted(cycles, t)
        if idx == 0 or idx >= len(cycles):
            return None
        c = idx - 1
        L_c = cycles[c]
        L_c1 = cycles[c + 1]
        return (t - L_c) / (L_c1 - L_c)

    def update(frame):
        phase = find_phase(frame)
        if phase is not None:
            playhead.set_xdata([phase, phase])
            y_pos = (frame - W_start) / (W_end - W_start)
            h_playhead.set_ydata([y_pos, y_pos])
            ax.set_title(f'File: {file_name} | Window: {W_start:.0f}s - {W_end:.0f}s | Onset: Feet | Time: {frame:.0f}s')
        return playhead, h_playhead,

    frames = np.arange(W_start, W_end, 1/24)
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=50, blit=True
    )
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_filename = f"dance_{W_start:.2f}_{W_end:.2f}.mp4"
        save_path = os.path.join(save_dir, save_filename)
        print(f"\nSaving animation to: {save_path}")
        try:
            # Temporarily modify PATH to use system ffmpeg
            original_path = os.environ.get('PATH', '')
            ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
            
            writer = animation.FFMpegWriter(fps=24, bitrate=2000)
            anim.save(save_path, writer=writer)
            plt.close(fig)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {str(e)}")
            plt.close(fig)
    else:
        print("Error: save_dir must be provided")
        plt.close(fig)
    return anim

def save_dance_phase_plot(
    file_name, W_start, W_end, cycles_csv_path, dance_csv_path,
    figsize=(10, 3), dpi=100, save_dir=None, save_format='png'
):
    """
    Create and save a static plot of the dance phase analysis.
    
    Args:
        file_name (str): Name of the file being analyzed
        W_start (float): Start time of the analysis window
        W_end (float): End time of the analysis window
        cycles_csv_path (str): Path to the cycles CSV file
        dance_csv_path (str): Path to the dance onsets CSV file
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Dots per inch for the output image
        save_dir (str): Directory to save the plot
        save_format (str): Format to save the plot ('png', 'jpg', 'pdf', etc.)
    """
    phases, window_positions, kde_xx, kde_h = analyze_dance_phases_no_plot(
        cycles_csv_path, dance_csv_path, W_start, W_end
    )
    if phases is None:
        print("Not enough data for plotting.")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    color = 'purple'
    kde_color = 'orange'

    # Plot scatter points
    ax.scatter(phases, window_positions, alpha=0.5, color=color, s=5)

    # Scale KDE to be between -0.5 and 0, starting from bottom
    kde_scaled = -0.5 + (0.5 * kde_h / np.max(kde_h))
    ax.fill_between(kde_xx, -0.5, kde_scaled, alpha=0.3, color=kde_color)

    ax.set_xlabel('Normalized metric cycle')
    ax.set_ylabel('Relative Position in Window')
    ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s | Onset: Dance')

    ax.set_xlim(-0.1, 1)
    xticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(xticks)
    
    ax.set_ylim(-0.55, 1.0)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ax.grid(True, alpha=0.3)
    
    # Draw vertical lines at each subdivision i/12
    ymin, ymax = ax.get_ylim()
    for subdiv in range(1, 13):
        xpos = (subdiv - 1) / 12
        ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), linewidth=1)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_filename = f"{file_name}_dance_phase_plot.{save_format}"
        save_path = os.path.join(save_dir, save_filename)
        print(f"\nSaving plot to: {save_path}")
        try:
            plt.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print("Plot saved successfully!")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
            plt.close(fig)
    else:
        print("Error: save_dir must be provided")
        plt.close(fig)
        
        
        
######## Animate the dance dot plot with a moving playhead within a user-defined window #########


def animate_dance_phase_analysis_with_user_window(
    file_name, 
    W_start,           # Full window start
    W_end,             # Full window end
    user_start,        # Animation start time
    user_end,          # Animation end time
    cycles_csv_path, 
    dance_csv_path,
    figsize=(10, 3), 
    dpi=100, 
    save_dir=None,
    fps=24,
    ):
    """
    Animate the phase analysis plot for dance onsets with a moving playhead within a user-defined window.
    
    Args:
        file_name: Base name of the file to analyze
        W_start: Start of full analysis window
        W_end: End of full analysis window
        user_start: Start time for animation playhead
        user_end: End time for animation playhead
        cycles_csv_path: Path to the cycles CSV file
        dance_csv_path: Path to the dance CSV file
        figsize: Figure size in inches
        dpi: Dots per inch for the figure
        save_dir: Directory to save the animation in
    """
    # Validate user window is within full window
    if not (W_start <= user_start <= user_end <= W_end):
        raise ValueError("User window must be within the full window range")

    print(f"Generating animation for dance dot plot for {file_name}")
    print(f"Full window: {W_start:.1f}s - {W_end:.1f}s")
    print(f"Animation window: {user_start:.1f}s - {user_end:.1f}s")
    
    cycles = load_cycles(cycles_csv_path)
    phases, window_positions, kde_xx, kde_h = analyze_dance_phases_no_plot(
        cycles_csv_path, dance_csv_path, W_start, W_end
    )
    if phases is None:
        print("Not enough data for animation.")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    color = 'purple'
    kde_color = 'purple'

    # Plot scatter points (above zero)
    ax.scatter(phases, window_positions, alpha=0.5, color=color, s=14)   # Scatter plot

    # Scale KDE to be between -0.5 and 0, starting from bottom
    kde_scaled = -0.6 + (0.5 * kde_h / np.max(kde_h))
    ax.fill_between(kde_xx, -0.6, kde_scaled, alpha=0.3, color=kde_color)

    ax.set_xlabel('4-beat cycle span')
    ax.set_ylabel('Relative Position in Window')
    ax.set_title(f'File: {file_name} | Window: {W_start:.1f}s - {W_end:.1f}s | Onset: Dance')

    ax.set_xlim(-0.1, 1)
    xticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(xticks)  
    ax.set_xticklabels([1, 2, 3, 4, 5])
    
    ax.set_ylim(-0.65, 1.0)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])  # comment to show y-axis labels
    
    # ax.grid(True, alpha=0.3)
    
    # Draw vertical lines at each subdivision i/12
    ymin, ymax = ax.get_ylim()
    for subdiv in range(1, 13):
        xpos = (subdiv - 1) / 12
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), 
                     linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), 
                     linestyle='--', linewidth=1, alpha=0.3)

    playhead, = ax.plot([0, 0], [-0.65, 1.0], color='orange', lw=1.5, alpha=0.7, linestyle='-')
    h_playhead, = ax.plot([0, 1], [0, 0], color='orange', lw=1.5, alpha=0.7, linestyle='-')

    def find_phase(t):
        idx = np.searchsorted(cycles, t)
        if idx == 0 or idx >= len(cycles):
            return None
        c = idx - 1
        L_c = cycles[c]
        L_c1 = cycles[c + 1]
        return (t - L_c) / (L_c1 - L_c)

    def update(frame):
        phase = find_phase(frame)
        if phase is not None:
            # Update vertical playhead
            playhead.set_xdata([phase, phase])
            
            # Update horizontal playhead (normalized to full window)
            y_pos = (frame - W_start) / (W_end - W_start)
            h_playhead.set_ydata([y_pos, y_pos])
            
            # Update title with both windows and current time
            ax.set_title(
                f'File: {file_name} | Window: {user_start:.0f}s - {user_end:.0f}s | Onset: Feet | Time: {frame:.0f}s'
                # f'File: {file_name}\n'
                # f'Full window: {W_start:.1f}s - {W_end:.1f}s\n'
                # f'Animation: {user_start:.1f}s - {user_end:.1f}s | Time: {frame:.2f}s'
            )
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

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_filename = f"dance_{user_start:.2f}_{user_end:.2f}.mp4"
        save_path = os.path.join(save_dir, save_filename)
        print(f"\nSaving animation to: {save_path}")
        try:
            # Temporarily modify PATH to use system ffmpeg
            original_path = os.environ.get('PATH', '')
            ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
            
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(save_path, writer=writer)
            plt.close(fig)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {str(e)}")
            plt.close(fig)
    else:
        print("Error: save_dir must be provided")
        plt.close(fig)
    return anim

