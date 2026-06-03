import os
import pickle
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import subprocess

# Use system ffmpeg which has libx264 support
FFMPEG_PATH = "/itf-fi-ml/home/sagardu/bin/ffmpeg"


############### Compute windows ###############

def find_and_load_pickle(traj_dir, base_name, status, value):
    """
    Search for a .pkl file in `traj_dir` matching: <base_name>_<status>_<value>.pkl
    Returns the loaded pickle contents or raises FileNotFoundError.
    """
    value_str = str(value)
    target_filename = f"{base_name}_{status}_{value_str}.pkl"
    target_path = os.path.join(traj_dir, target_filename)

    if os.path.isfile(target_path):
        with open(target_path, "rb") as f:
            return pickle.load(f)

    # Fallback: scan all .pkl files for a match
    for fname in os.listdir(traj_dir):
        if not fname.lower().endswith(".pkl"):
            continue
        name_no_ext = fname[:-4]  # remove ".pkl"
        parts = name_no_ext.split("_")
        if len(parts) >= 3 and parts[-2] == status and parts[-1] == value_str:
            candidate_base = "_".join(parts[:-2])
            if candidate_base == base_name:
                full_path = os.path.join(traj_dir, fname)
                with open(full_path, "rb") as f:
                    return pickle.load(f)

    raise FileNotFoundError(f"No pickle file matching '{base_name}_{status}_{value_str}.pkl' in '{traj_dir}'")

def compute_windows(traj_dir, base_name, status, value):
    """
    Load cycle_segments from the specified pickle, then compute
    ±1-cycle windows for each beat and subdivision POI.
    
    Returns:
        cycle_segments: list of (start, end) tuples
        windows: dict mapping POI names to lists of (win_start, win_end, t_poi)
    """
    # 1. Load cycle_segments
    cycle_segments = find_and_load_pickle(traj_dir, base_name, status, value)
    print(f"Loaded cycle_segments from {base_name}_{status}_{value}.pkl")

    # 2. Initialize the windows dictionary
    windows = {
        "beat_1": [], "beat_2": [], "beat_3": [], "beat_4": [],
        "subdiv_2": [], "subdiv_3": [], "subdiv_5": [], "subdiv_6": [],
        "subdiv_8": [], "subdiv_9": [], "subdiv_11": [], "subdiv_12": []
    }

    # 3. Define relative positions inside each cycle
    beat_positions = {
        "beat_1": 0.00,
        "beat_2": 0.25,
        "beat_3": 0.50,
        "beat_4": 0.75
    }
    subdiv_positions = {
        "subdiv_2":  1/12, "subdiv_3":  2/12,
        "subdiv_5":  4/12, "subdiv_6":  5/12,
        "subdiv_8":  7/12, "subdiv_9":  8/12,
        "subdiv_11": 10/12, "subdiv_12": 11/12
    }

    # 4. Process each cycle
    for (current_cycle_start, current_cycle_end) in cycle_segments:
        cycle_duration = current_cycle_end - current_cycle_start
        if cycle_duration <= 0:
            continue

        # Beats
        for beat, rel_pos in beat_positions.items():
            t_poi = current_cycle_start + rel_pos * cycle_duration
            win_start = t_poi - cycle_duration
            win_end = t_poi + cycle_duration
            windows[beat].append((win_start, win_end, t_poi))

        # Subdivisions
        for subdiv, rel_pos in subdiv_positions.items():
            t_poi = current_cycle_start + rel_pos * cycle_duration
            win_start = t_poi - cycle_duration
            win_end = t_poi + cycle_duration
            windows[subdiv].append((win_start, win_end, t_poi))

    # 5. Debug printout
    print("cycle_segments:", cycle_segments)
    print("Number of windows for each beat/subdivision:")
    for key, vals in windows.items():
        print(f"  {key}: {len(vals)} windows")

    return cycle_segments, windows

########################################################################

############### Extract cycle videos and plots ###############

def get_subdiv_color(subdiv_num):
    """Get color for a subdivision based on its number"""
    # For 3 subdivisions per beat
    if subdiv_num % 3 == 1:  # 1, 4, 7, 10
        return 'black'
    elif subdiv_num % 3 == 2:  # 2, 5, 8, 11
        return 'green'
    else:  # 3, 6, 9, 12
        return 'red'


def extract_centered_cycle_videos_and_plots(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    window_key: str,
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,  # Trajectory data frame rate
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    save_dir: str = "cycle_videos",
    figsize: tuple = (10, 3),
    dpi: int = 200
):
    """
    Extract video segments and create corresponding trajectory animations
    for windows around points of interest (beats or subdivisions).
    Each video/plot shows [-cycle, 0-cycle, +cycle] around the POI.
    """
    # Print windows data for debugging
    
    video_path = f"data/videos/{file_name}_pre_R_Mix.mp4"
    
    print("Windows data:")
    for i, (win_start, win_end, t_poi) in enumerate(windows):
        print(f"Window {i+1}:")
        print(f"  Start: {win_start:.3f}")
        print(f"  End: {win_end:.3f}")
        print(f"  POI: {t_poi:.3f}")
        print(f"  Duration: {win_end - win_start:.3f}")

    # Create save directories
    rec_dir = os.path.join(save_dir, file_name)
    window_key_dir = os.path.join(rec_dir, window_key)
    
    video_dir = os.path.join(window_key_dir, "videos")
    plot_dir = os.path.join(window_key_dir, "plots")
    
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(window_key_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Build file paths for foot data
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T", "onset_info")
    left_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_onsets.csv")
    right_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_onsets.csv")
    left_zpos_csv = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_zpos.csv")
    right_zpos_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_zpos.csv")
    
    # Load foot data
    left_df = pd.read_csv(left_onsets_csv)
    right_df = pd.read_csv(right_onsets_csv)
    
    # Debug prints for foot data
    print("\nFoot data ranges:")
    print(f"Left foot time range: {left_df['time_sec'].min():.3f} to {left_df['time_sec'].max():.3f}")
    print(f"Right foot time range: {right_df['time_sec'].min():.3f} to {right_df['time_sec'].max():.3f}")
    print(f"Number of left foot onsets: {len(left_df)}")
    print(f"Number of right foot onsets: {len(right_df)}")
    
    # Load trajectory data
    Lz = pd.read_csv(left_zpos_csv)["zpos"].values
    Rz = pd.read_csv(right_zpos_csv)["zpos"].values
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate  # Times at 240fps
    
    print(f"\nProcessing {len(windows)} windows")
    print(f"Total frames in trajectory data: {n_frames}")
    print(f"Time range in trajectory data: {times[0]:.3f} to {times[-1]:.3f}")
    
    # Process each window
    for i, (win_start, win_end, t_poi) in enumerate(windows):
        print(f"\nProcessing window {i+1}:")
        print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
        
        # Calculate segment times
        start_time = win_start
        end_time = win_end
        duration = end_time - start_time
        downbeat = t_poi  # This is the point of interest (beat or subdivision)
        
        # Calculate avg_cycle from the window duration
        avg_cycle = duration / 2  # Since window is ±1 cycle
        
        # Calculate window parameters
        beat_len = avg_cycle / n_beats_per_cycle
        subdiv_len = beat_len / n_subdiv_per_beat
        half_win = subdiv_len * nn
        
        # Get foot onsets for this window
        left_times = left_df[(left_df["time_sec"]>=win_start)&(left_df["time_sec"]<=win_end)]["time_sec"].values
        right_times = right_df[(right_df["time_sec"]>=win_start)&(right_df["time_sec"]<=win_end)]["time_sec"].values
        
        print(f"  Found {len(left_times)} left foot onsets and {len(right_times)} right foot onsets")
        if len(left_times) > 0:
            print(f"  Left foot onset times: {left_times}")
        if len(right_times) > 0:
            print(f"  Right foot onset times: {right_times}")
        
        # Calculate frame numbers for video (50fps)
        video_start_frame = int(start_time * 50)
        video_end_frame = int(end_time * 50)
        video_n_frames = video_end_frame - video_start_frame
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        traj_n_frames = traj_end_frame - traj_start_frame
        
        print(f"  Video frames: {video_start_frame} to {video_end_frame} (50fps)")
        print(f"  Trajectory frames: {traj_start_frame} to {traj_end_frame} (240fps)")
        
        # Check if we have valid frame numbers
        if traj_start_frame >= traj_end_frame:
            print(f"  Skipping window {i+1}: Invalid frame range (start >= end)")
            continue
        if traj_start_frame < 0:
            print(f"  Skipping window {i+1}: Start frame < 0")
            continue
        if traj_end_frame > len(Lz):
            print(f"  Skipping window {i+1}: End frame > total frames")
            continue
        
        # Trim trajectory data using frame numbers at 240fps
        L_win = Lz[traj_start_frame:traj_end_frame]
        R_win = Rz[traj_start_frame:traj_end_frame]
        t_win = times[traj_start_frame:traj_end_frame]
        
        # Check if we have valid trajectory data
        if len(L_win) == 0 or len(R_win) == 0:
            print(f"  Skipping window {i+1}: No trajectory data")
            continue
        
        print(f"  Trajectory data points: {len(L_win)}")
        
        # Extract video segment with audio using ffmpeg
        video_output_path = os.path.join(video_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        ffmpeg_cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            "-r", "24",
            video_output_path
        ]
        # subprocess.run(ffmpeg_cmd, capture_output=True)
        
        # Print the command and paths for debugging
        print(f"\nVideo extraction:")
        print(f"Input video: {video_path}")
        print(f"Output video: {video_output_path}")
        print(f"Start time: {start_time}")
        print(f"Duration: {duration}")
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        # Run ffmpeg and capture output
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting video:")
            print(f"Return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
        else:
            print("Video extraction successful")
            # Verify the file was created
            if os.path.exists(video_output_path):
                print(f"Output file exists: {video_output_path}")
                print(f"File size: {os.path.getsize(video_output_path)} bytes")
            else:
                print("Warning: Output file was not created")
        
        
        #######################################################################
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.tight_layout(pad=2.0) 
        
        # Calculate all subdivision times for the window
        all_subdiv_times = []
        for beat_idx in range(-n_beats_per_cycle, n_beats_per_cycle + 1):
            beat_time = downbeat + beat_idx * beat_len
            for subdiv_idx in range(n_subdiv_per_beat):
                subdiv_time = beat_time + subdiv_idx * subdiv_len
                if start_time <= subdiv_time <= end_time:
                    all_subdiv_times.append((subdiv_time, beat_idx * n_subdiv_per_beat + subdiv_idx + 1))

        # Plot subdivision lines with appropriate colors
        for subdiv_time, subdiv_num in all_subdiv_times:
            color = get_subdiv_color(subdiv_num)
            # Add yellow glow effect for t=0
            if abs(subdiv_time - downbeat) < 0.001:  # If it's the POI
                ax.axvline(subdiv_time, color='yellow', linestyle='-', linewidth=3, alpha=0.3)
            
            if subdiv_num in [-11, -8, -5, -2, 1, 4, 7, 10, 13]:
                ax.axvline(subdiv_time, color=color, linestyle='-', linewidth=1.5, alpha=0.7) #beat color
            else:
                ax.axvline(subdiv_time, color=color, linestyle='--', linewidth=1, alpha=0.3) #subdivision color
        
        # Plot trajectories
        ax.plot(t_win, L_win, '-', color='blue', alpha=0.5, label='Left Foot')
        ax.plot(t_win, R_win, '-', color='red', alpha=0.5, label='Right Foot')
        
        # Plot foot onset markers
        for onset in left_times:
            idx = np.argmin(np.abs(t_win - onset))
            ax.plot(onset, L_win[idx], 'o', color='blue', ms=6, alpha=0.8)
        
        for onset in right_times:
            idx = np.argmin(np.abs(t_win - onset))
            ax.plot(onset, R_win[idx], 'o', color='red', ms=6, alpha=0.8)
        
        ax.axvline(downbeat, color='yellow', linewidth=6, alpha=0.3, zorder=0)  # Glow effect
        
        # Set y-axis limits with safety checks
        try:
            y_min = min(L_win.min(), R_win.min())
            y_max = max(L_win.max(), R_win.max())
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        except ValueError as e:
            print(f"  Warning: Could not set y-axis limits: {e}")
            # Set default y-axis limits
            ax.set_ylim(-1, 1)
        
        # Create vertical playhead
        v_playhead, = ax.plot([start_time, start_time], 
                            [y_min - 0.1*y_range, y_max + 0.1*y_range],
                            lw=1.5, alpha=0.9, color='orange')
        
        # Set up the plot with scaled x-axis
        ax.set_xlabel(f'Beats relative to {window_key}')
        ax.set_ylabel('Foot Position')
        ax.set_title(f'Window {i+1} | {window_key}: {downbeat:.2f}s')
        ax.grid(True, alpha=0.3)
        
        # Scale x-axis to show beats instead of cycles
        x_ticks = np.arange(-n_beats_per_cycle, n_beats_per_cycle + 1)
        x_tick_positions = downbeat + x_ticks * beat_len
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_ticks)
        
        # Add legend
        custom = [
            Line2D([0],[0], color='blue', linestyle='-', lw=1),
            Line2D([0],[0], color='red', linestyle='-', lw=1),
            Line2D([0],[0], color='black', lw=1.5),
            Line2D([0],[0], color='green', linestyle='--', lw=1),
            Line2D([0],[0], color='red', linestyle='--', lw=1),
        ]
        labels = ["Left Foot", "Right Foot", "Subdiv-1 (1,4,7,10)", "Subdiv-2 (2,5,8,11)", "Subdiv-3 (3,6,9,12)"]
        ax.legend(custom, labels, loc='upper left', framealpha=0.3, fontsize=6)
        
        def update(frame):
            v_playhead.set_xdata([frame, frame])
            ax.set_title(f'Cycle {i+1} | {window_key}: {downbeat:.2f}s | Time: {frame:.2f}s')
            return v_playhead,
        
        # Create animation frames at 24fps  
        plot_n_frames = int((end_time - start_time) * 24)
        frames = np.linspace(start_time, end_time, plot_n_frames)
        anim = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=1000/24,  # 50fps
            blit=True
        )
        
        # Save animation
        plot_output_path = os.path.join(plot_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        writer = animation.FFMpegWriter(fps=24, bitrate=2000)  # 50fps to match video
        anim.save(plot_output_path, writer=writer)
        plt.close(fig)
        
        print(f"  Video saved: {video_output_path}")
        print(f"  Plot saved: {plot_output_path}")
        print(f"  Video duration: {duration:.3f}s")
        print(f"  Plot duration: {len(frames)/24:.3f}s")
        # break
    print("\nProcessing complete!")


def extract_forward_cycle_videos_and_plots(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    # window_key: str,
    base_path_logs: str = "data/dance_onsets_v4_0.007_foot_jun3",
    frame_rate: float = 240,  # Trajectory data frame rate
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    save_dir: str = "cycle_videos",
    figsize: tuple = (10, 3),
    dpi: int = 200,
    legend_flag: bool = True,
    fps= 24,    # original video fps is 50fps
    skip_existing: bool = False,
):
    """
    Extract video segments and create corresponding trajectory animations
    for windows around points of interest (beats or subdivisions).
    Each video/plot shows [0-cycle, +cycle] around the POI.
    """
    # Print windows data for debugging
    video_path = f"data/videos/{file_name}_pre_R_Mix.mp4"
    
    print("Windows data:")
    for i, (win_start, win_end, t_poi) in enumerate(windows):
        print(f"Window {i+1}:")
        print(f"  Start: {win_start:.3f}")
        print(f"  End: {win_end:.3f}")
        print(f"  Duration: {win_end - win_start:.3f}")

    # Create save directories
    # rec_dir = os.path.join(save_dir, file_name, f"_{start_time:.2f}_{end_time:.2f}")
    # window_key_dir = os.path.join(rec_dir, window_key)
    
    video_dir = os.path.join(save_dir, "videos")     # window_key_dir
    plot_dir = os.path.join(save_dir, "plots")     # window_key_dir
    
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(window_key_dir, exist_ok=True)     # window_key_dir
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Build file paths for foot data
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T", "onset_info")
    left_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_onsets.csv")
    right_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_onsets.csv")
    left_zpos_csv = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_zpos.csv")
    right_zpos_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_zpos.csv")
    
    # Load foot data
    left_df = pd.read_csv(left_onsets_csv)
    right_df = pd.read_csv(right_onsets_csv)
    
    # Debug prints for foot data
    print("\nFoot data ranges:")
    print(f"Left foot time range: {left_df['time_sec'].min():.3f} to {left_df['time_sec'].max():.3f}")
    print(f"Right foot time range: {right_df['time_sec'].min():.3f} to {right_df['time_sec'].max():.3f}")
    print(f"Number of left foot onsets: {len(left_df)}")
    print(f"Number of right foot onsets: {len(right_df)}")
    
    # Load trajectory data
    Lz = pd.read_csv(left_zpos_csv)["zpos"].values
    Rz = pd.read_csv(right_zpos_csv)["zpos"].values
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate  # Times at 240fps
    
    print(f"\nProcessing {len(windows)} windows")
    print(f"Total frames in trajectory data: {n_frames}")
    print(f"Time range in trajectory data: {times[0]:.3f} to {times[-1]:.3f}")
    
    # Process each window
    for i, (win_start, win_end, t_poi) in enumerate(windows):
        print(f"\nProcessing window {i+1}:")
        print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
        
        # Calculate segment times
        start_time = win_start  # This will be our reference point (beat 1)
        end_time = win_end
        duration = end_time - start_time
        
        # Calculate avg_cycle from the window duration
        avg_cycle = duration  # Since window is now 0 to +1 cycle
        
        # Calculate window parameters
        beat_len = avg_cycle / n_beats_per_cycle
        subdiv_len = beat_len / n_subdiv_per_beat
        
        # Get foot onsets for this window
        left_times = left_df[(left_df["time_sec"]>=win_start)&(left_df["time_sec"]<=win_end)]["time_sec"].values
        right_times = right_df[(right_df["time_sec"]>=win_start)&(right_df["time_sec"]<=win_end)]["time_sec"].values
        
        print(f"  Found {len(left_times)} left foot onsets and {len(right_times)} right foot onsets")
        if len(left_times) > 0:
            print(f"  Left foot onset times: {left_times}")
        if len(right_times) > 0:
            print(f"  Right foot onset times: {right_times}")
        
        # Calculate frame numbers for video (50fps)
        video_start_frame = int(start_time * fps)
        video_end_frame = int(end_time * fps)
        video_n_frames = video_end_frame - video_start_frame
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        traj_n_frames = traj_end_frame - traj_start_frame
        
        print(f"  Video frames: {video_start_frame} to {video_end_frame} ({fps}fps)")
        print(f"  Trajectory frames: {traj_start_frame} to {traj_end_frame} (240fps)")
        
        # Check if we have valid frame numbers
        if traj_start_frame >= traj_end_frame:
            print(f"  Skipping window {i+1}: Invalid frame range (start >= end)")
            continue
        if traj_start_frame < 0:
            print(f"  Skipping window {i+1}: Start frame < 0")
            continue
        if traj_end_frame > len(Lz):
            print(f"  Skipping window {i+1}: End frame > total frames")
            continue
        
        # Trim trajectory data using frame numbers at 240fps
        L_win = Lz[traj_start_frame:traj_end_frame] * 100    # *100 to make it cm
        R_win = Rz[traj_start_frame:traj_end_frame] * 100    # *100 to make it cm
        t_win = times[traj_start_frame:traj_end_frame]
        
        # Check if we have valid trajectory data
        if len(L_win) == 0 or len(R_win) == 0:
            print(f"  Skipping window {i+1}: No trajectory data")
            continue
        
        print(f"  Trajectory data points: {len(L_win)}")
        
        # Extract video segment with audio using ffmpeg
        video_output_path = os.path.join(video_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        plot_output_path = os.path.join(plot_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        if skip_existing and os.path.exists(video_output_path) and os.path.exists(plot_output_path):
            print(f"  Skipping window {i+1}: video and plot already exist")
            continue

        ffmpeg_cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            "-r", str(fps),
            video_output_path
        ]
        
        # Print the command and paths for debugging
        print(f"\nVideo extraction:")
        print(f"Input video: {video_path}")
        print(f"Output video: {video_output_path}")
        print(f"Start time: {start_time}")
        print(f"Duration: {duration}")
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        # Run ffmpeg and capture output
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting video:")
            print(f"Return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
        else:
            print("Video extraction successful")
            # Verify the file was created
            if os.path.exists(video_output_path):
                print(f"Output file exists: {video_output_path}")
                print(f"File size: {os.path.getsize(video_output_path)} bytes")
            else:
                print("Warning: Output file was not created")
        
        #######################################################################
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.tight_layout(pad=3.0) 
        
        # Calculate all subdivision times for the window
        all_subdiv_times = []
        for beat_idx in range(0, n_beats_per_cycle + 1):
            beat_time = start_time + beat_idx * beat_len
            for subdiv_idx in range(n_subdiv_per_beat):
                subdiv_time = beat_time + subdiv_idx * subdiv_len
                if start_time <= subdiv_time <= end_time:
                    all_subdiv_times.append((subdiv_time, beat_idx * n_subdiv_per_beat + subdiv_idx + 1))

        # Plot subdivision lines with appropriate colors
        for subdiv_time, subdiv_num in all_subdiv_times:
            color = get_subdiv_color(subdiv_num)
            if subdiv_num in [1, 4, 7, 10, 13]:
                ax.axvline(subdiv_time, color=color, linestyle='-', linewidth=1.5, alpha=0.7) #beat color
            else:
                ax.axvline(subdiv_time, color=color, linestyle='--', linewidth=1, alpha=0.3) #subdivision color
        
        # Plot trajectories
        ax.plot(t_win, L_win, '-', color='blue', alpha=0.5, label='Left Foot', linewidth=3.5)   
        ax.plot(t_win, R_win, '-', color='red', alpha=0.5, label='Right Foot', linewidth=3.5)  
        
        # Plot foot onset markers
        for onset in left_times:
            idx = np.argmin(np.abs(t_win - onset))
            ax.plot(onset, L_win[idx], 'o', color='blue', ms=10, alpha=0.8, markeredgecolor='k',zorder=3)
        
        for onset in right_times:
            idx = np.argmin(np.abs(t_win - onset))
            ax.plot(onset, R_win[idx], 'o', color='red', ms=10, alpha=0.8, markeredgecolor='k',zorder=3)
        
        # Set y-axis limits with safety checks
        try:
            y_min = min(L_win.min(), R_win.min())
            y_max = max(L_win.max(), R_win.max())
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        except ValueError as e:
            print(f"  Warning: Could not set y-axis limits: {e}")
            # Set default y-axis limits
            ax.set_ylim(-1, 1)
        
        # Create vertical playhead
        v_playhead, = ax.plot([start_time, start_time], 
                            [y_min - 0.1*y_range, y_max + 0.1*y_range],
                            lw=1.5, alpha=0.9, color='orange')
        
        # Set up the plot with scaled x-axis
        ax.set_xlabel(f'Beats span')
        ax.set_ylabel('Feet (cm)', fontsize=12)
        ax.set_title(f'{file_name} | Window:{start_time:.0f}s - {end_time:.0f}s: {start_time:.0f}s')
        ax.grid(True, alpha=0.3)
        
        # Scale x-axis to show beats instead of cycles
        x_ticks = np.arange(1, n_beats_per_cycle + 2)
        x_tick_positions = start_time + (x_ticks-1) * beat_len
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(start_time, end_time)
        
        # Add legend
        custom = [
            Line2D([0],[0], color='blue', linestyle='-', lw=1),
            Line2D([0],[0], color='red', linestyle='-', lw=1),
            Line2D([0],[0], color='black', lw=1.5),
            Line2D([0],[0], color='green', linestyle='--', lw=1),
            Line2D([0],[0], color='red', linestyle='--', lw=1),
        ]
        labels = ["Left Foot", "Right Foot", "Subdiv-1 (1,4,7,10)", "Subdiv-2 (2,5,8,11)", "Subdiv-3 (3,6,9,12)"]
        
        if legend_flag:
            ax.legend(custom, labels, loc='upper left', framealpha=0.3, fontsize=6)
        
        def update(frame):
            v_playhead.set_xdata([frame, frame])
            ax.set_title(f'{file_name} | Window:{start_time:.0f}s - {end_time:.0f}s | Time: {frame:.0f}s')
            return v_playhead,
        
        # Create animation frames at 24fps  
        # plot_n_frames = int((end_time - start_time) * 24)
        # frames = np.linspace(start_time, end_time, plot_n_frames)
        
        frames = np.arange(start_time, end_time, 1/fps)   # New 06 June 2025
        anim = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=1000/fps,  # 24fps
            blit=True
        )
        
        # Save animation
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(plot_output_path, writer=writer)
        plt.close(fig)
        
        print(f"  Video saved: {video_output_path}")
        print(f"  Plot saved: {plot_output_path}")
        print(f"  Video duration: {duration:.3f}s")
        print(f"  Plot duration: {len(frames)/fps:.3f}s")
        
    print("\nProcessing complete!")





