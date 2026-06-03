import os
import shutil
import pickle
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# Use system ffmpeg which has libx264 support
FFMPEG_PATH = "/itf-fi-ml/home/sagardu/bin/ffmpeg"

# Raw BVH→worldpos CSVs (see mocap_visualizer_base.dir_csv)
DIR_CSV = "bvh_to_csv"


def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'


def extract_kinematic_cycle_plots(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    joint_name: str,
    axis: str = 'y',
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,  # Trajectory data frame rate
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    output_dir2: str = None,
    figsize: tuple = (10, 3),
    dpi: int = 200,
    legend_flag: bool = True,
    fps=24,
    skip_existing: bool = False,
):
    """
    Create trajectory animations for windows around points of interest (beats or subdivisions).
    Each plot shows [-cycle, 0-cycle, +cycle] around the POI.
    """
    # Create save directory if not provided
    # if output_dir2 is None:
    #     output_dir2 = os.path.join("cycle_plots", file_name, window_key, joint_name)
    #     os.makedirs(output_dir2, exist_ok=True)
    
    bvh_to_mvnx = {
    'x': 'y',  # BVH side → MVNX side
    'y': 'z',  # BVH vertical → MVNX vertical
    'z': 'x',  # BVH forward → MVNX forward
    }
    
    
    # Load joint position data
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    worldpos_file = os.path.join(DIR_CSV, f"{base_name}_T_worldpos.csv")
    
    try:
        world_positions = pd.read_csv(worldpos_file)
        print(f"Successfully loaded CSV with {len(world_positions)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    # Get time column and position data
    time_column = world_positions.columns[0]  # First column is time
    times = world_positions[time_column].values
    positions = world_positions[f"{joint_name}.{axis.upper()}"].values
    
    print(f"\nProcessing {len(windows)} windows")
    # print(f"Total frames in trajectory data: {len(times)}")
    # print(f"Time range in trajectory data: {times[0]:.3f} to {times[-1]:.3f}")
    
    # Process each window or cycle
    for i, (win_start, win_end, _) in enumerate(windows):  # Removed t_poi
        print(f"\nProcessing window {i+1}:")
        print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
        
        # Calculate segment times
        start_time = win_start
        end_time = win_end
        duration = end_time - start_time
        
        # Calculate window parameters
        beat_len = duration / n_beats_per_cycle
        subdiv_len = beat_len / n_subdiv_per_beat
        half_win = subdiv_len * nn
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        traj_n_frames = traj_end_frame - traj_start_frame
        
        print(f"  Trajectory frames: {traj_start_frame} to {traj_end_frame} (240fps)")
        
        # Check if we have valid frame numbers
        if traj_start_frame >= traj_end_frame:
            print(f"  Skipping window {i+1}: Invalid frame range (start >= end)")
            continue
        if traj_start_frame < 0:
            print(f"  Skipping window {i+1}: Start frame < 0")
            continue
        if traj_end_frame > len(positions):
            print(f"  Skipping window {i+1}: End frame > total frames")
            continue
        
        # Trim trajectory data using frame numbers at 240fps
        pos_win = positions[traj_start_frame:traj_end_frame]
        t_win = times[traj_start_frame:traj_end_frame]
        
        # Check if we have valid trajectory data
        if len(pos_win) == 0:
            print(f"  Skipping window {i+1}: No trajectory data")
            continue
        
        print(f"  Trajectory data points: {len(pos_win)}")
        
        plot_output_path = os.path.join(output_dir2, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        if skip_existing and os.path.exists(plot_output_path):
            print(f"  Skipping window {i+1}: {joint_name} plot already exists")
            continue

        # Create figure and axis ------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.tight_layout(pad=3.0) 
        
        # Calculate all subdivision times for the window
        all_subdiv_times = []
        for beat_idx in range(0, n_beats_per_cycle + 1):  # Changed: now starts from 0
            beat_time = start_time + beat_idx * beat_len  # Changed: use start_time instead of downbeat
            for subdiv_idx in range(n_subdiv_per_beat):
                subdiv_time = beat_time + subdiv_idx * subdiv_len
                if start_time <= subdiv_time <= end_time:
                    all_subdiv_times.append((subdiv_time, beat_idx * n_subdiv_per_beat + subdiv_idx + 1))

        # Plot subdivision lines with appropriate colors
        for subdiv_time, subdiv_num in all_subdiv_times:
            color = get_subdiv_color(subdiv_num)
            if subdiv_num in [1, 4, 7, 10, 13]:
                ax.axvline(subdiv_time, color=color, linestyle='-', linewidth=2, alpha=0.7) #beat color
            else:
                ax.axvline(subdiv_time, color=color, linestyle='--', linewidth=1, alpha=0.3) #subdivision color
        
        # Plot trajectory
        ax.plot(t_win, pos_win, '-', color='green', alpha=0.5, label=f'{joint_name} {axis.upper()}', linewidth=3.5)
        
        # Set y-axis limits with safety checks
        try:
            y_min = pos_win.min()
            y_max = pos_win.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        except ValueError as e:
            print(f"  Warning: Could not set y-axis limits: {e}")
            ax.set_ylim(-1, 1)
        
        # Create vertical playhead
        v_playhead, = ax.plot([start_time, start_time], 
                            [y_min - 0.1*y_range, y_max + 0.1*y_range],
                            lw=1.5, alpha=0.9, color='orange')
        
        # Set up the plot with scaled x-axis
        ax.set_xlabel(f'Beat span')
        ax.set_ylabel(f'Pelvis (cm)') 
        # ax.set_ylabel(f'{joint_name} {bvh_to_mvnx[axis.lower()]} Position')      # {axis.upper()} y is vertical in bvh files, Z is vertical in mocap
        ax.set_title(f'{file_name} | Window:{start_time:.0f}s - {end_time:.2f}| Time: {start_time:.0f}s')
        ax.grid(True, alpha=0.3)
        
        # Scale x-axis to show beats instead of cycles
        x_ticks = np.arange(1, n_beats_per_cycle + 2)  # Changed: now 1 to 5
        x_tick_positions = start_time + (x_ticks - 1) * beat_len  # Changed: use start_time and adjust for 1-based indexing
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(start_time, end_time)
        
        # Add legend
        custom = [
            Line2D([0],[0], color='green', lw=1.5),
            Line2D([0],[0], color='black', lw=1),
            Line2D([0],[0], color='green', lw=1, linestyle='--'),
            Line2D([0],[0], color='red', lw=1, linestyle='--'),
        ]
        labels = [
            f"{joint_name} {axis.upper()}", 
            "Subdiv-1 (1,4,7,10)", 
            "Subdiv-2 (2,5,8,11)", 
            "Subdiv-3 (3,6,9,12)"
        ]
        
        if legend_flag:
            ax.legend(custom, labels, loc='upper left', framealpha=0.3, fontsize=6)
        
        def update(frame):
            v_playhead.set_xdata([frame, frame])
            ax.set_title(f'{file_name} | Window:{start_time:.0f}s - {end_time:.0f}s| Time: {frame:.0f}s')
            return v_playhead,
        
        # Create animation frames at 24fps
        # n_frames = int(duration * 24)           # 
        # frames = np.linspace(start_time, end_time, n_frames)
        
        frames = np.arange(start_time, end_time, 1/fps)      # New 06 June 2025
        anim = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=1000/fps,  # 24fps
            blit=True
        )
        
        # Save animation
        # Temporarily modify PATH to use system ffmpeg
        original_path = os.environ.get('PATH', '')
        ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
        
        writer = animation.FFMpegWriter(fps= fps, 
                                        bitrate=2000,
                                        codec='libx264',  # Specify codec
                                        # extra_args=['-preset', 'ultrafast']
                                        )  # 24fps
        anim.save(plot_output_path, writer=writer)
        plt.close(fig)
        
        print(f"Plot saved: {plot_output_path}")
        print(f"Plot duration: {len(frames)/fps:.3f}s")
    
    print("\nProcessing complete!")


#----------------------------- Concatenate videos ---------------------------------
#----------------------------------------------------------------------------------

def extract_category(filename):
    """
    Given a filename like "front_view_56.7_61.2.mp4" or
    "BKO_E1_D1_02_Maraka_pre_R_Mix_trimmed_56.7_61.2.mp4",
    return the category portion before the last two underscore-separated tokens.
    """
    name, _ = os.path.splitext(filename)     # strip .mp4
    parts = name.split('_')
    # Last two parts are start and end times, so category is everything before them
    if len(parts) > 2:
        return "_".join(parts[:-2])
    return name   # fallback if unexpected format

def write_all_categories(files, output_dir, video_dir):
    """
    From a list of filenames, group by category (as defined by extract_category),
    and write each group into its own .txt file in output_dir.
    """
    # os.makedirs(output_dir, exist_ok=True)

    # Group filenames by category
    categories = {}
    for fname in files:
        cat = extract_category(fname)
        categories.setdefault(cat, []).append(fname)

    # Write each category's filenames to a separate text file
    for cat, fnames in categories.items():
        txt_path = os.path.join(output_dir, f"{cat}.txt")
        with open(txt_path, "w") as fw:
            for f in fnames:
                if video_dir:
                    rel_path = os.path.relpath(os.path.join(video_dir, f), os.path.dirname(txt_path))
                    fw.write(f"file '{rel_path}'\n")
                else:
                    fw.write(f + "\n")  
                

def create_concat_file(video_dir, output_file):
    """Create a text file listing all videos in order for concatenation"""
    with open(output_file, 'w') as f:
        # Get all video files and sort them
        video_files = os.listdir(video_dir)
        video_files_sorted = sorted(
                video_files,
                key=lambda x: float(x.split('_')[-2]),
                reverse=False
            )
        # Write each file path - use relative path from the text file location
        for video in video_files_sorted:
            # Get relative path from output_file to video_dir
            rel_path = os.path.relpath(os.path.join(video_dir, video), os.path.dirname(output_file))
            f.write(f"file '{rel_path}'\n") 
            
def concatenate_and_overlay_videos(joint_name,  save_dir, views_to_generate):
    """Concatenate cycle videos and plot videos, then overlay them"""
    video_dir = os.path.join(save_dir, "videos")
    plot_dir = os.path.join(save_dir, "plots")
    joint_dir = os.path.join(save_dir, joint_name)
    vid_skel_dir = os.path.join(save_dir, "video_skeleton")
    drum_dot_dir = os.path.join(save_dir, "drum_dot_merged")
    dance_dot_dir = os.path.join(save_dir, "dance_dot")

    # Create text files for concatenation
    video_list = os.path.join(save_dir, "video_list.txt")
    plot_list = os.path.join(save_dir, "plot_list.txt")
    joint_list = os.path.join(save_dir, "joint_list.txt")
    
    drum_dot_list = os.path.join(save_dir, "drum_dot_list.txt")
    dance_dot_list = os.path.join(save_dir, "dance_dot_list.txt")
    
    # 'front', 'right' 'left', 'top'
    if 'front' in views_to_generate:
        front_view_list = os.path.join(save_dir, "front_view.txt")
    if 'left' in views_to_generate:
        left_view_list = os.path.join(save_dir, "left_view.txt")
    if 'right' in views_to_generate:
        right_view_list = os.path.join(save_dir, "right_view.txt")
    if 'top' in views_to_generate:
        top_view_list = os.path.join(save_dir, "top_view.txt")
    
    
    # Check if directories exist
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return
    if not os.path.exists(plot_dir):
        print(f"Plot directory not found: {plot_dir}")
        return
        
    # Check if text files already exist
    if os.path.exists(video_list) and os.path.exists(plot_list):
        print("Concatenation files already exist, skipping creation")
    else:
        print("Creating concatenation files...")
        create_concat_file(vid_skel_dir, front_view_list)
        create_concat_file(video_dir, video_list)
        create_concat_file(plot_dir, plot_list)
        create_concat_file(joint_dir, joint_list)
        create_concat_file(drum_dot_dir, drum_dot_list)
        create_concat_file(dance_dot_dir, dance_dot_list)
    
    
    
    concatenate_videos(video_list, save_dir, f"video_mix_concat")
    concatenate_videos(plot_list, save_dir, f"plot_concat")
    concatenate_videos(joint_list, save_dir, f"joint_{joint_name}_concat")
    
    concatenate_videos(drum_dot_list, save_dir, f"drum_dot_concat")
    concatenate_videos(dance_dot_list, save_dir, f"dance_dot_concat")
    
    if 'front' in views_to_generate:    
        concatenate_videos(front_view_list, save_dir, f"front_view_concat")
    if 'left' in views_to_generate:
        concatenate_videos(left_view_list, save_dir, f"left_view_concat")
    if 'right' in views_to_generate:
        concatenate_videos(right_view_list, save_dir, f"right_view_concat")
    if 'top' in views_to_generate:
        concatenate_videos(top_view_list, save_dir, f"top_view_concat") 

    
    print(f"Concatenation complete: {save_dir}")

def concatenate_videos(video_list, save_dir, save_name):
        
        concat_video = os.path.join(save_dir, f"{save_name}.mp4")
        try:
            result = subprocess.run([
                FFMPEG_PATH, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', video_list,
                '-c', 'copy',
                concat_video
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print("Error concatenating videos:", result.stderr)
                return
        except Exception as e:
            print("Error running ffmpeg:", str(e))
            return

#----------------------------- Resize video ----------------------------------------
#----------------------------------------------------------------------------------
def resize_video(video_path, width, height, save_dir):
    """
    Resize a video to the specified width and height using ffmpeg,
    with debug‐level output.

    Parameters:
    - video_path: str, path to the input video file
    - width: int, target width in pixels
    - height: int, target height in pixels
    - save_dir: str, directory where the resized video will be saved

    The output filename will be: <original_basename>_<width>x<height>.mp4
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Derive output filename from input basename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}.mp4"   # f"{base_name}_{width}x{height}.mp4"
    output_path = os.path.join(save_dir, output_filename)

    # Build ffmpeg command with debug-level logging
    cmd = [
        FFMPEG_PATH,
        "-y",                   # overwrite output if it exists
        "-loglevel", "debug",   # show full debug output
        "-i", video_path,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        output_path
    ]

    # Run ffmpeg and capture stdout/stderr
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print debug output
    # print("=== ffmpeg stdout ===")
    # print(result.stdout)
    # print("=== ffmpeg stderr ===")
    # print(result.stderr)

    # Check return code and report
    if result.returncode == 0:
        print(f"Resizing succeeded, output saved to: {output_path}")
    else:
        print(f"ffmpeg failed with return code {result.returncode}")

    return output_path if result.returncode == 0 else None


#----------------------------- Create composite video -----------------------------
#----------------------------------------------------------------------------------

# def create_composite_video(composite_video_elements, final_out, output_fps=24):

#     video_positions = []
#     for element in composite_video_elements:
#         video_positions.append({
#             'path': element['vid_path'],
#             'x': element['x_pos_pxl'],
#             'y': element['y_pos_pxl']
#         })

#     # Build the ffmpeg command
#     ffmpeg_inputs = []
#     for pos in video_positions:
#         ffmpeg_inputs.extend(['-i', pos['path']])

#     # Create the xstack layout string
#     # Format: xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0
#     layout = []
#     for pos in video_positions:
#         layout.append(f"{pos['x']}_{pos['y']}")

#     xstack_layout = "|".join(layout)

#     # final_out = os.path.join(base_output_dir, f"{file_name}_{start_time:.2f}_{end_time:.2f}.mp4")
#     ffmpeg_cmd = [
#         FFMPEG_PATH, '-y',
#         *ffmpeg_inputs,
#         # '-filter_complex', f'xstack=inputs={len(video_positions)}:layout={xstack_layout}[v]:fill=black[v]',
#         '-filter_complex', f'xstack=inputs={len(video_positions)}:layout={xstack_layout}:fill=black[v]',
#         '-map', '[v]',
#         '-map', '0:a?', '-c:a', 'aac', '-b:a', '192k',
#         '-r', str(output_fps),
#         '-c:v', 'libx264',   #'libx264',
#         '-crf', '23',
#         '-preset', 'ultrafast',
#         final_out
#     ]

#     # Execute the command

#     try:
#         subprocess.run(ffmpeg_cmd, check=True)
#         print(f"Video successfully created as {final_out}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error creating video: {e}")
        
        
# import subprocess
### CHatgpt version
def create_composite_video(composite_video_elements, final_out, output_fps=24):
    video_positions = []
    for element in composite_video_elements:
        video_positions.append({
            'path': element['vid_path'],
            'x': element['x_pos_pxl'],
            'y': element['y_pos_pxl']
        })

    # Build the ffmpeg inputs
    ffmpeg_inputs = []
    for pos in video_positions:
        ffmpeg_inputs.extend(['-i', pos['path']])

    # Create the xstack layout string
    layout = []
    for pos in video_positions:
        layout.append(f"{pos['x']}_{pos['y']}")
    xstack_layout = "|".join(layout)

    # ---------------------------
    # CHANGES START HERE
    # ---------------------------
    # Build a filter_complex that normalizes each input's timeline to:
    # - start at PTS=0 (setpts=PTS-STARTPTS)
    # - strict CFR at output_fps (fps=output_fps)
    # - consistent sample aspect ratio (setsar=1)
    norm_chains = []
    stack_inputs = []
    for i in range(len(video_positions)):
        norm_chains.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={output_fps},setsar=1[v{i}]"
        )
        stack_inputs.append(f"[v{i}]")

    filter_complex = (
        ";".join(norm_chains) + ";" +
        "".join(stack_inputs) +
        f"xstack=inputs={len(video_positions)}:layout={xstack_layout}:fill=black[v]"
    )
    # ---------------------------
    # CHANGES END HERE
    # ---------------------------

    ffmpeg_cmd = [
        FFMPEG_PATH, "-y",
        *ffmpeg_inputs,

        # CHANGED: use the new filter_complex that normalizes inputs before xstack
        "-filter_complex", filter_complex,

        "-map", "[v]",

        # Keep audio from input 0 if present.
        "-map", "0:a?", "-c:a", "aac", "-b:a", "192k",

        # CHANGED: force CFR muxing behavior (prevents timestamp-driven VFR output)
        "-vsync", "cfr",

        # Keep output fps explicit
        "-r", str(output_fps),

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "ultrafast",

        final_out
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully created as {final_out}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")


def process_layouts(layouts_to_export, view_videos, base_output_dir, file_name, dance_mode, mode_start_time, mode_end_time, output_fps=24):
    
    saved_resized_dir = os.path.join(base_output_dir, "temp_resized")
    
    for sv_name, current_layout in layouts_to_export.items():
        composite_video_elements = []
        
        try:
            os.makedirs(saved_resized_dir, exist_ok=True)
            for video_element in current_layout:
                video_path = view_videos[video_element['view']]
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                
                v_width, v_height = video_element['width'], video_element['height']
                x_pos_pxl, y_pos_pxl = video_element['x'], video_element['y']
                
                resized_path = resize_video(video_path, v_width, v_height, saved_resized_dir)
                
                composite_video_elements.append({
                    "view": video_element['view'], 
                    "vid_path":  resized_path,
                    "x_pos_pxl": x_pos_pxl,
                    "y_pos_pxl": y_pos_pxl,
                })
            
            final_save_dir = os.path.dirname(base_output_dir)
            final_out = os.path.join(final_save_dir, f"{file_name}_{dance_mode}_{sv_name}_{mode_start_time}_{mode_end_time}.mp4")
            
            if os.path.exists(final_out):
                print(f"Composite video already exists: {final_out}")
                continue
            
            create_composite_video(composite_video_elements, final_out, output_fps= output_fps)
            
        except Exception as e:
            print(f"Error processing layout {sv_name}: {str(e)}")
            raise
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(saved_resized_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {str(e)}")
                
     
     
     
##------------------------- Separate functions for trimmed video and feet plots -------------------------

def extract_trimmed_cycle_videos(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    save_dir: str = "cycle_videos",
    video_dir: str = "data/videos/",
    export_fps: int = 24,
):
    """
    Extract video segments for windows around points of interest.
    Each video shows [0-cycle, +cycle] around the POI.
    """
    # Default video path if not provided
    # if video_dir is None:
    video_path = os.path.join(video_dir, f"{file_name}_pre_R_Mix.mp4")
    # video_path = f"data/videos/{file_name}_pre_R_Mix.mp4"
    
    print("Windows data:")
    for i, (win_start, win_end, t_poi) in enumerate(windows):
        print(f"Window {i+1}:")
        print(f"  Start: {win_start:.3f}")
        print(f"  End: {win_end:.3f}")
        print(f"  Duration: {win_end - win_start:.3f}")

    # Create save directory
    video_dir = os.path.join(save_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"\nProcessing {len(windows)} windows for video extraction")
    
    # Process each window
    for i, (win_start, win_end, t_poi) in enumerate(windows):
        print(f"\nProcessing window {i+1}:")
        print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
        
        # Calculate segment times
        start_time = win_start
        end_time = win_end
        duration = end_time - start_time
        
        # Extract video segment with audio using ffmpeg
        video_output_path = os.path.join(video_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        ffmpeg_cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            "-r", str(export_fps),
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
        
        print(f"  Video saved: {video_output_path}")
        print(f"  Video duration: {duration:.3f}s")
        
    print("\nVideo extraction complete!")


def extract_feet_cycle_plots(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,  # Trajectory data frame rate
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    save_dir: str = "cycle_videos",
    figsize: tuple = (10, 3),
    dpi: int = 200,
    legend_flag: bool = True,
):
    """
    Create trajectory animations for windows around points of interest.
    Each plot shows [0-cycle, +cycle] around the POI.
    """
    # Create save directory
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
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
    
    print(f"\nProcessing {len(windows)} windows for plot creation")
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
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        traj_n_frames = traj_end_frame - traj_start_frame
        
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
        frames = np.arange(start_time, end_time, 1/24)
        anim = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=1000/24,  # 24fps
            blit=True
        )
        
        # Save animation
        plot_output_path = os.path.join(plot_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        # Temporarily modify PATH to use system ffmpeg
        original_path = os.environ.get('PATH', '')
        ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
        
        writer = animation.FFMpegWriter(fps=24, bitrate=2000)
        anim.save(plot_output_path, writer=writer)
        plt.close(fig)
        
        print(f"  Plot saved: {plot_output_path}")
        print(f"  Plot duration: {len(frames)/24:.3f}s")
        
    print("\nPlot creation complete!")

           
#---------------------- Hand Distance and Velocity ---------------------------------
#----------------------------------------------------------------------------------

def extract_hand_distance_cycle_plots(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    frame_rate: float = 240,  # Trajectory data frame rate
    export_fps: int = 24,
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    output_dir: str = None,
    figsize: tuple = (10, 3),
    dpi: int = 200,
    legend_flag: bool = True,
    motion_feature: int = 0,  # If 0 - distance, if 1 - relative velocity, if 2 - relative acceleration
    
    refined_window: float = 0.1,  # seconds
    
    
):
    """
    Create trajectory animations for hand distance or relative velocity between LeftWrist and RightWrist.
    Each plot shows a cycle window.
    """
    # Create save directory if not provided
    if output_dir is None:
        output_dir = os.path.join("cycle_plots", file_name, "hand_distance")
        os.makedirs(output_dir, exist_ok=True)
    
    # bvh_to_mvnx = {
    #     'X': 'Y',  # BVH side → MVNX side
    #     'Y': 'Z',  # BVH vertical → MVNX vertical
    #     'Z': 'X',  # BVH forward → MVNX forward
    # }
    
    # Load joint position data for both wrists
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    worldpos_file = os.path.join(DIR_CSV, f"{base_name}_T_worldpos.csv")
    
    try:
        world_positions = pd.read_csv(worldpos_file)
        print(f"Successfully loaded CSV with {len(world_positions)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    # Get time column and position data for both wrists
    time_column = world_positions.columns[0]  # First column is time
    times = world_positions[time_column].values
    
    # Get positions for both wrists along the chosen axis
    L_pos_x = world_positions[f"LeftWrist.X"].values
    R_pos_x = world_positions[f"RightWrist.X"].values
    
    L_pos_y = world_positions[f"LeftWrist.Y"].values
    R_pos_y = world_positions[f"RightWrist.Y"].values
    
    L_pos_z = world_positions[f"LeftWrist.Z"].values
    R_pos_z = world_positions[f"RightWrist.Z"].values
    
    
    # Calculate velocities
    dt = 1.0 / frame_rate
    L_vel_x = np.gradient(L_pos_x, dt)
    R_vel_x = np.gradient(R_pos_x, dt)
    
    L_vel_y = np.gradient(L_pos_y, dt)
    R_vel_y = np.gradient(R_pos_y, dt)
    
    L_vel_z = np.gradient(L_pos_z, dt)
    R_vel_z = np.gradient(R_pos_z, dt)

    rel_vel = np.sqrt(
        (L_vel_x - R_vel_x)**2 +
        (L_vel_y - R_vel_y)**2 +
        (L_vel_z - R_vel_z)**2
    )
    
    # Calculate accelerations
    L_acc_x = np.gradient(L_vel_x, dt)
    R_acc_x = np.gradient(R_vel_x, dt)
    
    L_acc_y = np.gradient(L_vel_y, dt)
    R_acc_y = np.gradient(R_vel_y, dt)
    
    L_acc_z = np.gradient(L_vel_z, dt)
    R_acc_z = np.gradient(R_vel_z, dt)
    
    rel_acc = np.sqrt(
        (L_acc_x - R_acc_x)**2 +
        (L_acc_y - R_acc_y)**2 +
        (L_acc_z - R_acc_z)**2
    )
    
    
    hand_distance = np.sqrt(
        (L_pos_x - R_pos_x)**2 +
        (L_pos_y - R_pos_y)**2 +
        (L_pos_z - R_pos_z)**2
    )
    
    
    hand_contacts = detect_hand_contacts_from_distance(
                    hand_distance,
                    frame_rate=frame_rate,
                    min_gap_sec=0.3,
                    min_distance_thresh=0.19*100,
                    slope_min=0.3
                )
    
    # Choose  minima around the contacts using window from rel_speed or rel_acc
    refined_contacts = []
    win = int(refined_window * frame_rate)   # ± ms window

    file_parts = file_name.split("_")
    ensmble = file_parts[1]
    
    if ensmble == "E1" or ensmble == "E2":
        acc_threshold = 3000
    
    if ensmble == "E3":
        acc_threshold = 1500   


    for c in hand_contacts:
        # for rel_acc
        start = max(0, c - win)
        end = min(len(rel_acc), c + int(0.03* frame_rate))
        seg = rel_acc[start:end]
        
        # find local maxima in the window
        peaks, _ = find_peaks(seg, prominence=0.1* np.max(seg))
        if len(peaks) == 0:
            continue

        # convert to absolute indices
        abs_peaks = start + peaks
        peak_vals = seg[peaks]
        
        # keep only peaks above threshold
        mask = peak_vals > acc_threshold
        peak_vals = peak_vals[mask]
        abs_peaks = abs_peaks[mask]
        
        # sort peaks by magnitude (descending)
        order = np.argsort(peak_vals)[::-1]     # order = np.argsort(peak_vals)[::-1]
        abs_peaks = abs_peaks[order]
        peak_vals = peak_vals[order]
        
        # ---- NEW LOGIC ----
        if len(peak_vals) == 0:
            continue  # or skip this window safely

        elif len(peak_vals) == 1:
            local_max = abs_peaks[0]

        else:
            # identify earlier and later in time
            if abs_peaks[0] <= abs_peaks[1]:
                earlier_idx, later_idx = 0, 1
            else:
                earlier_idx, later_idx = 1, 0

            p_earlier = peak_vals[earlier_idx]
            p_later = peak_vals[later_idx]

            # symmetric relative difference
            rel_diff = abs(p_earlier - p_later) / max(p_earlier, p_later)

            # Rule 1: earlier peak stronger → choose earlier
            if p_earlier >= p_later:
                local_max = abs_peaks[earlier_idx]

            # Rule 2: later stronger, but earlier within 30% → choose earlier
            elif rel_diff < 0.30:
                local_max = abs_peaks[earlier_idx]

            # Rule 3: otherwise choose strongest
            else:
                # local_max = abs_peaks[later_idx]
                if p_earlier > p_later:
                    local_max = abs_peaks[earlier_idx]
                else:
                    local_max = abs_peaks[later_idx]
# 
        refined_contacts.append(local_max)

        # local_max = start + np.argmax(rel_acc[start:end])
        # refined_contacts.append(local_max)
            
    contacts = refined_contacts
    print(f"Detected {len(contacts)} hand contacts")
    if len(contacts) > 0:
        contact_times_all = times[contacts]
        print(f"Contact times range: {contact_times_all.min():.3f}s to {contact_times_all.max():.3f}s")
    
    
    if motion_feature == 0:
        plot_data = hand_distance
        plot_label = f"Hand Distance"
        ylabel = f"Hand Distance"
        plot_color = 'red'
    
    
    if motion_feature == 1:
        plot_data = rel_vel
        plot_label = f"Relative Velocity"
        ylabel = f"Relative Velocity"
        plot_color = 'green'

    
    if motion_feature == 2:
        plot_data = rel_acc
        plot_label = f"Relative Acceleration"
        ylabel = f"Relative Acceleration"
        plot_color = 'purple'

        
    print(f"\nProcessing {len(windows)} windows")
    
    # Process each window or cycle
    for i, (win_start, win_end, _) in enumerate(windows):
        print(f"\nProcessing window {i+1}:")
        print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
        
        # Calculate segment times
        start_time = win_start
        end_time = win_end
        duration = end_time - start_time
        
        # Calculate window parameters
        beat_len = duration / n_beats_per_cycle
        subdiv_len = beat_len / n_subdiv_per_beat
        half_win = subdiv_len * nn
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        traj_n_frames = traj_end_frame - traj_start_frame
        
        print(f"  Trajectory frames: {traj_start_frame} to {traj_end_frame} (240fps)")
        
        # Check if we have valid frame numbers
        if traj_start_frame >= traj_end_frame:
            print(f"  Skipping window {i+1}: Invalid frame range (start >= end)")
            continue
        if traj_start_frame < 0:
            print(f"  Skipping window {i+1}: Start frame < 0")
            continue
        if traj_end_frame > len(plot_data):
            print(f"  Skipping window {i+1}: End frame > total frames")
            continue
        
        # Trim trajectory data using frame numbers at 240fps
        data_win = plot_data[traj_start_frame:traj_end_frame]
        t_win = times[traj_start_frame:traj_end_frame]
        
        # Check if we have valid trajectory data
        if len(data_win) == 0:
            print(f"  Skipping window {i+1}: No trajectory data")
            continue
        
        print(f"  Trajectory data points: {len(data_win)}")
        
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
                ax.axvline(subdiv_time, color=color, linestyle='-', linewidth=2, alpha=0.7)  # beat color
            else:
                ax.axvline(subdiv_time, color=color, linestyle='--', linewidth=1, alpha=0.3)  # subdivision color
        
        # Plot trajectory
        ax.plot(t_win, data_win, '-', color=plot_color, alpha=0.5, label=plot_label, linewidth=3.5)
        
        # Plot hand contact markers (only for distance plots)
        if len(contacts) > 0:            
            # Convert contact frame indices to time values using times array
            contact_times = times[contacts]
            
            # Find contacts within this window
            window_contacts = contact_times[(contact_times >= start_time) & (contact_times <= end_time)]
            
            print(f"  Found {len(window_contacts)} contacts in window {i+1}")
            
            if len(window_contacts) > 0:
                # Get corresponding distance values for markers
                contact_dist_values = []
                for ct in window_contacts:
                    # Find closest time point in t_win
                    closest_idx = np.argmin(np.abs(t_win - ct))
                    contact_dist_values.append(data_win[closest_idx])
                
                # Plot contact markers
                ax.plot(window_contacts, contact_dist_values, 'o', 
                       color='blue', ms=5, alpha=0.8, markeredgecolor='k', 
                       markeredgewidth=1, zorder=5, label="Hand contacts")
        
    
        
        # Set y-axis limits with safety checks
        try:
            y_min = data_win.min()
            y_max = data_win.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        except ValueError as e:
            print(f"  Warning: Could not set y-axis limits: {e}")
            ax.set_ylim(-1, 1)
        
        # Create vertical playhead
        v_playhead, = ax.plot([start_time, start_time], 
                            [y_min - 0.1*y_range, y_max + 0.1*y_range],
                            lw=1.5, alpha=0.9, color='orange')
        
        # Set up the plot with scaled x-axis
        ax.set_xlabel(f'Beat span')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{file_name} | Window:{start_time:.0f}s - {end_time:.2f}| Time: {start_time:.0f}s')
        ax.grid(True, alpha=0.3)
        
        # Scale x-axis to show beats instead of cycles
        x_ticks = np.arange(1, n_beats_per_cycle + 2)
        x_tick_positions = start_time + (x_ticks - 1) * beat_len
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(start_time, end_time)
        
        # Add legend
        custom = [
            Line2D([0],[0], color=plot_color, lw=1.5),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', 
                   ms=5, markeredgecolor='k', markeredgewidth=1, alpha=0.8),  # Add contact marker         if not use_relative_velocity else Line2D([0],[0], color='none')
            Line2D([0],[0], color='black', lw=1),
            Line2D([0],[0], color='green', lw=1, linestyle='--'),
            Line2D([0],[0], color='red', lw=1, linestyle='--'),
        ]
        
        labels = [
            plot_label,
            "Refined contacts",  # Add contact label    if not use_relative_velocity else ""
            "Subdiv-1 (1,4,7,10)", 
            "Subdiv-2 (2,5,8,11)", 
            "Subdiv-3 (3,6,9,12)"
        ]
        
        if legend_flag:
            ax.legend(custom, labels, loc='lower right', framealpha=0.2, fontsize=6)
        
        def update(frame):
            v_playhead.set_xdata([frame, frame])
            ax.set_title(f'{file_name} | Window:{start_time:.0f}s - {end_time:.0f}s| Time: {frame:.0f}s')
            return v_playhead,
        
        # Create animation frames at export_fps
        frames = np.arange(start_time, end_time, 1/export_fps)
        anim = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=1000/export_fps,  # export_fps
            blit=True
        )
        
        # Save animation
        plot_output_path = os.path.join(output_dir, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        # Temporarily modify PATH to use system ffmpeg
        original_path = os.environ.get('PATH', '')
        ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
        
        writer = animation.FFMpegWriter(fps= export_fps, 
                                        bitrate=2000,
                                        codec='libx264',
                                        )
        anim.save(plot_output_path, writer=writer)
        plt.close(fig)
        
        print(f"Plot saved: {plot_output_path}")
        print(f"Plot duration: {len(frames)/export_fps:.3f}s")
    
    print("\nProcessing complete!")
    
    
def dynamic_concatenate_and_overlay_videos(folder_names, save_dir, views_to_generate=None):
    """
    Concatenate videos from multiple folders dynamically.
    
    Args:
        file_name: Name of the file (for logging)
        folder_names: List of folder names to process. Can be:
            - String: 'videos' (uses 'videos' as both folder and output name)
            - Single-element tuple: ('video_mix',) (uses 'video_mix' as both folder and output name)
            - Two-element tuple: ('videos', 'video_mix') (folder_name, output_name)
        save_dir: Base directory where folders are located
        views_to_generate: Optional list of view names (e.g., ['front', 'left', 'right', 'top'])
    """
    # Normalize folder_names to list of tuples (folder_name, output_name)
    normalized_folders = []
    for item in folder_names:
        if isinstance(item, tuple):
            if len(item) == 1:
                # Single element tuple: use it as both folder and output name
                folder_name = output_name = item[0]
            elif len(item) == 2:
                # Two element tuple: (folder_name, output_name)
                folder_name, output_name = item
            else:
                raise ValueError(f"Tuple must have 1 or 2 elements, got {len(item)}: {item}")
        else:
            # String: use it as both folder name and output name
            folder_name = output_name = item
        normalized_folders.append((folder_name, output_name))
    
    # Create directory paths and text file paths dynamically
    folder_configs = {}
    for folder_name, output_name in normalized_folders:
        folder_dir = os.path.join(save_dir, folder_name)
        text_file = os.path.join(save_dir, f"{output_name}_list.txt")
        folder_configs[output_name] = {
            'dir': folder_dir,
            'text_file': text_file,
            'folder_name': folder_name
        }
    
    # Handle view folders if provided
    if views_to_generate:
        for view in views_to_generate:
            folder_dir = os.path.join(save_dir, f"{view}_view")
            text_file = os.path.join(save_dir, f"{view}_view.txt")
            folder_configs[f"{view}_view"] = {
                'dir': folder_dir,
                'text_file': text_file,
                'folder_name': f"{view}_view"
            }
    
    # Check if directories exist
    missing_dirs = []
    for output_name, config in folder_configs.items():
        if not os.path.exists(config['dir']):
            missing_dirs.append(config['dir'])
            print(f"Warning: Directory not found: {config['dir']}")
    
    if missing_dirs:
        print(f"Skipping concatenation for {len(missing_dirs)} missing directories")
        # Remove missing directories from processing
        for dir_path in missing_dirs:
            for output_name, config in list(folder_configs.items()):
                if config['dir'] == dir_path:
                    del folder_configs[output_name]
    
    # Check if text files already exist
    existing_files = []
    missing_files = []
    for output_name, config in folder_configs.items():
        if os.path.exists(config['text_file']):
            existing_files.append(output_name)
        else:
            missing_files.append(output_name)
    
    if existing_files and not missing_files:
        print("Concatenation files already exist, skipping creation")
    else:
        print("Creating concatenation files...")
        for output_name, config in folder_configs.items():
            if output_name in missing_files:
                create_concat_file(config['dir'], config['text_file'])
    
    # Concatenate videos for each folder
    for output_name, config in folder_configs.items():
        concatenate_videos(config['text_file'], save_dir, f"{output_name}_concat")
    
    print(f"Concatenation complete: {save_dir}")
    
    
def detect_hand_contacts_from_distance(
    hand_distance,
    frame_rate,
    min_gap_sec=0.3,
    min_distance_thresh=0.19,
    prominence=0.01,
    slope_min=0.3
):
    """
    Detect hand-contact events using only inter-hand distance.

    Parameters
    ----------
    hand_distance : np.ndarray
        1D array of inter-hand distance over time
    frame_rate : float
        Sampling rate (frames per second)
    min_gap_sec : float, optional
        Minimum temporal gap between extrema (seconds)
    min_distance_thresh : float, optional
        Keep only minima where hand_distance <= this value
    prominence : float, optional
        Peak prominence for maxima/minima detection
    slope_min : float, optional
        Minimum descent slope (distance units per second)

    Returns
    -------
    contacts : list of int
        Frame indices of detected hand contacts
    """

    min_gap_samples = int(min_gap_sec * frame_rate)

    # --- find maxima ---
    max_idx, _ = find_peaks(
        hand_distance,
        distance=min_gap_samples,
        prominence=prominence
    )

    # --- find minima (invert signal) ---
    min_idx, _ = find_peaks(
        -hand_distance,
        distance=min_gap_samples,
        prominence=prominence,
        height=-min_distance_thresh
    )

    contacts = []

    for t_min in min_idx:
        prev_max = max_idx[max_idx < t_min]
        if len(prev_max) == 0:
            continue

        t_max = prev_max[-1]

        delta_d = hand_distance[t_max] - hand_distance[t_min]
        delta_t = (t_min - t_max) / frame_rate

        if delta_t <= 0:
            continue

        slope = delta_d / delta_t

        if slope >= slope_min:
            contacts.append(t_min)

    return contacts


def extract_hand_distance_cycle_plots_static(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    frame_rate: float = 240,  # Trajectory data frame rate
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    output_dir: str = None,
    figsize_per_plot: tuple = (10, 3),  # Size per individual subplot
    dpi: int = 100,
    legend_flag: bool = True,
    refined_window: float = 0.1,  # seconds
):
    """
    Create static plots for hand distance, relative velocity, and relative acceleration 
    between LeftWrist and RightWrist. Each plot shows a cycle window. 
    All three motion features are arranged in three rows, with windows as columns.
    """
    # Create save directory if not provided
    if output_dir is None:
        output_dir = os.path.join("cycle_plots", file_name, "hand_distance")
        os.makedirs(output_dir, exist_ok=True)
    
    bvh_to_mvnx = {
        'x': 'y',  # BVH side → MVNX side
        'y': 'z',  # BVH vertical → MVNX vertical
        'z': 'x',  # BVH forward → MVNX forward
    }
    
    # Load joint position data for both wrists
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    worldpos_file = os.path.join(DIR_CSV, f"{base_name}_T_worldpos.csv")
    
    try:
        world_positions = pd.read_csv(worldpos_file)
        print(f"Successfully loaded CSV with {len(world_positions)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    # Get time column and position data for both wrists
    time_column = world_positions.columns[0]  # First column is time
    times = world_positions[time_column].values
    
    # Get positions for both wrists along the chosen axis
    L_pos_x = world_positions[f"LeftWrist.X"].values
    R_pos_x = world_positions[f"RightWrist.X"].values
    
    L_pos_y = world_positions[f"LeftWrist.Y"].values
    R_pos_y = world_positions[f"RightWrist.Y"].values
    
    L_pos_z = world_positions[f"LeftWrist.Z"].values
    R_pos_z = world_positions[f"RightWrist.Z"].values
    
    
    # Calculate velocities
    dt = 1.0 / frame_rate
    L_vel_x = np.gradient(L_pos_x, dt)
    R_vel_x = np.gradient(R_pos_x, dt)
    
    L_vel_y = np.gradient(L_pos_y, dt)
    R_vel_y = np.gradient(R_pos_y, dt)
    
    L_vel_z = np.gradient(L_pos_z, dt)
    R_vel_z = np.gradient(R_pos_z, dt)

    rel_vel = np.sqrt(
        (L_vel_x - R_vel_x)**2 +
        (L_vel_y - R_vel_y)**2 +
        (L_vel_z - R_vel_z)**2
    )
    
    # Calculate accelerations
    L_acc_x = np.gradient(L_vel_x, dt)
    R_acc_x = np.gradient(R_vel_x, dt)
    
    L_acc_y = np.gradient(L_vel_y, dt)
    R_acc_y = np.gradient(R_vel_y, dt)
    
    L_acc_z = np.gradient(L_vel_z, dt)
    R_acc_z = np.gradient(R_vel_z, dt)
    
    rel_acc = np.sqrt(
        (L_acc_x - R_acc_x)**2 +
        (L_acc_y - R_acc_y)**2 +
        (L_acc_z - R_acc_z)**2
    )
    
    
    hand_distance = np.sqrt(
        (L_pos_x - R_pos_x)**2 +
        (L_pos_y - R_pos_y)**2 +
        (L_pos_z - R_pos_z)**2
    )
    
    
    hand_contacts = detect_hand_contacts_from_distance(
                    hand_distance,
                    frame_rate=frame_rate,
                    min_gap_sec=0.3,
                    min_distance_thresh=0.19*100,
                    slope_min=0.3
                )
    
    # Choose  minima around the contacts using window from rel_speed or rel_acc
    refined_contacts = []
    win = int(refined_window * frame_rate)   # ± ms window

    file_parts = file_name.split("_")
    ensmble = file_parts[1]
    
    if ensmble == "E1" or ensmble == "E2":
        acc_threshold = 3000
    
    if ensmble == "E3":
        acc_threshold = 1500    
        
        
    for c in hand_contacts:
        # for rel_acc
        start = max(0, c - win)
        end = min(len(rel_acc), c + 0.03* frame_rate)
        seg = rel_acc[start:end]
        
        # find local maxima in the window
        peaks, _ = find_peaks(seg, prominence=0.1* np.max(seg))
        if len(peaks) == 0:
            continue

        # convert to absolute indices
        abs_peaks = start + peaks
        peak_vals = seg[peaks]
        
        # keep only peaks above threshold
        mask = peak_vals > acc_threshold
        peak_vals = peak_vals[mask]
        abs_peaks = abs_peaks[mask]
        
        # sort peaks by magnitude (descending)
        order = np.argsort(peak_vals)[::-1]
        abs_peaks = abs_peaks[order]
        peak_vals = peak_vals[order]
        
        # ---- NEW LOGIC ----
        if len(peak_vals) == 0:
            continue  # or skip this window safely

        elif len(peak_vals) == 1:
            local_max = abs_peaks[0]

        else:
            # identify earlier and later in time
            if abs_peaks[0] <= abs_peaks[1]:
                earlier_idx, later_idx = 0, 1
            else:
                earlier_idx, later_idx = 1, 0

            p_earlier = peak_vals[earlier_idx]
            p_later = peak_vals[later_idx]

            # symmetric relative difference
            rel_diff = abs(p_earlier - p_later) / max(p_earlier, p_later)

            # Rule 1: earlier peak stronger → choose earlier
            if p_earlier >= p_later:
                local_max = abs_peaks[earlier_idx]

            # Rule 2: later stronger, but earlier within 30% → choose earlier
            elif rel_diff < 0.30:
                local_max = abs_peaks[earlier_idx]

            # Rule 3: otherwise choose strongest
            else:
                # local_max = abs_peaks[later_idx]
                if p_earlier > p_later:
                    local_max = abs_peaks[earlier_idx]
                else:
                    local_max = abs_peaks[later_idx]
# 
        refined_contacts.append(local_max)

        

                  
    contacts = refined_contacts
    contact_times = times[contacts]
    contact_total = contact_times[(contact_times >= windows[0][0]) & (contact_times <= windows[-1][1])]
    
    # Save hand contacts as pickle
    contacts_path = os.path.join("data/hand_contacts_new", f"{file_name}__{windows[0][0]:.2f}_{windows[-1][1]:.2f}_contacts.pkl")
    os.makedirs(os.path.dirname(contacts_path), exist_ok=True)
    with open(contacts_path, 'wb') as f:
        pickle.dump(contact_total, f)
    # #################
    
    
    print(f"Detected {len(contacts)} hand contacts")
    if len(contacts) > 0:
        contact_times_all = times[contacts]
        print(f"Contact times range: {contact_times_all.min():.3f}s to {contact_times_all.max():.3f}s")
    
    # Define all three motion features
    motion_features = [
        {
            'data': hand_distance,
            'label': 'Hand Distance',
            'ylabel': 'Hand Distance',
            'color': 'red'
        },
        {
            'data': rel_vel,
            'label': 'Relative Velocity',
            'ylabel': 'Relative Velocity',
            'color': 'green'
        },
        {
            'data': rel_acc,
            'label': 'Relative Acceleration',
            'ylabel': 'Relative Acceleration',
            'color': 'purple'
        }
    ]
        
    print(f"\nProcessing {len(windows)} windows")
    
    # Filter valid windows first (check against hand_distance as reference)
    valid_windows = []
    for i, (win_start, win_end, _) in enumerate(windows):
        start_time = win_start
        end_time = win_end
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        
        # Check if we have valid frame numbers
        if traj_start_frame >= traj_end_frame:
            print(f"  Skipping window {i+1}: Invalid frame range (start >= end)")
            continue
        if traj_start_frame < 0:
            print(f"  Skipping window {i+1}: Start frame < 0")
            continue
        if traj_end_frame > len(hand_distance):
            print(f"  Skipping window {i+1}: End frame > total frames")
            continue
        
        # Trim trajectory data using frame numbers at 240fps
        data_win = hand_distance[traj_start_frame:traj_end_frame]
        
        # Check if we have valid trajectory data
        if len(data_win) == 0:
            print(f"  Skipping window {i+1}: No trajectory data")
            continue
        
        valid_windows.append((i, win_start, win_end))
    
    if len(valid_windows) == 0:
        print("No valid windows to plot!")
        return
    
    # Create figure with 3 rows (one per motion feature) and n_plots columns (one per window)
    n_plots = len(valid_windows)
    n_rows = 3  # One row for each motion feature
    fig_width = figsize_per_plot[0] * n_plots
    fig_height = figsize_per_plot[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_plots, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Handle case where there's only one subplot (axes won't be a 2D array)
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    
    # Process each motion feature (row)
    for row_idx, feature in enumerate(motion_features):
        plot_data = feature['data']
        plot_label = feature['label']
        ylabel = feature['ylabel']
        plot_color = feature['color']
        
        # Process each window (column)
        for plot_idx, (orig_idx, win_start, win_end) in enumerate(valid_windows):
            ax = axes[row_idx, plot_idx]
            
            if row_idx == 0:  # Only print window info once (for first row)
                print(f"\nProcessing window {orig_idx+1}:")
                print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
            
            # Calculate segment times
            start_time = win_start
            end_time = win_end
            duration = end_time - start_time
            
            # Calculate window parameters
            beat_len = duration / n_beats_per_cycle
            subdiv_len = beat_len / n_subdiv_per_beat
            half_win = subdiv_len * nn
            
            # Calculate frame numbers for trajectory (240fps)
            traj_start_frame = int(start_time * frame_rate)
            traj_end_frame = int(end_time * frame_rate)
            traj_n_frames = traj_end_frame - traj_start_frame
            
            if row_idx == 0:  # Only print frame info once
                print(f"  Trajectory frames: {traj_start_frame} to {traj_end_frame} (240fps)")
                print(f"  Trajectory data points: {traj_n_frames}")
            
            # Trim trajectory data using frame numbers at 240fps
            data_win = plot_data[traj_start_frame:traj_end_frame]
            t_win = times[traj_start_frame:traj_end_frame]
            
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
                    ax.axvline(subdiv_time, color=color, linestyle='-', linewidth=2, alpha=0.7)  # beat color
                else:
                    ax.axvline(subdiv_time, color=color, linestyle='--', linewidth=1, alpha=0.3)  # subdivision color
            
            # Plot trajectory
            ax.plot(t_win, data_win, '-', color=plot_color, alpha=0.5, label=plot_label, linewidth=3.5)
            
            # Plot hand contact markers (only for distance plots - row 0)
            if len(contacts) > 0:            
                # Convert contact frame indices to time values using times array
                
                
                # Find contacts within this window
                window_contacts = contact_times[(contact_times >= start_time) & (contact_times <= end_time)]
                
                if row_idx == 0:  # Only print contact info once
                    print(f"  Found {len(window_contacts)} contacts in window {orig_idx+1}")
                
                if len(window_contacts) > 0:
                    # Get corresponding distance values for markers
                    contact_dist_values = []
                    for ct in window_contacts:
                        # Find closest time point in t_win
                        closest_idx = np.argmin(np.abs(t_win - ct))
                        contact_dist_values.append(data_win[closest_idx])
                    
                    # Plot contact markers
                    ax.plot(window_contacts, contact_dist_values, 'o', 
                           color='blue', ms=5, alpha=0.8, markeredgecolor='k', 
                           markeredgewidth=1, zorder=5, label="Hand contacts")
            
            # Set y-axis limits with safety checks
            try:
                y_min = data_win.min()
                y_max = data_win.max()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
            except ValueError as e:
                print(f"  Warning: Could not set y-axis limits: {e}")
                ax.set_ylim(-1, 1)
                y_min = -1
                y_max = 1
                y_range = 2
            
            # Set up the plot with scaled x-axis
            if plot_idx == 0:  # Only label y-axis on the leftmost plot of each row
                ax.set_ylabel(ylabel)
            
            # Only show x-axis label on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel('Beat span')
            
            # Only show title on top row
            if row_idx == 0:
                ax.set_title(f'Window {orig_idx+1}\n {file_name} | {start_time:.0f}s - {end_time:.2f}s | Tot. Contacts: {len(contact_total)}')
            else:
                ax.set_title('')  # Clear title for other rows
            
            ax.grid(True, alpha=0.3)
            
            # Scale x-axis to show beats instead of cycles
            x_ticks = np.arange(1, n_beats_per_cycle + 2)
            x_tick_positions = start_time + (x_ticks - 1) * beat_len
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_ticks)
            ax.set_xlim(start_time, end_time)
            
            # Add legend only on the rightmost plot of the bottom row
            if legend_flag and plot_idx == n_plots - 1 and row_idx == n_rows - 1:
                custom = [
                    Line2D([0],[0], color='red', lw=1.5, label='Hand Distance'),
                    Line2D([0],[0], color='green', lw=1.5, label='Relative Velocity'),
                    Line2D([0],[0], color='purple', lw=1.5, label='Relative Acceleration'),
                    Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', 
                           ms=5, markeredgecolor='k', markeredgewidth=1, alpha=0.8, label='Refined contacts'),
                    Line2D([0],[0], color='black', lw=1, label='Subdiv-1 (1,4,7,10)'),
                    Line2D([0],[0], color='green', lw=1, linestyle='--', label='Subdiv-2 (2,5,8,11)'),
                    Line2D([0],[0], color='red', lw=1, linestyle='--', label='Subdiv-3 (3,6,9,12)'),
                ]
                
                labels = [
                    'Hand Distance',
                    'Relative Velocity',
                    'Relative Acceleration',
                    'Refined contacts',
                    'Subdiv-1 (1,4,7,10)', 
                    'Subdiv-2 (2,5,8,11)', 
                    'Subdiv-3 (3,6,9,12)'
                ]
                
                ax.legend(custom, labels, loc='lower right', framealpha=0.2, fontsize=6)
    
    # Save the complete figure as a static image
    plot_output_path = os.path.join(output_dir, f"{file_name}_{windows[0][0]:.2f}_{windows[-1][1]:.2f}_all_cycles_static.png")
    fig.savefig(plot_output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nStatic plot saved: {plot_output_path}")
    print("Processing complete!")
    
    
    