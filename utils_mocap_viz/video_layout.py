import os
import subprocess

# def get_layout_config(layout_name='L2'):
#     """Get layout configuration by name"""
#     layouts = {
#         'L1': [
#             {'view': 'front', 'x': 0, 'y': 0, 'width': 640, 'height': 360},
#             {'view': 'BKO_E1_D5_01_Maraka_pre_R_Mix.mp4', 'x': 640, 'y': 0, 'width': 640, 'height': 360},
#             {'view': 'joint_position', 'x': 0, 'y': 360, 'width': 1280, 'height': 180}
#         ],
#         'L2': [
#             {'view': 'front', 'x': 0, 'y': 0, 'width': 640, 'height': 360},
#             {'view': 'right', 'x': 640, 'y': 0, 'width': 640, 'height': 360},
#             {'view': 'joint_position', 'x': 0, 'y': 360, 'width': 1280, 'height': 360}
#         ]
#     }
#     return layouts.get(layout_name, layouts['L2'])  # Default to L2 if layout not found

def get_video_paths(output_dir, filename, joint_name="LeftAnkle", axis="y"):
    """
    Scan `output_dir` for expected video/audio files and return a dict mapping
    descriptive keys to full file paths. Only files that actually exist are included.
    
    Parameters:
    - output_dir (str): directory containing the video/audio files
    - filename (str): base name used for the mixed video/audio, e.g. "BKO_E1_D1_02_Maraka"
    - joint_name (str): joint name for the joint_position file, e.g. "LeftAnkle"
    - axis (str): axis for the joint_position file, e.g. "y"
    
    Returns:
    - dict[str, str]: mapping of keys to their corresponding file paths
      Possible keys (if the matching file is found):
        • "front", "right", "left", "top"       ← any file ending in "_view.mp4"
        • "dundun"                               ← "Dun.mp4"
        • "J1", "J2"                             ← "J1.mp4", "J2.mp4"
        • "combined_drum"                             ← "drum_combined.mp4"
        • "joint_pos"                            ← f"{joint_name}_{axis}_position.mp4"
        • "video_mix"                            ← f"{filename}_pre_R_Mix_trimmed.mp4"
        • "audio"                                ← f"{filename}_pre_R_Mix_trimmed_audio.mp3"
    """

    video_paths = {}
    files = os.listdir(output_dir)

    # 1) _view files: map "<prefix>_view.mp4" → key = "<prefix>"
    for fname in files:
        if not fname.lower().endswith(".mp4"):
            continue

        base, ext = os.path.splitext(fname)
        if base.endswith("_view"):
            # e.g. "front_view.mp4" → key "front"
            view_key = base[: -len("_view")]
            video_paths[view_key] = os.path.join(output_dir, fname)

    # 2) "Dun.mp4" → key "dundun"
    dun_name = "Dun.mp4"
    if dun_name in files:
        video_paths["dundun"] = os.path.join(output_dir, dun_name)

    # 3) "J1.mp4" and "J2.mp4" → keys "J1", "J2"
    for joint_vid in ("J1.mp4", "J2.mp4"):
        if joint_vid in files:
            key = os.path.splitext(joint_vid)[0]  # yields "J1" or "J2"
            video_paths[key] = os.path.join(output_dir, joint_vid)

    # 4) "drum_combined.mp4" → key "combined"
    combined_name = "drum_combined.mp4"
    if combined_name in files:
        video_paths["combined_drum"] = os.path.join(output_dir, combined_name)

    # 5) joint position: f"{joint_name}_{axis}_position.mp4" → key "joint_pos"
    joint_vid_name = f"{joint_name}_{axis}_position.mp4"
    if joint_vid_name in files:
        video_paths["joint_pos"] = os.path.join(output_dir, joint_vid_name)

    # 6) mixed video: f"{filename}_pre_R_Mix_trimmed.mp4" → key "video_mix"
    mix_vid = f"{filename}_pre_R_Mix_trimmed.mp4"
    if mix_vid in files:
        video_paths["video_mix"] = os.path.join(output_dir, mix_vid)

    # 7) audio: f"{filename}_pre_R_Mix_trimmed_audio.mp3" → key "audio"
    audio_name = f"{filename}_pre_R_Mix_trimmed_audio.mp3"
    if audio_name in files:
        video_paths["audio"] = os.path.join(output_dir, audio_name)

    return video_paths

def combine_views(
    filename,
    start_time,
    end_time,
    output_dir,
    view_videos,
    video_path=None,
    layout_config=None,
    video_size=(1280, 720),
    fps=24
):
    """Combine multiple views into a single video with custom layout
    
    Args:
        filename: Base filename for output
        start_time: Start time in seconds
        end_time: End time in seconds
        view_videos: Dictionary mapping view names to their video paths
        video_path: Path to video file for replacement views
        layout_config: List of dictionaries defining the layout structure, where each dict contains:
                      - view: Name of the view/video
                      - x: X coordinate position
                      - y: Y coordinate position
                      - width: Width of the video
                      - height: Height of the video
        video_size: Overall output video size (width, height)
        fps: Target frame rate for output video
    """
    # Get output directory
    # output_dir = os.path.join("output", f"{filename}_{start_time:.1f}_{end_time:.1f}")
    # os.makedirs(output_dir, exist_ok=True)
    
    # Build ffmpeg filter complex for layout
    filter_complex = []
    input_files = []
    input_count = 0
    
    # Create a black background layer
    filter_complex.append(
        f'color=c=black:s={video_size[0]}x{video_size[1]}:d={end_time-start_time}[base];'
    )
    
    # Process each video in the layout
    last_output = 'base'
    for item in layout_config:
        view_name = item['view']
        x = item['x']
        y = item['y']
        width = item['width']
        height = item['height']
        
        # Get video path from view_videos dictionary
        video_path = view_videos[view_name]
        input_files.extend(['-i', video_path])
        
        # Scale video to specified dimensions and ensure frame rate and timing
        filter_complex.append(
            f'[{input_count}:v]'
            f'fps={fps},'  # First ensure consistent frame rate
            f'setpts=PTS-STARTPTS,'  # Reset timestamps
            f'scale={width}:{height}'  # Then scale to desired size
            f'[v{input_count}];'
        )
        
        # Overlay this video on top of the previous result
        if input_count == 0:
            filter_complex.append(f'[base][v{input_count}]overlay=x={x}:y={y}[overlay{input_count}];')
        else:
            filter_complex.append(f'[overlay{input_count-1}][v{input_count}]overlay=x={x}:y={y}[overlay{input_count}];')
        last_output = f'overlay{input_count}'
        input_count += 1
    
    # Output file path
    output_file = os.path.join(output_dir, f"combined_{os.path.splitext(os.path.basename(filename))[0]}.mp4")
    
    # Add audio input if available
    if 'audio' in view_videos and os.path.exists(view_videos['audio']):
        input_files.extend(['-i', view_videos['audio']])
        print(f"Adding audio from: {view_videos['audio']}")
    else:
        print("No audio file found in view_videos")
    
    # Use ffmpeg to create the final video
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        *input_files,
        '-filter_complex', ''.join(filter_complex)[:-1],  # Remove last semicolon
        '-map', f'[{last_output}]',
        *(['-map', f'{input_count}:a'] if 'audio' in view_videos and os.path.exists(view_videos['audio']) else []),  # Add audio mapping if available
        '-c:v', 'libx264',
        '-c:a', 'aac',  # Add audio codec
        '-preset', 'slow',  # Use slower preset for better quality
        '-crf', '18',  # Use higher quality (lower value = higher quality, 18-28 is good range)
        '-pix_fmt', 'yuv420p',
        '-r', str(fps),
        '-t', str(end_time - start_time),
        output_file
    ]
    
    print("\nPreparing to combine videos:")
    print(f"Output file: {output_file}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Frame rate: {fps} fps")
    print(f"Number of input files: {len(input_files)}")
    if 'audio' in view_videos and os.path.exists(view_videos['audio']):
        print("Audio file will be included")
    
    print("\nFFmpeg command:")
    print(' '.join(ffmpeg_cmd))
    
    try:
        print("\nExecuting FFmpeg command...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print("FFmpeg command completed successfully")
        if result.stdout:
            print("FFmpeg output:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing FFmpeg command:")
        print(f"Error code: {e.returncode}")
        print(f"Error output:")
        print(e.stderr)
        raise
    except Exception as e:
        print(f"\nUnexpected error during video combination:")
        print(str(e))
        raise
    
    print(f"\nCombined video saved to {output_file}")
    
    
    return output_file