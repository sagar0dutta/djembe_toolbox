import os
import subprocess
from .mocap_visualizer_F_view import MocapVisualizerFront
from .mocap_visualizer_RS_view import MocapVisualizerRightSide
from .mocap_visualizer_LS_view import MocapVisualizerLeftSide
from .mocap_visualizer_T_view import MocapVisualizerTop
from .kinematic_visualizer import visualize_joint_position

# Use system ffmpeg which has libx264 support
FFMPEG_PATH = "/itf-fi-ml/home/sagardu/bin/ffmpeg"

MIN_VALID_VIDEO_BYTES = 10_000

VIEW_CLASSES = {
    'front': MocapVisualizerFront,
    'right': MocapVisualizerRightSide,
    'left': MocapVisualizerLeftSide,
    'top': MocapVisualizerTop,
}


def is_valid_video_file(path, min_bytes=MIN_VALID_VIDEO_BYTES):
    """True if path looks like a complete rendered MP4, not a partial/corrupt file."""
    return os.path.isfile(path) and os.path.getsize(path) >= min_bytes


def get_output_dir(bvh_file, start_time, end_time, base_dir="output"):
    """Create and return output directory path based on filename and time range"""
    filename = os.path.splitext(os.path.basename(bvh_file))[0]
    dir_name = f"{filename}_{start_time:.1f}_{end_time:.1f}"
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_view_visualizers(bvh_file, views_to_generate, debug=False, **viz_kwargs):
    """Create one PyVista visualizer per view (reuse across cycle windows).

    Extra kwargs (frontal_method, markers, smooth_sec) are forwarded to the
    visualizer; omit them to use the module defaults.
    """
    if not bvh_file.endswith('.bvh'):
        bvh_file = bvh_file + '.bvh'
    visualizers = {}
    for view in views_to_generate:
        visualizer_class = VIEW_CLASSES[view]
        visualizers[view] = visualizer_class(bvh_file, debug=debug, **viz_kwargs)
    return visualizers


def close_view_visualizers(visualizers):
    if not visualizers:
        return
    for visualizer in visualizers.values():
        plotter = getattr(visualizer, 'plotter', None)
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


def generate_individual_videos(
    bvh_file,
    start_time,
    end_time,
    output_dir,
    output_fps,
    video_size,
    views_to_generate=None,
    visualizers=None,
    **viz_kwargs,
):
    """Generate videos for specified views.

    ``viz_kwargs`` (frontal_method, markers, smooth_sec) apply only when this
    function creates its own visualizers (i.e. when ``visualizers`` is None).
    """
    if views_to_generate is None:
        views_to_generate = ['front', 'right']

    if not bvh_file.endswith('.bvh'):
        bvh_file = bvh_file + '.bvh'

    view_filenames = {
        'front': f'front_view_{start_time:.1f}_{end_time:.1f}.mp4',
        'right': f'right_view_{start_time:.1f}_{end_time:.1f}.mp4',
        'left': f'left_view_{start_time:.1f}_{end_time:.1f}.mp4',
        'top': f'top_view_{start_time:.1f}_{end_time:.1f}.mp4',
    }

    owned_visualizers = []
    try:
        for view in views_to_generate:
            if view not in VIEW_CLASSES:
                continue

            output_filename = view_filenames[view]
            output_file = os.path.join(output_dir, output_filename)

            if visualizers and view in visualizers:
                visualizer = visualizers[view]
                close_after = False
            else:
                visualizer = VIEW_CLASSES[view](bvh_file, debug=False, **viz_kwargs)
                owned_visualizers.append(visualizer)
                close_after = True

            print(f"\nGenerating {view} view...")
            visualizer.generate_video(
                output_file=output_file,
                start_time=start_time,
                end_time=end_time,
                output_fps=output_fps,
                video_size=video_size,
                show_info=(view == 'front'),
                close_plotter=close_after,
            )
    finally:
        for visualizer in owned_visualizers:
            plotter = getattr(visualizer, 'plotter', None)
            if plotter is not None:
                try:
                    plotter.close()
                except Exception:
                    pass

    return output_dir


def trim_video(input_file, output_file, start_time, end_time, target_fps=24):
    """Trim a video using FFmpeg and convert to target frame rate, also extract audio separately"""
    video_command = [
        FFMPEG_PATH, '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', input_file,
        '-c:v', 'libx264',
        '-r', str(target_fps),
        '-filter:v', f'fps={target_fps},setpts=PTS-STARTPTS',
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    print(f"\nExecuting video processing command:")
    print(" ".join(video_command))
    try:
        result = subprocess.run(video_command, check=True, capture_output=True, text=True)
        print("Video processing completed successfully")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing: {e}")
        print(f"Error output: {e.stderr}")
        raise

    audio_output = os.path.splitext(output_file)[0] + f'.mp3'
    audio_command = [
        FFMPEG_PATH, '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', input_file,
        '-q:a', '0',
        '-map', 'a',
        audio_output
    ]
    print(f"\nExecuting audio processing command:")
    print(" ".join(audio_command))
    try:
        result = subprocess.run(audio_command, check=True, capture_output=True, text=True)
        print("Audio processing completed successfully")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio processing: {e}")
        print(f"Error output: {e.stderr}")
        raise


def prepare_videos(
    filename,
    start_time,
    end_time,
    views_to_generate=['front'],
    video_path=None,
    video_size=(1280, 720),
    fps=24,
    output_dir=None,
    visualizers=None,
    **viz_kwargs,
):
    """Prepare skeleton view videos for one time window.

    ``viz_kwargs`` (frontal_method, markers, smooth_sec) apply only when
    ``visualizers`` is None (otherwise the passed-in visualizers' options win).
    """
    view_videos = {}
    print(
        f"Generating Skeleton views for {filename} | "
        f"Window: {start_time:.1f}s - {end_time:.1f}s"
    )

    for view in views_to_generate:
        output_dir_view = os.path.join(output_dir, f"{view}_view")
        os.makedirs(output_dir_view, exist_ok=True)

    for view in views_to_generate:
        output_dir_current = os.path.join(output_dir, f"{view}_view")
        view_path = os.path.join(
            output_dir_current,
            f"{view}_view_{start_time:.1f}_{end_time:.1f}.mp4",
        )
        if not is_valid_video_file(view_path):
            generate_individual_videos(
                bvh_file=filename + ".bvh",
                start_time=start_time,
                end_time=end_time,
                output_dir=output_dir_current,
                output_fps=fps,
                video_size=video_size,
                views_to_generate=[view],
                visualizers=visualizers,
                **viz_kwargs,
            )
        view_videos[view] = view_path

    return view_videos
