import os
import sys
import time
import numpy as np
import pandas as pd
import pyvista as pv
from bvh_converter import bvh_mod
from .worldpos_transform_v2 import (
    ensure_root_centered_worldpos_csv,
    file_lock,
    DEFAULT_FRONTAL_METHOD,
    DEFAULT_MARKERS,
    DEFAULT_SMOOTH_SEC,
)

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_JUPYTER"] = "false"

# Use system ffmpeg which has libx264 support
FFMPEG_PATH = "/itf-fi-ml/home/sagardu/bin/ffmpeg"




# Set environment variable for off-screen rendering
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

skeleton_indices = {
    'Hips': 0,
    'Chest': 1,
    'Chest2': 2,
    'Chest3': 3,
    'Chest4': 4,
    'Neck': 5,
    'Head': 6,
    'RightCollar': 7,
    'RightShoulder': 8,
    'RightElbow': 9,
    'RightWrist': 10,
    'LeftCollar': 11,
    'LeftShoulder': 12,
    'LeftElbow': 13,
    'LeftWrist': 14,
    'RightHip': 15,
    'RightKnee': 16,
    'RightAnkle': 17,
    'RightToe': 18,
    'LeftHip': 19,
    'LeftKnee': 20,
    'LeftAnkle': 21,
    'LeftToe': 22
}

class MocapVisualizerBase:
    def __init__(self, bvh_file, debug=False, frontal_method=DEFAULT_FRONTAL_METHOD,
                 markers=DEFAULT_MARKERS, smooth_sec=DEFAULT_SMOOTH_SEC):
        self.debug = debug
        self.frontal_method = frontal_method
        self.markers = markers
        self.smooth_sec = smooth_sec
        try:
            if not bvh_file.endswith('.bvh'):
                raise ValueError("Please provide a .bvh file")
            
            self.dir_csv = "bvh_to_csv"
            self.dir_csv_centered = "bvh_to_csv_centered"
            os.makedirs(self.dir_csv, exist_ok=True)
            os.makedirs(self.dir_csv_centered, exist_ok=True)
            self.load_bvh(bvh_file)
            
            # OSMesa first — more stable than EGL when several jobs render in parallel
            try:
                self.plotter = pv.Plotter(off_screen=True, backend='osmesa')
            except Exception:
                try:
                    self.plotter = pv.Plotter(off_screen=True, backend='egl')
                except Exception:
                    self.plotter = pv.Plotter(off_screen=True)

            # self.plotter._initialized = True
            self.plotter.show = lambda *a, **k: None
            # self.plotter.set_background('white')
            
            # Define connections for standard skeleton
            self.connections = [
                # Spine
                ('Hips', 'Chest'),
                ('Chest', 'Chest2'),
                ('Chest2', 'Chest3'),
                ('Chest3', 'Chest4'),
                ('Chest4', 'Neck'),
                ('Neck', 'Head'),
                # Right Arm
                ('Chest4', 'RightCollar'),
                ('RightCollar', 'RightShoulder'),
                ('RightShoulder', 'RightElbow'),
                ('RightElbow', 'RightWrist'),
                # Left Arm
                ('Chest4', 'LeftCollar'),
                ('LeftCollar', 'LeftShoulder'),
                ('LeftShoulder', 'LeftElbow'),
                ('LeftElbow', 'LeftWrist'),
                # Right Leg
                ('Hips', 'RightHip'),
                ('RightHip', 'RightKnee'),
                ('RightKnee', 'RightAnkle'),
                ('RightAnkle', 'RightToe'),
                # Left Leg
                ('Hips', 'LeftHip'),
                ('LeftHip', 'LeftKnee'),
                ('LeftKnee', 'LeftAnkle'),
                ('LeftAnkle', 'LeftToe'),
            ]
            
            
            
            # if self.debug:
            #     print("\nDebug Information:")
            #     print(f"Available markers: {self.labels}")
            #     print(f"T-pose positions: {self.t_pose_positions}")
            #     print(f"Connections: {self.connections}")
            
        except Exception as e:
            print(f"Error initializing visualizer: {str(e)}")
            sys.exit(1)

    def load_bvh(self, bvh_file):
        """Load data from BVH file"""
        # Convert BVH to CSV
        base_name = os.path.splitext(os.path.basename(bvh_file))[0] #os.path.splitext(bvh_file)[0]
        pos_csv = os.path.join(self.dir_csv, f"{base_name}_worldpos.csv")
        rot_csv = os.path.join(self.dir_csv, f"{base_name}_rotations.csv")
        
        lock_path = os.path.join(self.dir_csv, f".{base_name}.lock")
        with file_lock(lock_path):
            if not os.path.exists(pos_csv) or not os.path.exists(rot_csv):
                if self.debug:
                    print(f"Converting BVH to CSV files...")
                bvh_mod.convert_bvh_to_csv(bvh_file, output_dir=self.dir_csv, do_rotations=True)

            root_centered_csv = ensure_root_centered_worldpos_csv(
                pos_csv,
                dir_csv=self.dir_csv_centered,
                force=False,
                debug=self.debug,
                markers=self.markers,
                frontal_method=self.frontal_method,
                smooth_sec=self.smooth_sec,
            )

        # Load root-centered position data for skeleton / video rendering
        self.positions_df = pd.read_csv(root_centered_csv)
        self.uses_root_centered_data = True
        self.frame_rate = 240
        
        # Get marker names (excluding 'end' markers and Time column)
        self.labels = [col.split('.')[0] for col in self.positions_df.columns 
                      if col != 'Time' and not col.endswith('End')]
        self.labels = list(dict.fromkeys(self.labels))  # Remove duplicates
        
        # Get frame information
        self.total_frames = len(self.positions_df)
        self.total_time = self.total_frames / self.frame_rate
        
        # if self.debug:
        #     print(f"\nBVH file information:")
        #     print(f"Frame rate: {self.frame_rate} Hz")
        #     print(f"Total frames: {self.total_frames}")
        #     print(f"Total time: {self.total_time:.2f} seconds")
        #     print(f"Position columns: {self.positions_df.columns.tolist()}")
        #     print(f"Available markers: {self.labels}")
        
        # Store T-pose positions (first row after time column)
        self.t_pose_positions = {}
        for label in self.labels:
            x = self.positions_df.iloc[0][f"{label}.X"]
            y = self.positions_df.iloc[0][f"{label}.Y"]
            z = self.positions_df.iloc[0][f"{label}.Z"]
            self.t_pose_positions[label] = np.array([x, y, z])
        
        # Create label to index mapping for faster lookup
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Get reference position from first frame
        self.reference_pos = self.get_marker_position('Hips', 0)
        if self.reference_pos is None:
            raise ValueError("Reference marker 'Hips' not found in the first frame")

        self.display_scale = self._compute_display_scale(reference_frame=0)
    
    def _compute_display_scale(self, reference_frame=0):
        """Fixed scale from reference pose — avoids per-frame zoom pulsing."""
        points = []
        for label in self.labels:
            pos = self.get_marker_position(label, reference_frame)
            if pos is not None and not np.any(np.isnan(pos)):
                points.append(pos)
        if not points:
            return 1.0
        arr = np.array(points)
        hips = self.get_marker_position('Hips', reference_frame)
        if hips is not None:
            arr[:, 0] -= hips[0]
            arr[:, 2] -= hips[2]
        max_range = np.max(np.abs(arr))
        return max_range if max_range > 0 else 1.0

    def get_marker_position(self, label, frame):
        """Get the 3D position of a marker at a specific frame"""
        try:
            x = self.positions_df.iloc[frame][f"{label}.X"]
            y = self.positions_df.iloc[frame][f"{label}.Y"]
            z = self.positions_df.iloc[frame][f"{label}.Z"]
            return np.array([x, y, z])  # Keep original axis order: X (side), Y (vertical), Z (forward)
        except (KeyError, IndexError) as e:
            if self.debug:
                print(f"Warning: Marker {label} not found in frame {frame}: {str(e)}")
            return None
    
    def build_skeleton(self, frame):
        """Build the skeleton for a specific frame"""
        points = []
        lines = []
        
        current_pelvis = self.get_marker_position('Hips', frame)
        if current_pelvis is None:
            if self.debug:
                print(f"Warning: Missing pelvis in frame {frame}")
            return False

        # Collect valid marker positions (already root-centered per frame in CSV)
        marker_positions = {}
        for label in self.labels:
            pos = self.get_marker_position(label, frame)
            if pos is not None and not np.any(np.isnan(pos)):
                marker_positions[label] = pos
        
        if self.debug:
            print(f"Frame {frame} valid markers: {len(marker_positions)}/{len(self.labels)}")
        
        # Create points array and build connections
        points = []
        point_indices = {}
        current_idx = 0
        
        for start, end in self.connections:
            if start in marker_positions and end in marker_positions:
                # Add points if not already added
                if start not in point_indices:
                    points.append(marker_positions[start])
                    point_indices[start] = current_idx
                    current_idx += 1
                if end not in point_indices:
                    points.append(marker_positions[end])
                    point_indices[end] = current_idx
                    current_idx += 1
                
                # Add line connecting the points
                line = [2, point_indices[start], point_indices[end]]
                lines.append(line)
            elif self.debug:
                print(f"Warning: Missing connection {start} -> {end} in frame {frame}")
        
        if points:
            points = np.array(points)
            
            # Center on hips in X/Z only (Y unchanged; CSV is already hip-rooted)
            hips = marker_positions['Hips']
            points[:, 0] -= hips[0]
            points[:, 2] -= hips[2]

            # Fixed scale from reference frame (no per-frame normalize)
            if self.display_scale > 0:
                points = points / self.display_scale
            
            # Create the polydata with explicit cells array
            cells = []
            n_lines = len(lines)
            for line in lines:
                cells.extend(line)
            
            # Create cells array with proper format
            if n_lines > 0:
                cells = np.array(cells)
                self.skeleton = pv.PolyData(points, lines=cells)
                return True
            
        if self.debug:
            print(f"Warning: No valid skeleton built for frame {frame}")
        return False
    
    def generate_video(
        self,
        start_time=0,
        end_time=None,
        output_file="animation.mp4",
        output_fps=24,
        video_size=(1920, 1080),
        show_info=True,
        close_plotter=True,
    ):
        """Generate an MP4 video of the animation"""
        temp_dir = None
        try:
            # Convert times to frames
            start_frame = int(start_time * self.frame_rate)
            if end_time is None:
                end_frame = self.total_frames
            else:
                end_frame = int(end_time * self.frame_rate)
            
            # Ensure valid frame range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, self.total_frames))

            frame_step = max(1, int(self.frame_rate / output_fps))
            expected_frames = len(range(start_frame, end_frame, frame_step))

            temp_dir = os.path.join(
                "/tmp", f"temp_frames_{os.getpid()}_{time.time_ns()}"
            )
            os.makedirs(temp_dir, exist_ok=True)

            self.plotter.clear_actors()
            frame_count = 0
            failed_frames = []

            for frame in range(start_frame, end_frame, frame_step):
                if not self.build_skeleton(frame):
                    failed_frames.append(frame)
                    continue

                self.plotter.clear_actors()
                self.plotter.add_mesh(
                    self.skeleton, color='black', line_width=5, render_lines_as_tubes=True
                )
                self.plotter.add_points(self.skeleton.points, color='black', point_size=12)

                li = skeleton_indices['LeftAnkle']
                lj = skeleton_indices['LeftToe']
                ri = skeleton_indices['RightAnkle']
                rj = skeleton_indices['RightToe']

                if li < len(self.skeleton.points) and lj < len(self.skeleton.points):
                    left_line = pv.Line(self.skeleton.points[li], self.skeleton.points[lj])
                    self.plotter.add_mesh(
                        left_line, color='blue', line_width=5, render_lines_as_tubes=True
                    )

                if ri < len(self.skeleton.points) and rj < len(self.skeleton.points):
                    right_line = pv.Line(self.skeleton.points[ri], self.skeleton.points[rj])
                    self.plotter.add_mesh(
                        right_line, color='red', line_width=5, render_lines_as_tubes=True
                    )

                if show_info:
                    current_time = frame / self.frame_rate
                    self.plotter.add_text(
                        f'Time: {current_time:.0f}s', position='upper_left'
                    )

                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                self.plotter.screenshot(frame_path, window_size=video_size)
                frame_count += 1
                print(
                    f"Saved frame {frame_count}/{expected_frames}",
                    end='\r',
                )

            print()
            if failed_frames and self.debug:
                print(
                    f"Warning: build_skeleton failed for {len(failed_frames)} frames "
                    f"(first few: {failed_frames[:5]})"
                )

            if frame_count == 0:
                raise RuntimeError(
                    f"No frames rendered for {output_file} "
                    f"(window {start_time:.2f}-{end_time:.2f}s)"
                )
            if frame_count < expected_frames:
                raise RuntimeError(
                    f"Incomplete skeleton video {output_file}: "
                    f"expected {expected_frames} frames, got {frame_count}"
                )

            print("Converting frames to video...")
            ffmpeg_cmd = (
                f'{FFMPEG_PATH} -y -framerate {output_fps} '
                f'-i {temp_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p '
                f'-s {video_size[0]}x{video_size[1]} {output_file}'
            )
            ret = os.system(ffmpeg_cmd)
            if ret != 0:
                raise RuntimeError(f"ffmpeg failed (exit {ret}) for {output_file}")

            print(f"Video generation complete! Saved to {output_file}")
        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            raise
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
            if close_plotter and getattr(self, "plotter", None) is not None:
                self.plotter.close()
