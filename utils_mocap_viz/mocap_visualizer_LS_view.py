from .mocap_visualizer_base import MocapVisualizerBase

class MocapVisualizerLeftSide(MocapVisualizerBase):
    def __init__(self, bvh_file, debug=False, **kwargs):
        super().__init__(bvh_file, debug, **kwargs)
        # Left profile: camera on +X
        self.plotter.camera_position = [(2, 0, 0), (0, 0.5, 0), (0, 1, 0)]
        self.plotter.camera.zoom(1.0)

# if __name__ == "__main__":
#     # Create visualizer with debug enabled
#     visualizer = MocapVisualizerLeftSide("BKO_E1_D5_01_Maraka_T.bvh", debug=True)
    
#     # Generate video with time selection and custom size
#     visualizer.generate_video(
#         output_file="output_LS_view.mp4",
#         start_time=75.0,  # Start at 75 seconds
#         end_time=78.0,    # End at 78 seconds
#         output_fps=24,    # Set to standard video frame rate
#         video_size=(640, 480)  # Set video size
#     ) 