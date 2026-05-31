from .mocap_visualizer_base import MocapVisualizerBase

class MocapVisualizerRightSide(MocapVisualizerBase):
    def __init__(self, bvh_file, debug=False):
        super().__init__(bvh_file, debug)
        # Right profile: camera on -X (BVH +X is stage left / dancer's left)
        self.plotter.camera_position = [(-2, 0, 0), (0, 0.5, 0), (0, 1, 0)]
        self.plotter.camera.zoom(1.0)

# if __name__ == "__main__":
#     # Create visualizer with debug enabled
#     visualizer = MocapVisualizerRightSide("BKO_E1_D2_03_Suku_T.bvh", debug=True)
#     # BKO_E1_D2_03_Suku_T     BKO_E1_D5_01_Maraka_T
#     # Generate video with time selection and custom size
#     visualizer.generate_video(
#         output_file="output_RS_view.mp4",
#         start_time=75.0,  # Start at 75 seconds
#         end_time=105.0,    # End at 78 seconds
#         output_fps=24,    # Set to standard video frame rate
#         video_size=(640, 480)  # Set video size
#     ) 