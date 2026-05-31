from .mocap_visualizer_base import MocapVisualizerBase

class MocapVisualizerTop(MocapVisualizerBase):
    def __init__(self, bvh_file, debug=False):
        super().__init__(bvh_file, debug)
        # Set up fixed camera position for top view
        self.plotter.camera_position = [(0, 3, 0), (0, 0, 0), (0, 0, 1)]
        self.plotter.camera.zoom(0.5)

if __name__ == "__main__":
    # Create visualizer with debug enabled
    visualizer = MocapVisualizerTop("BKO_E1_D5_01_Maraka_T.bvh", debug=True)
    
    # Generate video with time selection and custom size
    visualizer.generate_video(
        output_file="output_T_view.mp4",
        start_time=75.0,  # Start at 75 seconds
        end_time=78.0,    # End at 78 seconds
        output_fps=24,    # Set to standard video frame rate
        video_size=(640, 480)  # Set video size
    ) 