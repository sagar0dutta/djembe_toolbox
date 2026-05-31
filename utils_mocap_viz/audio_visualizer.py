import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import subprocess
import sys
from matplotlib.animation import FuncAnimation

class AudioVisualizer:
    def __init__(self, audio_file, debug=False):
        self.debug = debug
        self.audio_file = audio_file
        
        # Load audio file
        self.y, self.sr = librosa.load(audio_file)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        if self.debug:
            print(f"\nAudio file information:")
            print(f"Sample rate: {self.sr} Hz")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Number of samples: {len(self.y)}")
    
    def generate_video(self, start_time=0, end_time=None, output_file="audio_visualization.mp4", 
                      output_fps=24, video_size=(1280, 720)):
        """Generate a video of the audio waveform with a moving time marker"""
        try:
            # Convert times to samples
            start_sample = int(start_time * self.sr)
            if end_time is None:
                end_sample = len(self.y)
            else:
                end_sample = int(end_time * self.sr)
            
            # Ensure valid sample range
            start_sample = max(0, min(start_sample, len(self.y) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(self.y)))
            
            # Calculate duration and number of frames
            duration = end_time - start_time
            num_frames = int(duration * output_fps)
            
            # Calculate samples per frame
            samples_per_frame = int(self.sr / output_fps)
            
            print(f"\nVideo generation settings:")
            print(f"Start time: {start_time:.2f}s")
            print(f"End time: {end_time:.2f}s")
            print(f"Duration: {duration:.3f}s")
            print(f"Output file: {output_file}")
            print(f"Output FPS: {output_fps}")
            print(f"Video size: {video_size[0]}x{video_size[1]}")
            print(f"Number of frames: {num_frames}")
            print(f"Samples per frame: {samples_per_frame}")

            # Create temporary directory for frames
            temp_dir = "temp_audio_frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Set up the figure
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(video_size[0]/100, video_size[1]/100), dpi=100)
            
            # Plot the full waveform in background
            times = np.linspace(start_time, end_time, end_sample - start_sample)
            ax.plot(times, self.y[start_sample:end_sample], color='cyan', alpha=0.3)
            
            # Add time marker line
            time_marker = ax.axvline(x=start_time, color='red', linewidth=2)
            
            # Customize the plot
            ax.set_xlim(start_time, end_time)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Audio Waveform')
            ax.grid(True, alpha=0.3)
            
            # Function to update the time marker and show current frame's samples
            def update(frame):
                # Calculate current time
                current_time = start_time + (frame / output_fps)
                
                # Calculate sample range for this frame
                frame_start_sample = start_sample + int(frame * samples_per_frame)
                frame_end_sample = min(frame_start_sample + samples_per_frame, end_sample)
                
                # Clear the axis and redraw everything
                ax.clear()
                
                # Redraw the background waveform
                times = np.linspace(start_time, end_time, end_sample - start_sample)
                ax.plot(times, self.y[start_sample:end_sample], color='cyan', alpha=0.3)
                
                # Plot current frame's samples
                frame_times = np.linspace(
                    current_time,
                    current_time + (1/output_fps),
                    frame_end_sample - frame_start_sample
                )
                ax.plot(frame_times, self.y[frame_start_sample:frame_end_sample], 
                       color='cyan', linewidth=2)
                
                # Add time marker
                time_marker = ax.axvline(x=current_time, color='red', linewidth=2)
                
                # Redraw plot elements
                ax.set_xlim(start_time, end_time)
                ax.set_ylim(-1, 1)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Audio Waveform')
                ax.grid(True, alpha=0.3)
                
                return [time_marker]
            
            # Create animation
            anim = FuncAnimation(fig, update, frames=num_frames, 
                               interval=1000/output_fps, blit=True)
            
            # Save frames
            for i in range(num_frames):
                # Update the animation
                anim._step()
                # Save the current frame
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
                print(f"Saved frame {i+1}/{num_frames}", end='\r')
            
            plt.close()
            
            print("\nConverting frames to video...")
            
            # Use ffmpeg to create video from frames
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(output_fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-ss', str(start_time),  # Start time for audio
                '-i', self.audio_file,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-pix_fmt', 'yuv420p',
                '-s', f'{video_size[0]}x{video_size[1]}',
                '-t', str(duration),  # Exact duration
                output_file
            ]
            
            subprocess.run(ffmpeg_cmd)
            
            # Clean up temporary files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            
            print(f"Video generation complete! Saved to {output_file}")
            
        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            plt.close()
            sys.exit(1)

if __name__ == "__main__":
    # Example usage
    audio_file = "BKO_E1_D5_01_Maraka_M-Jem-1.wav"
    visualizer = AudioVisualizer(audio_file, debug=True)
    
    # Generate video for a specific time range
    visualizer.generate_video(
        start_time=80.0,  # Start at 70 seconds
        end_time=100.0,    # End at 90 seconds
        output_file="output/audio_visualization.mp4",
        output_fps=24,    # Match skeleton visualization frame rate
        video_size=(1280, 720)
    ) 