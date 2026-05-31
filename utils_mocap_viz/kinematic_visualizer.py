import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

def visualize_joint_position(
    bvh_file,
    joint_name,
    start_time,
    end_time,
    axis='x',  # 'x', 'y', or 'z'
    output_fps=24,
    output_dir=None,
    fig_size=(1280, 360),
    dpi=200,
):
    """
    Visualize joint position data for a single axis with a play head
    
    Args:
        bvh_file: Path to BVH file
        joint_name: Name of joint to visualize (e.g., 'Hips', 'RightHand')
        axis: Which axis to plot ('x', 'y', or 'z')
        start_time: Start time in seconds
        end_time: End time in seconds
        output_fps: Output video FPS
        output_dir: Directory to save output files
        fig_size: Output figure size (width, height)
    """
    print(f"\nStarting joint position visualization for {joint_name} along {axis}-axis")
    print(f"Time range: {start_time:.2f}s to {end_time:.2f}s")
    
    # 1. Get output directory
    if output_dir is None:
        filename = os.path.splitext(os.path.basename(bvh_file))[0]
        dir_name = f"{filename}_{start_time:.1f}_{end_time:.1f}"
        output_dir = os.path.join("output", dir_name)
        os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 2. Load CSV file from the same directory as BVH file
    dir_csv = "extracted_mocap_csv"
    base_name = os.path.splitext(os.path.basename(bvh_file))[0]
    worldpos_file = os.path.join(dir_csv, f"{base_name}_worldpos.csv")
    print(f"Loading world positions from: {worldpos_file}")
    
    try:
        world_positions = pd.read_csv(worldpos_file)
        print(f"Successfully loaded CSV with {len(world_positions)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    # 3. Get time column and find frame indices
    time_column = world_positions.columns[0]  # First column is time
    mask = (world_positions[time_column] >= start_time) & (world_positions[time_column] <= end_time)
    filtered_data = world_positions[mask]
    print(f"Filtered data points: {len(filtered_data)}")
    
    # Get position data for specified axis
    time_data = filtered_data[time_column]
    axis_data = filtered_data[f"{joint_name}.{axis.upper()}"]
    print(f"Data range for {axis}-axis: {axis_data.min():.2f} to {axis_data.max():.2f}")
    
    # 4. Create the plot
    print("\nCreating visualization plot...")
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    
    # Plot position data
    ax.plot(time_data, axis_data, 'b-', label=f'{axis.upper()}-position')
    
    # Add play head line
    play_head = ax.axvline(x=start_time, color='r', linestyle='--')
    
    # Add title and labels
    ax.set_title(f'Position of {joint_name} along {axis.upper()}-axis', fontsize=12, pad=10)
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel(f'{axis.upper()}-position', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Create animation
    print("Creating animation...")
    def update(frame):
        current_time = start_time + (frame / output_fps)
        play_head.set_xdata([current_time, current_time])
        return play_head,
    
    # 6. Save animation
    total_frames = int((end_time - start_time) * output_fps)
    print(f"Total frames to generate: {total_frames}")
    anim = FuncAnimation(fig, update, frames=total_frames,
                        interval=1000/output_fps, blit=True)
    
    # 7. Save video
    output_file = os.path.join(output_dir, f"{joint_name}_{axis}_position_{start_time:.1f}_{end_time:.1f}.mp4")

    print(f"\nSaving animation to: {output_file}")
    try:
        anim.save(output_file, writer='ffmpeg', fps=output_fps)
        print("Animation saved successfully")
    except Exception as e:
        print(f"Error saving animation: {e}")
        raise
    finally:
        plt.close()
    
    return output_file 