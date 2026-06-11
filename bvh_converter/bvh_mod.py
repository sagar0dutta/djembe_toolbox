from __future__ import print_function, division
import sys
import csv
import os
import io

from .bvhplayer_skeleton import process_bvhfile, process_bvhkeyframe

def open_csv(filename, mode='r'):
    """Open a CSV file in proper mode depending on Python version."""
    if sys.version_info < (3,):
        return io.open(filename, mode=mode+'b')
    else:
        return io.open(filename, mode=mode, newline='')

# def convert_bvh_to_csv(file_in, do_rotations=False):
#     if not os.path.exists(file_in):
#         print("Error: file {} not found.".format(file_in))
#         return

#     print("Input filename: {}".format(file_in))
#     other_s = process_bvhfile(file_in)
    
#     print("Analyzing frames...")
#     for i in range(other_s.frames):
#         # Process each keyframe; update positions, rotations etc.
#         process_bvhkeyframe(other_s.keyframes[i], other_s.root, other_s.dt * i)
#     print("done")
    
#     # Write world positions CSV
#     file_out = file_in[:-4] + "_worldpos.csv"
#     with open_csv(file_out, 'w') as f:
#         writer = csv.writer(f)
#         header, frames = other_s.get_frames_worldpos()
#         writer.writerow(header)
#         for frame in frames:
#             writer.writerow(frame)
#     print("World Positions Output file: {}".format(file_out))
    
#     # Optionally, write rotations CSV
#     if do_rotations:
#         file_out = file_in[:-4] + "_rotations.csv"
#         with open_csv(file_out, 'w') as f:
#             writer = csv.writer(f)
#             header, frames = other_s.get_frames_rotations()
#             writer.writerow(header)
#             for frame in frames:
#                 writer.writerow(frame)
#         print("Rotations Output file: {}".format(file_out))


def convert_bvh_to_csv(file_in, output_dir = "bvh_to_csv", do_rotations=True):
    """
    Convert a BVH file to CSV files of world positions and optionally rotations, writing outputs to the specified directory.

    Args:
        file_in (str): Path to the input BVH file.
        output_dir (str): Directory where output CSV files will be saved.
        do_rotations (bool): If True, also write a rotations CSV.
    """
    # Check input file
    if not os.path.exists(file_in):
        print(f"Error: file '{file_in}' not found.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Input filename: {file_in}")
    other_s = process_bvhfile(file_in)

    print("Analyzing frames...")
    for i in range(other_s.frames):
        process_bvhkeyframe(other_s.keyframes[i], other_s.root, other_s.dt * i)
    print("Frame analysis done")

    # Prepare base name for outputs
    base_name = os.path.splitext(os.path.basename(file_in))[0]

    # Write world positions CSV
    worldpos_file = os.path.join(output_dir, f"{base_name}_worldpos.csv")
    with open_csv(worldpos_file, 'w') as f:
        writer = csv.writer(f)
        header, frames = other_s.get_frames_worldpos()
        writer.writerow(header)
        for frame in frames:
            writer.writerow(frame)
    print(f"World Positions Output file: {worldpos_file}")

    # Optionally, write rotations CSV
    if do_rotations:
        rotations_file = os.path.join(output_dir, f"{base_name}_rotations.csv")
        with open_csv(rotations_file, 'w') as f:
            writer = csv.writer(f)
            header, frames = other_s.get_frames_rotations()
            writer.writerow(header)
            for frame in frames:
                writer.writerow(frame)
        print(f"Rotations Output file: {rotations_file}")