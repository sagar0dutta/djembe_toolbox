import os
import gc
import shutil
import pickle
import subprocess
# import pandas as pd
from utils_mocap_viz.generate_views import (    # organize this
    # get_output_dir,
    prepare_videos
)

from utils_mocap_viz.animated_merged_phase_analysis import animate_merged_phase_analysis, animate_merged_phase_analysis_with_user_window
from utils_dance_anim.dance_dot import animate_dance_phase_analysis, animate_dance_phase_analysis_with_user_window
from utils_pipeline.pipeline_B import *
from utils_composite_video.by_time_segment import *



############################ LAYOUTS #########################################
views_to_generate = ['front', 'left']       # skeleton views ['front', 'right' 'left', 'top'] 

layout_5views = [
    # Top row - side by side
    {'view': 'videos', 'x': 0, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'front_view', 'x': 960, 'y': 0, 'width': 960, 'height': 540},
    
    # Bottom row - stacked vertically
    {'view': 'hand_distance', 'x': 0, 'y': 540, 'width': 960, 'height': 270},
    {'view': 'hand_velocity', 'x': 0, 'y': 810, 'width': 960, 'height': 270},
    
    {'view': 'left_view', 'x': 960, 'y': 540, 'width': 960, 'height': 540},
]

layout_6views = [
    # Top row - side by side
    {'view': 'videos', 'x': 0, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'front_view', 'x': 960, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'left_view', 'x': 960, 'y': 540, 'width': 960, 'height': 540},
    
    # Bottom row - stacked vertically
    {'view': 'hand_distance', 'x': 0, 'y': 540, 'width': 960, 'height': 180},
    {'view': 'hand_velocity', 'x': 0, 'y': 720, 'width': 960, 'height': 180},
    {'view': 'hand_acceleration', 'x': 0, 'y': 900, 'width': 960, 'height': 180},
    
]

layouts_to_export = {
                    #  "layout_5views": layout_5views,
                     "layout_6views": layout_6views
                     
                     }

############################ CONFIGURATION #########################################
piece_list = ['BKO_E1_D1_01_Suku','BKO_E1_D1_01_Suku','BKO_E1_D1_01_Suku','BKO_E1_D1_01_Suku','BKO_E1_D1_02_Maraka','BKO_E1_D1_02_Maraka','BKO_E1_D1_02_Maraka','BKO_E1_D1_02_Maraka','BKO_E1_D1_03_Wasulunka','BKO_E1_D1_06_Manjanin','BKO_E1_D2_03_Suku','BKO_E1_D2_05_Wasulunka','BKO_E2_D3_01_Maraka','BKO_E2_D3_01_Maraka','BKO_E2_D3_01_Maraka','BKO_E2_D3_01_Maraka','BKO_E2_D3_01_Maraka','BKO_E2_D3_02_Suku','BKO_E2_D3_02_Suku','BKO_E2_D3_02_Suku','BKO_E2_D3_02_Suku','BKO_E2_D3_02_Suku','BKO_E2_D3_02_Suku','BKO_E2_D3_02_Suku','BKO_E2_D3_03_Wasulunka','BKO_E2_D3_03_Wasulunka','BKO_E2_D3_06_Manjanin','BKO_E2_D3_06_Manjanin','BKO_E2_D3_06_Manjanin','BKO_E2_D3_06_Manjanin','BKO_E2_D3_14_Maraka','BKO_E2_D3_14_Maraka','BKO_E2_D4_01_Suku','BKO_E2_D4_01_Suku','BKO_E2_D4_01_Suku','BKO_E2_D4_01_Suku','BKO_E2_D4_02_Maraka','BKO_E2_D4_02_Maraka','BKO_E2_D4_03_Wasulunka','BKO_E2_D4_03_Wasulunka','BKO_E2_D4_03_Wasulunka','BKO_E2_D4_06_Manjanin','BKO_E2_D4_06_Manjanin','BKO_E2_D4_06_Manjanin','BKO_E2_D4_06_Manjanin','BKO_E2_D4_06_Manjanin','BKO_E2_D4_12_Suku','BKO_E2_D4_12_Suku','BKO_E2_D4_12_Suku','BKO_E2_D4_12_Suku','BKO_E3_D5_01_Maraka','BKO_E3_D5_02_Suku','BKO_E3_D5_02_Suku','BKO_E3_D5_03_Wasulunka','BKO_E3_D5_03_Wasulunka','BKO_E3_D5_03_Wasulunka','BKO_E3_D5_06_Manjanin','BKO_E3_D5_06_Manjanin','BKO_E3_D5_06_Manjanin','BKO_E3_D5_06_Manjanin','BKO_E3_D5_13_Suku','BKO_E3_D5_13_Suku','BKO_E3_D6_01_Maraka','BKO_E3_D6_01_Maraka','BKO_E3_D6_01_Maraka','BKO_E3_D6_02_Suku','BKO_E3_D6_02_Suku','BKO_E3_D6_02_Suku','BKO_E3_D6_06_Manjanin','BKO_E3_D6_12_Suku','BKO_E3_D6_12_Suku']
time_segments = [ (92.085,116.46), (112.76,143.14), (178.1,198.288), (195.16,221.579), (65.2,137.78), (137.78,156.18), (155.46,168.58), (205.9,268.06), (241.9,288.16), (146.28,186.74), (113.4,153.54), (78.8,123.42), (81.8,92.32), (123.16,145.3), (160.1,169.74), (223.28,236.52), (224.22,238.44), (116.28,130.52), (171.86,186.96), (185.74,214.44), (187.74,216.36), (233.12,250.68), (233.74,250.84), (250.46,265.58), (167.2,185.48), (167.22,185.44), (86.34,101.24), (99.46,118.86), (178.42,191.32), (192.36,246.76), (73.42,93.74), (91.38,101.46), (145.1,155.46), (158.0,172.88), (206.98,230.38), (232.42,250.64), (94.9,106.88), (121.42,133.84), (88.94,103.68), (129.62,139.0), (160.54,185.76), (99.36,116.24), (137.06,150.9), (148.08,164.52), (169.82,191.06), (169.84,185.66), (72.34,84.3), (72.62,85.92), (85.42,98.32), (115.16,137.04), (70.98,93.5), (68.02,86.42), (193.74,210.76), (128.24,148.1), (149.16,195.1), (171.18,186.28), (125.64,182.0), (211.26,261.12), (259.58,280.0), (279.0,316.58), (71.92,91.72), (120.88,162.72), (51.42,70.02), (69.7,101.7), (123.1,152.0), (70.78,90.34), (90.34,114.86), (181.28,200.06), (286.56,307.74), (159.46,179.44), (249.24,271.08) ]

skip_pieces = [
    ("BKO_E2_D3_02_Suku", (185.74, 214.44)),
    ("BKO_E2_D3_02_Suku", (187.74, 216.36)),
    ("BKO_E2_D3_02_Suku", (233.12, 250.68)),
    ("BKO_E2_D3_02_Suku", (233.74, 250.84)),
    ("BKO_E2_D3_03_Wasulunka", (167.20, 185.48)),
    ("BKO_E2_D3_03_Wasulunka", (167.22, 185.44)),
    ("BKO_E2_D4_06_Manjanin", (169.82, 191.06)),
    ("BKO_E2_D4_06_Manjanin", (169.84, 185.66)),
    ("BKO_E2_D4_12_Suku", (72.34, 84.30)),
    ("BKO_E2_D4_12_Suku", (72.62, 85.92)),
    ("BKO_E2_D4_12_Suku", (115.16, 137.04)),
    ("BKO_E3_D6_12_Suku", (159.46, 179.44)),
    ("BKO_E3_D6_12_Suku", (249.24, 271.08)),
]


m_idx = 2    # CHOOSE MODE
mode = ["group", "individual", "audience"]
dance_mode = mode[m_idx]


traj_dir  = "traj_files_presentation"
status    = "included"   # or "excluded"
traj_threshold = "0.001"        # or any other threshold

bvh_dir = os.path.join("data", "bvh_files")
motion_data_dir = "data/motion_data_pkl"

select_fps = 48
    
for file_name, time_segment in zip(piece_list, time_segments):
    
    if (file_name, time_segment) in skip_pieces:
        print(f"Skipping {file_name} {time_segment} as per skip list.")
        continue
    
    bvh_file = os.path.join(bvh_dir, file_name + "_T")

    # path to onsets and cycles csv files
    cycles_csv_path = f"data/virtual_cycles/{file_name}_C.csv"
    onsets_csv_path = f"data/drum_onsets/{file_name}.csv"
    dance_csv_path = f"data/dance_onsets/{file_name}_T_dance_onsets.csv"

    cyc_df = pd.read_csv(cycles_csv_path)
    cycle_onsets = cyc_df["Virtual Onset"].values

    
    
    dance_mode_path = f"data/dance_modes_ts/{file_name}_{dance_mode}.pkl"
    print("Dance Mode:", dance_mode)
    
    start_time = int(time_segment[0])
    end_time = int(time_segment[1])

    # get cycle onsets in range 
    cycle_onsets_in_range = cycle_onsets[(cycle_onsets >= start_time) & (cycle_onsets <= end_time)] 
    traj_tuples = [ (cycle_onsets_in_range[i], cycle_onsets_in_range[i+1], 0) for i in range(len(cycle_onsets_in_range)-1)]
    
    print("total number of cycles in range:", len(traj_tuples))
    print("\n")

    base_output_dir = os.path.join("composite_videos", dance_mode, f"{file_name}_{start_time:.2f}_{end_time:.2f}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Check if any contents exist in the output directory
    if os.path.exists(base_output_dir) and os.listdir(base_output_dir):
        print(f"Contents already exist in: {base_output_dir}")
        continue
    

    
    output_dir5 = os.path.join(base_output_dir, "hand_distance")
    os.makedirs(output_dir5, exist_ok=True)
    
    output_dir6 = os.path.join(base_output_dir, "hand_velocity")
    os.makedirs(output_dir6, exist_ok=True)
    
    output_dir7 = os.path.join(base_output_dir, "hand_acceleration")
    os.makedirs(output_dir7, exist_ok=True)
    
    
    extract_trimmed_cycle_videos(
    file_name = file_name,
    windows = traj_tuples,  # List of (win_start, win_end, t_poi) tuples
    save_dir = base_output_dir,
    export_fps = select_fps,
    )

    
    for c_start_time, c_end_time, _ in traj_tuples:
        
        try:
            # Generate Skeleton views
            view_videos = prepare_videos(
                filename= bvh_file,
                start_time= c_start_time,
                end_time= c_end_time,
                views_to_generate = views_to_generate,
                video_path= None,             # video_path, wont generate video
                video_size= (1280, 720),
                fps= select_fps,
                output_dir = base_output_dir,  
                )
            
            # Explicitly delete the return value if it contains large objects
            # del view_videos
            
            # Force garbage collection to free memory (especially DataFrames and PyVista plotters)
            # gc.collect()
            
        except Exception as e:
            print(f"Error in prepare_videos for window {c_start_time:.2f}-{c_end_time:.2f}: {e}")
            import traceback
            traceback.print_exc()
            # Force cleanup even on error
            # gc.collect()
            continue
    
    
    extract_hand_distance_cycle_plots(
        file_name=file_name,
        windows=traj_tuples,
        export_fps= select_fps,
        output_dir=output_dir5,
        motion_feature=0  # If 0 - distance, if 1 - relative velocity, if 2 - relative acceleration
        )
    
    extract_hand_distance_cycle_plots(
        file_name=file_name,
        windows=traj_tuples,
        export_fps= select_fps,
        output_dir=output_dir6,
        motion_feature=1  # If 0 - distance, if 1 - relative velocity, if 2 - relative acceleration
        )
    
    extract_hand_distance_cycle_plots(
        file_name=file_name,
        windows=traj_tuples,
        export_fps= select_fps,
        output_dir=output_dir7,
        motion_feature=2  # If 0 - distance, if 1 - relative velocity, if 2 - relative acceleration
        )
    
    
    
    # gc.collect()
    print("All individual videos generated.")    

    #################### CONCATENATE VIDEOS ####################
    
    required_folders = ['videos', 'hand_distance', 'hand_velocity', "hand_acceleration",
                    *[f'{view}_view' for view in views_to_generate]
                    ]
    
    
    dynamic_concatenate_and_overlay_videos(
        folder_names= required_folders,
        save_dir=base_output_dir,
        views_to_generate=views_to_generate
    )
    
    # concatenate_and_overlay_videos(joint_name, base_output_dir, views_to_generate)        # modify
    print("All concatenated videos generated.")
    
    
    concat_file_list = [f for f in os.listdir(base_output_dir) if f.lower().endswith(".mp4")]
    concat_dict = {
        f.replace('_concat.mp4', ''): os.path.join(base_output_dir, f) 
        for f in concat_file_list
        }

    
    # Check if all required keys exist in concat_dict
    if all(key in concat_dict for key in required_folders):
        view_videos = {
            'videos': concat_dict['videos'],  
            'hand_distance': concat_dict['hand_distance'],
            'hand_velocity': concat_dict['hand_velocity'],
            'hand_acceleration': concat_dict['hand_acceleration'],
            # 'drum_dot_merged': concat_dict['drum_dot_merged'],
            # 'dance_dot': concat_dict['dance_dot'],
            
            **{f'{view}_view': concat_dict[f'{view}_view'] for view in views_to_generate}
            }

        process_layouts(layouts_to_export, view_videos, base_output_dir, file_name, dance_mode, start_time, end_time)

    else:
        missing = [k for k in required_folders if k not in concat_dict]
        print(f"⚠️ Skipping {file_name}: missing videos → {missing}")