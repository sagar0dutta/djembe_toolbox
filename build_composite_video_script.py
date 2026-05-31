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
views_to_generate = ['front']       # skeleton views ['front', 'right' 'left', 'top'] 

layout_5views = [
    # Top row - side by side
    {'view': 'videos', 'x': 0, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'front_view', 'x': 960, 'y': 0, 'width': 960, 'height': 540},
    
    # Bottom row - stacked vertically
    {'view': 'Hips', 'x': 0, 'y': 540, 'width': 960, 'height': 270},
    {'view': 'plots', 'x': 0, 'y': 810, 'width': 960, 'height': 270},
    
    # {'view': 'drum_dot', 'x': 960, 'y': 540, 'width': 960, 'height': 540},
    {'view': 'drum_dot', 'x': 960, 'y': 540, 'width': 960, 'height': 270},
    {'view': 'dance_dot', 'x': 960, 'y': 810, 'width': 960, 'height': 270},
]

layout_2views = [
    # Top row 
    {'view': 'videos', 'x': 480, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'drum_dot', 'x': 0, 'y': 540, 'width': 1920, 'height': 540}, # bottom row

]

layouts_to_export = {
                    # "layout_2views": layout_2views,
                     "layout_5views": layout_5views
                     }

############################ CONFIGURATION #########################################

with open('data/selected_piece_list.pkl', 'rb') as f:
    piece_list = pickle.load(f)


mvnx_to_bvh = {
    'x': 'z',  # forward mvnx → backward bvh
    'y': 'x',  # side mvnx → side bvh
    'z': 'y',  # vertical mvnx → vertical bvh
}

m_idx = 1    # CHOOSE MODE
# mode_seg_idx = 0        # CHOOSE MODE SEGMENT IDX
mode = ["group", "individual", "audience"]
dance_mode = mode[m_idx]

joint_name = "Hips"  
axis = 'z'      # z is vertical in mvnx files


traj_dir  = "traj_files_presentation"
status    = "included"   # or "excluded"
traj_threshold = "0.001"        # or any other threshold

bvh_dir = os.path.join("data", "bvh_files")
motion_data_dir = "data/motion_data_pkl"

select_fps = 48
    
for file_name in piece_list:

    # file_name = "BKO_E1_D1_01_Suku"
    # if file_name not in ["BKO_E3_D6_01_Maraka", "BKO_E2_D4_06_Manjanin", "BKO_E2_D3_02_Suku"]:
    #     continue
    
    bvh_file = os.path.join(bvh_dir, file_name + "_T")

    # path to onsets and cycles csv files
    cycles_csv_path = f"data/virtual_cycles/{file_name}_C.csv"
    onsets_csv_path = f"data/drum_onsets/{file_name}.csv"
    dance_csv_path = f"data/dance_onsets/{file_name}_T_dance_onsets.csv"

    cyc_df = pd.read_csv(cycles_csv_path)
    cycle_onsets = cyc_df["Virtual Onset"].values

    
    
    dance_mode_path = f"data/dance_modes_ts/{file_name}_{dance_mode}.pkl"

    # load dance mode time segments
    if os.path.exists(dance_mode_path):
        with open(dance_mode_path, "rb") as f:
            dmode_ts = pickle.load(f)
    else:
        print(f"{file_name} {dance_mode} does not exist")
        continue
    
    nb_dmode_segs = len(dmode_ts)    
    
    
    for mode_seg_idx in range(nb_dmode_segs):
        mode_start_time, mode_end_time = dmode_ts[mode_seg_idx]

        print("Dance Mode:", dance_mode)
        print(f"Mode start time: {mode_start_time:.2f}")
        print(f"Mode end time: {mode_end_time:.2f}")

        start_time = mode_start_time    # CHOOSE START TIME
        end_time = mode_end_time        # CHOOSE END TIME

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
        
        # output_dir1 = os.path.join(base_output_dir, "video_skeleton")
        # os.makedirs(output_dir1, exist_ok=True)

        output_dir3 = os.path.join(base_output_dir, "drum_dot")
        os.makedirs(output_dir3, exist_ok=True)

        output_dir4 = os.path.join(base_output_dir, "dance_dot")
        os.makedirs(output_dir4, exist_ok=True)
        
        output_dir2 = os.path.join(base_output_dir, joint_name)
        os.makedirs(output_dir2, exist_ok=True)
        
        
        # foot and video mix
        extract_forward_cycle_videos_and_plots(
        file_name = file_name,
        windows = traj_tuples,  # List of (win_start, win_end, t_poi) tuples
        base_path_logs = "data/logs_v4_0.007_foot_jun3",            # logs_v4_0.007_foot_jun3       logs_v2_may
        figsize = (10, 3),
        dpi = 200,
        save_dir = base_output_dir,
        legend_flag = False,
        fps = select_fps,
        )

        
        for c_start_time, c_end_time, _ in traj_tuples:

            # Drum dot plot video
            animate_merged_phase_analysis_with_user_window(
                file_name= file_name,
                W_start= mode_start_time,   # dance mode start time
                W_end= mode_end_time,       # dance mode end time
                user_start= c_start_time,     # user defined start time
                user_end= c_end_time,         # user defined end time
                cycles_csv_path= cycles_csv_path,
                onsets_csv_path= onsets_csv_path,
                save_dir=output_dir3,
                figsize=(10, 3),
                dpi=200,
                legend_flag=False,
                fps=select_fps
            )


            # Dance dot plot video
            animate_dance_phase_analysis_with_user_window(
                file_name= file_name,
                W_start= mode_start_time,   # dance mode start time
                W_end= mode_end_time,       # dance mode end time
                user_start= c_start_time,     # user defined start time
                user_end= c_end_time,         # user defined end time
                cycles_csv_path= cycles_csv_path,
                dance_csv_path= dance_csv_path,
                save_dir=output_dir4,
                figsize=(10, 3),
                dpi=200,
                fps = select_fps
            )
            
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
        
        
        # Generate joint views
        extract_kinematic_cycle_plots(     
            file_name= file_name,
            windows= traj_tuples,
            joint_name= joint_name,
            axis= mvnx_to_bvh[axis],
            output_dir2= output_dir2,
            figsize = (10, 3),  # 2000 x 600 px
            dpi= 200,
            legend_flag = False,
            fps=select_fps,
            )
        # gc.collect()
        print("All individual videos generated.")    

        
        
        #################### CONCATENATE VIDEOS ####################
        
        required_folders = ['videos', 'plots', 'Hips', 'drum_dot', 'dance_dot',
                        *[f'{view}_view' for view in views_to_generate]
                        ]
        
        dynamic_concatenate_and_overlay_videos(
            folder_names= required_folders,
            save_dir=base_output_dir,
            views_to_generate=views_to_generate
        )
        
        # concatenate_and_overlay_videos(file_name, joint_name, base_output_dir, views_to_generate)        # modify
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
                'plots': concat_dict['plots'],    
                'Hips': concat_dict['Hips'],
                'drum_dot': concat_dict['drum_dot'],
                'dance_dot': concat_dict['dance_dot'],
                
                **{f'{view}_view': concat_dict[f'{view}_view'] for view in views_to_generate}
                }

            process_layouts(layouts_to_export, view_videos, base_output_dir, file_name, dance_mode, mode_start_time, mode_end_time)

        else:
            missing = [k for k in required_folders if k not in concat_dict]
            print(f"⚠️ Skipping {file_name}: missing videos → {missing}")