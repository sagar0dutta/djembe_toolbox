import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

from utils_mocap_viz.generate_views import (
    prepare_videos,
    create_view_visualizers,
    close_view_visualizers,
    is_valid_video_file,
)
from utils_mocap_viz.animated_merged_phase_analysis import animate_merged_phase_analysis_with_user_window
from utils_dance_anim.dance_dot import animate_dance_phase_analysis_with_user_window
from utils_pipeline.pipeline_B import *
from utils_composite_video.by_time_segment import *


############################ LAYOUTS #########################################
views_to_generate = ['front']

# Frontal-alignment for skeleton views (utils_mocap_viz.worldpos_transform_v2):
#   FRONTAL_METHOD: "frame_smooth" (front-facing, de-jittered) | "frame" | "mean" | "none"
#   FRONTAL_MARKERS: "both" (hip+shoulder) | "shoulders" | "hips"
# Each option caches to its own CSV, so switching here rebuilds instead of reusing stale data.
FRONTAL_METHOD = "frame_smooth"
FRONTAL_MARKERS = "both"
FRONTAL_SMOOTH_SEC = 0.4

layout_5views = [
    {'view': 'videos', 'x': 0, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'front_view', 'x': 960, 'y': 0, 'width': 960, 'height': 540},
    {'view': 'Hips', 'x': 0, 'y': 540, 'width': 960, 'height': 270},
    {'view': 'plots', 'x': 0, 'y': 810, 'width': 960, 'height': 270},
    {'view': 'drum_dot', 'x': 960, 'y': 540, 'width': 960, 'height': 270},
    {'view': 'dance_dot', 'x': 960, 'y': 810, 'width': 960, 'height': 270},
]

layouts_to_export = {
    "layout_5views": layout_5views
}

############################ CONFIGURATION #########################################

mvnx_to_bvh = {
    'x': 'z',
    'y': 'x',
    'z': 'y',
}

m_idx = 1
mode = ["group", "individual", "audience"]
dance_mode = mode[m_idx]

joint_name = "Hips"
axis = 'z'

bvh_dir = os.path.join("data", "bvh_files")
select_fps = 48

STATUS_FILENAME = ".composite_status.json"
REQUIRED_FOLDERS = [
    'videos', 'plots', joint_name, 'drum_dot', 'dance_dot',
    *[f'{view}_view' for view in views_to_generate],
]

def build_job_list():
    """One job per (piece, dance-mode segment index)."""
    with open('data/selected_piece_list.pkl', 'rb') as f:
        piece_list = pickle.load(f)

    jobs = []
    for file_name in piece_list:
        dance_mode_path = f"data/dance_modes_ts/{file_name}_{dance_mode}.pkl"
        if not os.path.exists(dance_mode_path):
            continue
        with open(dance_mode_path, "rb") as f:
            dmode_ts = pickle.load(f)
        for mode_seg_idx in range(len(dmode_ts)):
            jobs.append((file_name, mode_seg_idx))
    return jobs


def final_layout_path(file_name, mode_start, mode_end, layout_name="layout_5views"):
    parent = os.path.join("composite_videos", dance_mode)
    return os.path.join(
        parent,
        f"{file_name}_{dance_mode}_{layout_name}_{mode_start}_{mode_end}.mp4",
    )


def status_path(base_output_dir):
    return os.path.join(base_output_dir, STATUS_FILENAME)


def load_status(base_output_dir):
    path = status_path(base_output_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_status(base_output_dir, payload):
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = status_path(base_output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def cycle_artifact_paths(file_name, cycle_idx, c_start, c_end, base_output_dir):
    """Expected per-cycle outputs (must match generator naming)."""
    i = cycle_idx + 1
    paths = {
        "video": os.path.join(
            base_output_dir, "videos",
            f"{file_name}_window_{i:03d}_{c_start:.2f}_{c_end:.2f}.mp4",
        ),
        "plot": os.path.join(
            base_output_dir, "plots",
            f"{file_name}_window_{i:03d}_{c_start:.2f}_{c_end:.2f}.mp4",
        ),
        "drum": os.path.join(
            base_output_dir, "drum_dot",
            f"drum_dot_merged_{c_start:.2f}_{c_end:.2f}.mp4",
        ),
        "dance": os.path.join(
            base_output_dir, "dance_dot",
            f"dance_{c_start:.2f}_{c_end:.2f}.mp4",
        ),
        "hips": os.path.join(
            base_output_dir, joint_name,
            f"{file_name}_window_{i:03d}_{c_start:.2f}_{c_end:.2f}.mp4",
        ),
    }
    for view in views_to_generate:
        paths[f"{view}_view"] = os.path.join(
            base_output_dir, f"{view}_view",
            f"{view}_view_{c_start:.1f}_{c_end:.1f}.mp4",
        )
    return paths


def all_paths_exist(paths):
    return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths)


def stage_forward_complete(file_name, traj_tuples, base_output_dir):
    if not traj_tuples:
        return True
    for idx, (c_start, c_end, _) in enumerate(traj_tuples):
        p = cycle_artifact_paths(file_name, idx, c_start, c_end, base_output_dir)
        if not all_paths_exist([p["video"], p["plot"]]):
            return False
    return True


def stage_per_cycle_complete(file_name, traj_tuples, base_output_dir):
    if not traj_tuples:
        return True
    for idx, (c_start, c_end, _) in enumerate(traj_tuples):
        p = cycle_artifact_paths(file_name, idx, c_start, c_end, base_output_dir)
        if not all_paths_exist([p["drum"], p["dance"]]):
            return False
        view_paths = [p[f"{v}_view"] for v in views_to_generate]
        if not all(is_valid_video_file(path) for path in view_paths):
            return False
    return True


def stage_hips_complete(file_name, traj_tuples, base_output_dir):
    if not traj_tuples:
        return True
    for idx, (c_start, c_end, _) in enumerate(traj_tuples):
        p = cycle_artifact_paths(file_name, idx, c_start, c_end, base_output_dir)
        if not os.path.isfile(p["hips"]) or os.path.getsize(p["hips"]) <= 0:
            return False
    return True


def build_concat_dict(base_output_dir):
    concat_file_list = [
        f for f in os.listdir(base_output_dir) if f.lower().endswith(".mp4")
    ]
    return {
        f.replace('_concat.mp4', ''): os.path.join(base_output_dir, f)
        for f in concat_file_list
        if f.endswith('_concat.mp4')
    }


def stage_concat_complete(base_output_dir):
    concat_dict = build_concat_dict(base_output_dir)
    return all(key in concat_dict for key in REQUIRED_FOLDERS)


def stage_layout_complete(file_name, mode_start, mode_end):
    return os.path.isfile(final_layout_path(file_name, mode_start, mode_end))


def resolve_job_context(task_id):
    job_list = build_job_list()
    if task_id >= len(job_list):
        raise IndexError(
            f"Task id {task_id} exceeds number of jobs ({len(job_list)})"
        )
    file_name, mode_seg_idx = job_list[task_id]

    cycles_csv_path = f"data/virtual_cycles/{file_name}_C.csv"
    cyc_df = pd.read_csv(cycles_csv_path)
    cycle_onsets = cyc_df["Virtual Onset"].values

    dance_mode_path = f"data/dance_modes_ts/{file_name}_{dance_mode}.pkl"
    with open(dance_mode_path, "rb") as f:
        dmode_ts = pickle.load(f)

    mode_start_time, mode_end_time = dmode_ts[mode_seg_idx]
    cycle_onsets_in_range = cycle_onsets[
        (cycle_onsets >= mode_start_time) & (cycle_onsets <= mode_end_time)
    ]
    traj_tuples = [
        (cycle_onsets_in_range[i], cycle_onsets_in_range[i + 1], 0)
        for i in range(len(cycle_onsets_in_range) - 1)
    ]

    base_output_dir = os.path.join(
        "composite_videos",
        dance_mode,
        f"{file_name}_{mode_start_time:.2f}_{mode_end_time:.2f}",
    )

    return {
        "task_id": task_id,
        "file_name": file_name,
        "mode_seg_idx": mode_seg_idx,
        "mode_start_time": mode_start_time,
        "mode_end_time": mode_end_time,
        "traj_tuples": traj_tuples,
        "base_output_dir": base_output_dir,
        "bvh_file": os.path.join(bvh_dir, file_name + "_T"),
        "cycles_csv_path": cycles_csv_path,
        "onsets_csv_path": f"data/drum_onsets/{file_name}.csv",
        "dance_csv_path": f"data/dance_onsets_v4_0.007_foot_jun3/{file_name}_T/onset_info/{file_name}_T_both_feet_onsets.csv",
    }


def list_pending_task_ids():
    pending = []
    for task_id in range(len(build_job_list())):
        ctx = resolve_job_context(task_id)
        if not stage_layout_complete(ctx["file_name"], ctx["mode_start_time"], ctx["mode_end_time"]):
            pending.append(task_id)
    return pending


def run_concat_and_layout(ctx, force=False):
    base_output_dir = ctx["base_output_dir"]
    file_name = ctx["file_name"]
    mode_start_time = ctx["mode_start_time"]
    mode_end_time = ctx["mode_end_time"]

    if not force and stage_concat_complete(base_output_dir):
        print("Concat outputs already complete, skipping dynamic_concatenate")
    else:
        dynamic_concatenate_and_overlay_videos(
            folder_names=REQUIRED_FOLDERS,
            save_dir=base_output_dir,
            views_to_generate=views_to_generate,
        )
        print("Concatenation finished")

    concat_dict = build_concat_dict(base_output_dir)
    if not all(key in concat_dict for key in REQUIRED_FOLDERS):
        missing = [k for k in REQUIRED_FOLDERS if k not in concat_dict]
        raise RuntimeError(f"Missing concat videos: {missing}")

    view_videos = {
        'videos': concat_dict['videos'],
        'plots': concat_dict['plots'],
        joint_name: concat_dict[joint_name],
        'drum_dot': concat_dict['drum_dot'],
        'dance_dot': concat_dict['dance_dot'],
        **{f'{view}_view': concat_dict[f'{view}_view'] for view in views_to_generate},
    }

    process_layouts(
        layouts_to_export,
        view_videos,
        base_output_dir,
        file_name,
        dance_mode,
        mode_start_time,
        mode_end_time,
        output_fps=select_fps,
    )


def run_task(task_id, force=False, concat_only=False):
    ctx = resolve_job_context(task_id)
    file_name = ctx["file_name"]
    mode_seg_idx = ctx["mode_seg_idx"]
    mode_start_time = ctx["mode_start_time"]
    mode_end_time = ctx["mode_end_time"]
    traj_tuples = ctx["traj_tuples"]
    base_output_dir = ctx["base_output_dir"]

    print(f"[TASK {task_id}] Processing {file_name} segment {mode_seg_idx} ({dance_mode})")
    print(f"Mode window: {mode_start_time:.2f} – {mode_end_time:.2f}")
    print(f"Cycles: {len(traj_tuples)}")
    print(f"Output: {base_output_dir}")

    os.makedirs(base_output_dir, exist_ok=True)

    if not force and stage_layout_complete(file_name, mode_start_time, mode_end_time):
        print(f"Final layout already exists, nothing to do.")
        save_status(base_output_dir, {
            "task_id": task_id,
            "file_name": file_name,
            "n_cycles": len(traj_tuples),
            "fps": select_fps,
            "stages": {
                "forward": "done",
                "per_cycle": "done",
                "hips": "done",
                "concat": "done",
                "layout": "done",
            },
            "complete": True,
        })
        return True

    per_cycle_ready = (
        stage_forward_complete(file_name, traj_tuples, base_output_dir)
        and stage_per_cycle_complete(file_name, traj_tuples, base_output_dir)
        and stage_hips_complete(file_name, traj_tuples, base_output_dir)
    )

    if concat_only:
        if not per_cycle_ready and not force:
            raise RuntimeError(
                "concat-only requested but per-cycle artifacts are incomplete; "
                "run full job or use --force after fixing inputs"
            )
        run_concat_and_layout(ctx, force=force)
        save_status(base_output_dir, {
            "task_id": task_id,
            "file_name": file_name,
            "n_cycles": len(traj_tuples),
            "fps": select_fps,
            "stages": {
                "forward": "done",
                "per_cycle": "done",
                "hips": "done",
                "concat": "done",
                "layout": "done",
            },
            "complete": True,
        })
        print(f"[TASK {task_id}] Completed (concat-only).")
        return True

    output_dir3 = os.path.join(base_output_dir, "drum_dot")
    output_dir4 = os.path.join(base_output_dir, "dance_dot")
    output_dir2 = os.path.join(base_output_dir, joint_name)
    for d in (output_dir2, output_dir3, output_dir4):
        os.makedirs(d, exist_ok=True)

    # --- Stage 1: foot videos + plots ---
    if force or not stage_forward_complete(file_name, traj_tuples, base_output_dir):
        print("Stage: forward videos + foot plots")
        extract_forward_cycle_videos_and_plots(
            file_name=file_name,
            windows=traj_tuples,
            base_path_logs="data/dance_onsets_v4_0.007_foot_jun3",
            figsize=(10, 3),
            dpi=200,
            save_dir=base_output_dir,
            legend_flag=False,
            fps=select_fps,
            skip_existing=not force,
        )
        save_status(base_output_dir, {
            "task_id": task_id,
            "file_name": file_name,
            "n_cycles": len(traj_tuples),
            "fps": select_fps,
            "stages": {"forward": "done"},
        })
    else:
        print("Stage: forward — already complete, skipping")

    # --- Stage 2: per-cycle drum, dance, skeleton ---
    if force or not stage_per_cycle_complete(file_name, traj_tuples, base_output_dir):
        print("Stage: per-cycle drum / dance / skeleton views")
        bvh_path = ctx["bvh_file"] + ".bvh"
        view_visualizers = create_view_visualizers(
            bvh_path, views_to_generate, debug=False,
            frontal_method=FRONTAL_METHOD,
            markers=FRONTAL_MARKERS,
            smooth_sec=FRONTAL_SMOOTH_SEC,
        )
        try:
            for cycle_idx, (c_start_time, c_end_time, _) in enumerate(traj_tuples):
                paths = cycle_artifact_paths(
                    file_name, cycle_idx, c_start_time, c_end_time, base_output_dir
                )

                if force or not os.path.isfile(paths["drum"]):
                    animate_merged_phase_analysis_with_user_window(
                        file_name=file_name,
                        W_start=mode_start_time,
                        W_end=mode_end_time,
                        user_start=c_start_time,
                        user_end=c_end_time,
                        cycles_csv_path=ctx["cycles_csv_path"],
                        onsets_csv_path=ctx["onsets_csv_path"],
                        save_dir=output_dir3,
                        figsize=(10, 3),
                        dpi=200,
                        legend_flag=False,
                        fps=select_fps,
                    )
                else:
                    print(f"  Skip drum cycle {cycle_idx + 1}: exists")

                if force or not os.path.isfile(paths["dance"]):
                    animate_dance_phase_analysis_with_user_window(
                        file_name=file_name,
                        W_start=mode_start_time,
                        W_end=mode_end_time,
                        user_start=c_start_time,
                        user_end=c_end_time,
                        cycles_csv_path=ctx["cycles_csv_path"],
                        dance_csv_path=ctx["dance_csv_path"],
                        save_dir=output_dir4,
                        figsize=(10, 3),
                        dpi=200,
                        fps=select_fps,
                    )
                else:
                    print(f"  Skip dance cycle {cycle_idx + 1}: exists")

                view_paths = [paths[f"{v}_view"] for v in views_to_generate]
                if force or not all(is_valid_video_file(p) for p in view_paths):
                    try:
                        prepare_videos(
                            filename=ctx["bvh_file"],
                            start_time=c_start_time,
                            end_time=c_end_time,
                            views_to_generate=views_to_generate,
                            video_path=None,
                            video_size=(1280, 720),
                            fps=select_fps,
                            output_dir=base_output_dir,
                            visualizers=view_visualizers,
                        )
                    except Exception as e:
                        print(
                            f"Error in prepare_videos for window "
                            f"{c_start_time:.2f}-{c_end_time:.2f}: {e}"
                        )
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  Skip skeleton cycle {cycle_idx + 1}: exists")

                save_status(base_output_dir, {
                    "task_id": task_id,
                    "file_name": file_name,
                    "n_cycles": len(traj_tuples),
                    "fps": select_fps,
                    "stages": {"forward": "done", "per_cycle": "in_progress"},
                    "last_cycle_completed": cycle_idx,
                })
        finally:
            close_view_visualizers(view_visualizers)
    else:
        print("Stage: per-cycle — already complete, skipping")

    # --- Stage 3: joint (Hips) batch ---
    if force or not stage_hips_complete(file_name, traj_tuples, base_output_dir):
        print("Stage: joint kinematic plots")
        extract_kinematic_cycle_plots(
            file_name=file_name,
            windows=traj_tuples,
            joint_name=joint_name,
            axis=mvnx_to_bvh[axis],
            output_dir2=output_dir2,
            figsize=(10, 3),
            dpi=200,
            legend_flag=False,
            fps=select_fps,
            skip_existing=not force,
        )
    else:
        print("Stage: hips — already complete, skipping")

    print("All individual videos generated.")

    # --- Stage 4–5: concat + layout ---
    run_concat_and_layout(ctx, force=force)

    save_status(base_output_dir, {
        "task_id": task_id,
        "file_name": file_name,
        "n_cycles": len(traj_tuples),
        "fps": select_fps,
        "stages": {
            "forward": "done",
            "per_cycle": "done",
            "hips": "done",
            "concat": "done",
            "layout": "done",
        },
        "complete": True,
    })
    print(f"[TASK {task_id}] Completed successfully.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build composite videos (one segment per task id)."
    )
    parser.add_argument(
        "task_id",
        nargs="?",
        help="Job index (0 .. N-1). Omit for --count / --pending.",
    )
    parser.add_argument("--count", action="store_true", help="Print number of jobs and exit")
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Print task ids missing final layout (one per line)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if outputs exist",
    )
    parser.add_argument(
        "--concat-only",
        action="store_true",
        help="Only concat + layout (requires per-cycle artifacts)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.count:
        print(len(build_job_list()))
        return

    if args.pending:
        for tid in list_pending_task_ids():
            print(tid)
        return

    if args.task_id is None:
        print("Usage: python build_composite_video_script_slurm.py <task_id>", file=sys.stderr)
        print("       python build_composite_video_script_slurm.py --count", file=sys.stderr)
        print("       python build_composite_video_script_slurm.py --pending", file=sys.stderr)
        sys.exit(2)

    task_id = int(args.task_id)
    try:
        ok = run_task(task_id, force=args.force, concat_only=args.concat_only)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            ctx = resolve_job_context(task_id)
            status = load_status(ctx["base_output_dir"])
            status["complete"] = False
            status["error"] = True
            status["error_message"] = f"{type(e).__name__}: {e}"
            status["traceback"] = tb
            status["failed_at"] = datetime.now(timezone.utc).isoformat()
            save_status(ctx["base_output_dir"], status)
        except Exception:
            pass
        print(tb, file=sys.stderr)
        raise

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
