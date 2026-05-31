#!/usr/bin/env bash

# ~/miniconda3/etc/profile.d/conda.sh
# source ~/.bashrc


MAX_JOBS=10      # 🔴 start conservative (8 is safe)
# TOTAL_JOBS=3
TOTAL_JOBS=71   # you have 71 file_name entries

echo "Running $TOTAL_JOBS jobs with max $MAX_JOBS in parallel"

for ((i=0; i<TOTAL_JOBS; i++)); do
    echo "Launching job $i"
    python build_composite_video_script_Hand_Only_slurm.py $i &

  #   python build_composite_video_script_Hand_Only_slurm.py $i \
  # > logs/job_$i.out 2> logs/job_$i.err &


    # Throttle: wait if too many jobs are running
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 2
    done
done

wait
echo "All jobs finished."

# rsync -avz -e 'ssh' composite_videos_18Dec /itf-fi-ml/home/sagardu/djembe_drive/Hand-Clapping-Plots

# pkill -f build_composite_video_script_Hand_Only_slurm.py

# ~/miniconda3/etc/profile.d/conda.sh
# source ~/.bashrc
