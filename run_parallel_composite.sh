#!/usr/bin/env bash

# ~/miniconda3/etc/profile.d/conda.sh
# source ~/.bashrc

MAX_JOBS=5
# Skeleton rendering fixes allow higher parallelism; try MAX_JOBS=5 if videos look correct.
RESUME_PENDING="${RESUME_PENDING:-1}"  # set to 0 to run all task ids 0..N-1

if [[ "$RESUME_PENDING" == "1" ]]; then
    mapfile -t TASK_IDS < <(python build_composite_video_script_slurm.py --pending)
    TOTAL_JOBS=${#TASK_IDS[@]}
    echo "Resuming $TOTAL_JOBS pending jobs (skip segments with final layout)"
else
    TOTAL_JOBS=$(python build_composite_video_script_slurm.py --count)
    TASK_IDS=()
    for ((i=0; i<TOTAL_JOBS; i++)); do
        TASK_IDS+=("$i")
    done
    echo "Running $TOTAL_JOBS jobs with max $MAX_JOBS in parallel"
fi

if [[ "$TOTAL_JOBS" -eq 0 ]]; then
    echo "No pending jobs."
    exit 0
fi

LOG_DIR="logs/composite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Running ${#TASK_IDS[@]} jobs with max $MAX_JOBS in parallel"
echo "Per-job logs (stdout+stderr): $LOG_DIR/job_<id>.log"

for i in "${TASK_IDS[@]}"; do
    echo "Launching job $i  (log: $LOG_DIR/job_$i.log)"
    python build_composite_video_script_slurm.py "$i" > "$LOG_DIR/job_$i.log" 2>&1 &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 2
    done
done

wait
echo "All jobs finished."
echo "Jobs that hit errors (see logs above + each segment's .composite_status.json):"
grep -lE "Traceback|Error" "$LOG_DIR"/job_*.log 2>/dev/null || echo "  (none detected)"

# Force full rerun of a single task:
#   python build_composite_video_script_slurm.py 5 --force
# Concat + layout only (per-cycle files must exist):
#   python build_composite_video_script_slurm.py 5 --concat-only
# Run all task ids (ignore pending filter):
#   RESUME_PENDING=0 ./run_parallel_composite.sh

# pkill -f run_parallel_composite.sh
# pkill -f build_composite_video_script_slurm.py
