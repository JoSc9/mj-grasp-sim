#!/usr/bin/env bash
# Parallel scene runner for 10K scenes with random num_objects
# Usage:
#  ./gen_scenes_10k.sh [-p PARALLEL] [-o OUTDIR] [--stop-on-failure]
#
# Generates 10,000 scenes with random num_objects parameter (1-7) for each scene

set -u

TOTAL=15000
PARALLEL=80
OUTDIR=parallel_runs_10k
BASE_CMD="python -m mgs.cli.gen_scene"
STOP_ON_FAILURE=0

usage() {
  cat <<EOF
Usage: $0 [-p PARALLEL] [-o OUTDIR] [--stop-on-failure]

Run $TOTAL scene jobs in parallel with up to PARALLEL concurrent jobs.
Each scene will be generated with a random num_objects parameter between 1-7.

Each job's stdout+stderr is written to OUTDIR/job_<i>.log
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--parallel)
      PARALLEL="$2"; shift 2;;
    -o|--outdir)
      OUTDIR="$2"; shift 2;;
    --stop-on-failure)
      STOP_ON_FAILURE=1; shift;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1" >&2; usage;;
  esac
done

mkdir -p "$OUTDIR"

FIFO="/tmp/gen_scenes_10k_fifo_$$"
rm -f "$FIFO"
mkfifo "$FIFO"

cleanup() {
  rm -f "$FIFO"
}
trap cleanup EXIT

spawn_job() {
  local i=$1
  # Sample random number between 1 and 7 for num_objects
  local num_objects=$((RANDOM % 5 + 1))
  local cmd="$BASE_CMD num_objects=$num_objects"
  local log="${OUTDIR}/job_${i}.log"
  # run in background; when finished write "i:rc" to FIFO
  (
    echo "[JOB $i] START: $cmd (num_objects=$num_objects)" > "$log"
    eval "$cmd" >> "$log" 2>&1
    rc=$?
    echo "$i:$rc" > "$FIFO"
  ) &
}

submitted=0
completed=0
successful=0
failed=0

# start initial batch
while [[ $submitted -lt $TOTAL && $submitted -lt $PARALLEL ]]; do
  spawn_job $submitted
  ((submitted++))
done

# loop: read FIFO lines as jobs finish and start new ones
while [[ $successful -lt $TOTAL ]]; do
  if read -r line < "$FIFO"; then
    # line format: index:rc
    idx=${line%%:*}
    rc=${line##*:}
    ((completed++))
    if [[ "$rc" -ne 0 ]]; then
      echo "Job $idx: FAIL (code=$rc) -> ${OUTDIR}/job_${idx}.log"
      echo "Error details:"
      tail -10 "${OUTDIR}/job_${idx}.log" | sed 's/^/  /'
      echo ""
      ((failed++))
      if [[ $STOP_ON_FAILURE -eq 1 ]]; then
        echo "Stopping further submissions due to failure."
        # drain remaining FIFO messages for running jobs
        # but do not submit new jobs
        # wait for all running background jobs to finish
        wait
        break
      fi
    else
      echo "Job $idx: OK -> ${OUTDIR}/job_${idx}.log"
      ((successful++))
    fi

    # submit next job if we still need more successful ones
    if [[ $successful -lt $TOTAL ]]; then
      spawn_job $submitted
      ((submitted++))
    fi
  fi
done

echo "All done: submitted=$submitted successful=$successful failed=$failed (completed=$completed)"

exit 0