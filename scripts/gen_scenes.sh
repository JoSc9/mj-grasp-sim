#!/usr/bin/env bash
# Parallel scene runner
# Usage:
#  ./gen_scenes.sh -t TOTAL [-p PARALLEL] [-o OUTDIR] [-c CMD] [--stop-on-failure]
#
# Default CMD: "python -m mgs.cli.gen_scene" 

set -u

TOTAL=4
PARALLEL=$(nproc)
OUTDIR=parallel_runs
CMD="python -m mgs.cli.gen_scene"
STOP_ON_FAILURE=0

usage() {
  cat <<EOF
Usage: $0 -t TOTAL [-p PARALLEL] [-o OUTDIR] [-c CMD] [--stop-on-failure]

Run TOTAL scene jobs in parallel with up to PARALLEL concurrent jobs.
CMD is a command template that must contain the token which will be
replaced by the job index (0..TOTAL-1). Default:
  python -m mgs.cli.gen_scene 

Each job's stdout+stderr is written to OUTDIR/job_<i>.log
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--total)
      TOTAL="$2"; shift 2;;
    -p|--parallel)
      PARALLEL="$2"; shift 2;;
    -o|--outdir)
      OUTDIR="$2"; shift 2;;
    -c|--cmd)
      CMD="$2"; shift 2;;
    --stop-on-failure)
      STOP_ON_FAILURE=1; shift;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1" >&2; usage;;
  esac
done

mkdir -p "$OUTDIR"

FIFO="/tmp/gen_scenes_fifo_$$"
rm -f "$FIFO"
mkfifo "$FIFO"

cleanup() {
  rm -f "$FIFO"
}
trap cleanup EXIT

spawn_job() {
  local i=$1
  local cmd=${CMD//\{i\}/$i}
  local log="${OUTDIR}/job_${i}.log"
  # run in background; when finished write "i:rc" to FIFO
  (
    echo "[JOB $i] START: $cmd" > "$log"
    eval "$cmd" >> "$log" 2>&1
    rc=$?
    echo "$i:$rc" > "$FIFO"
  ) &
}

submitted=0
completed=0
failed=0

# start initial batch
while [[ $submitted -lt $TOTAL && $submitted -lt $PARALLEL ]]; do
  spawn_job $submitted
  ((submitted++))
done

# loop: read FIFO lines as jobs finish and start new ones
while [[ $completed -lt $TOTAL ]]; do
  if read -r line < "$FIFO"; then
    # line format: index:rc
    idx=${line%%:*}
    rc=${line##*:}
    ((completed++))
    if [[ "$rc" -ne 0 ]]; then
      echo "Job $idx: FAIL (code=$rc) -> ${OUTDIR}/job_${idx}.log"
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
    fi

    # submit next job if any remaining
    if [[ $submitted -lt $TOTAL ]]; then
      spawn_job $submitted
      ((submitted++))
    fi
  fi
done

echo "All done: submitted=$submitted completed=$completed failed=$failed"

exit 0
python -m mgs.cli.gen_scene ()