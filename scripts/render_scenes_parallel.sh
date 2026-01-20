#!/usr/bin/env bash
# Run 16 jobs in parallel, binding each to a (different) CPU core.
set -euo pipefail

ids=(0 1 2 3 4 5 6 7 8 9 a b c d e f)
cores=$(nproc)

if [ "$cores" -lt "${#ids[@]}" ]; then
  echo "Warning: only $cores cores available; cores will be reused" >&2
fi

for i in "${!ids[@]}"; do
  id=${ids[i]}
  core=$(( i % cores ))
  echo "Starting input_id=$id on cpu $core"
  taskset -c "$core" python -m mgs.cli.render_scene_point_label input_id="$id" &
done

wait
echo "All jobs finished"