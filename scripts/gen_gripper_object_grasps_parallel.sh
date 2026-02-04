#!/usr/bin/env bash
# Run 16 jobs in parallel, binding each to a (different) CPU core.
set -euo pipefail

ids=(fast_eta_objects_part_00.txt fast_eta_objects_part_01.txt fast_eta_objects_part_02.txt fast_eta_objects_part_03.txt fast_eta_objects_part_04.txt fast_eta_objects_part_05.txt fast_eta_objects_part_06.txt fast_eta_objects_part_07.txt fast_eta_objects_part_08.txt fast_eta_objects_part_09.txt)
cores=$(nproc)

if [ "$cores" -lt "${#ids[@]}" ]; then
  echo "Warning: only $cores cores available; cores will be reused" >&2
fi

for i in "${!ids[@]}"; do
  id=${ids[i]}
  core=$(( i % cores ))
  echo "Starting input_id=$id on cpu $core"
  taskset -c "$core" python -m mgs.cli.gen_gripper_object_grasps obj_file_name="$id" &
done

wait
echo "All jobs finished"