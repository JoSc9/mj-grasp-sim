#!/usr/bin/env bash
# Generate 5 scenes for each object ID in train_obj.txt with parallelization across CPUs
# Usage: ./gen_scenes_train_objects.sh [NUM_CPUS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_OBJ_FILE="$PROJECT_ROOT/asset/mj-objects/obj_unsymmetric_test.txt"
NUM_CPUS="${1:-80}"

# Check if train_obj.txt exists
if [[ ! -f "$TRAIN_OBJ_FILE" ]]; then
    echo "Error: $TRAIN_OBJ_FILE not found"
    exit 1
fi

# Create a temporary directory for job coordination
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

QUEUE_FILE="$TEMP_DIR/queue.txt"
INDEX_FILE="$TEMP_DIR/index.txt"
LOCK_FILE="$TEMP_DIR/queue.lock"

# Populate queue with object IDs and initialize index
INDEX=0
while IFS= read -r object_id || [[ -n "$object_id" ]]; do
    # Skip empty lines or comments
    [[ -z "$object_id" ]] && continue
    [[ "$object_id" =~ ^[[:space:]]*# ]] && continue
    echo "$object_id" >> "$QUEUE_FILE"
    INDEX=$((INDEX + 1))
done < "$TRAIN_OBJ_FILE"

echo "0" > "$INDEX_FILE"
TOTAL_OBJECTS=$INDEX

# Function to process objects from the queue
worker_process() {
    local worker_id=$1
    
    while true; do
        # Lock and get next object ID from queue using line number indexing
        exec 200>"$LOCK_FILE"
        flock 200
        
        CURRENT_INDEX=$(cat "$INDEX_FILE")
        if [[ $CURRENT_INDEX -ge $TOTAL_OBJECTS ]]; then
            # Queue is exhausted
            flock -u 200
            break
        fi
        
        # Read object ID at current index (1-based for sed)
        object_id=$(sed -n "$((CURRENT_INDEX + 1))p" "$QUEUE_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Increment index for next worker
        echo "$((CURRENT_INDEX + 1))" > "$INDEX_FILE"
        
        flock -u 200
        exec 200>&-
        
        # Safety check: ensure object_id is not empty
        if [[ -z "$object_id" ]]; then
            echo "[Worker $worker_id] Error: Empty object_id at index $CURRENT_INDEX"
            continue
        fi
        
        # Process this object
        echo "[Worker $worker_id] Generating 5 scenes for object: $object_id"
        
        i=1
        while [[ $i -le 5 ]]; do
            echo "[Worker $worker_id] Scene $i/5 for $object_id"
            
            # Capture output and check for exceptions
            set +e
            output=$(cd "$PROJECT_ROOT" && python -m mgs.cli.gen_scene object.ids=$object_id 2>&1)
            exit_code=$?
            set -e
            
            # Debug: print output and exit code
            echo "[Worker $worker_id] Exit code: $exit_code"
            echo "[Worker $worker_id] Output:" 
            echo "$output" | head -5
            
            # Check if command failed (non-zero exit) or output contains error/exception messages
            if [[ $exit_code -ne 0 ]] || echo "$output" | grep -qiE "(exception|error|Not enough collision free grasps)"; then
                # Command failed, retry without incrementing
                echo "[Worker $worker_id] Failed, retrying scene $i/5 for $object_id"
            else
                # Command succeeded, move to next scene
                i=$((i + 1))
            fi
        done
        
        echo "[Worker $worker_id] Completed $object_id"
    done
}

# Start worker processes
echo "Starting $NUM_CPUS workers to process objects in parallel..."
for ((i = 1; i <= NUM_CPUS; i++)); do
    worker_process $i &
done

# Wait for all workers to finish
wait

echo "All scenes generated successfully!"
