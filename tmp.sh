#!/bin/bash

# Script to process jobs completed in the last 2 hours
# Get job IDs, extract JSON file paths from logs, and move files to precision_benchmark_results

# Set up directories
LOGS_DIR="/scratch/ty296/logs"
JSON_DATA_DIR="/scratch/ty296/json_data/p_proj0.5"
DEST_DIR="/scratch/ty296/precision_benchmark_results/p_proj0.5"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Getting job IDs of jobs completed in the last 2 hours..."

# Get job IDs completed in the last 2 hours (main job IDs only, not batch/extern)
RECENT_JOBS=$(sacct -S $(date -d '2 hours ago' +%Y-%m-%d-%H:%M:%S) -E now --format=JobID,JobName,State,End --parsable2 | grep "COMPLETED" | cut -d'|' -f1 | grep -v "batch\|extern")

if [ -z "$RECENT_JOBS" ]; then
    echo "No completed jobs found in the last 2 hours."
    exit 0
fi

echo "Found $(echo "$RECENT_JOBS" | wc -l) completed jobs in the last 2 hours."

# Array to store JSON file paths
declare -a json_files=()

# Process each job ID
for job_id in $RECENT_JOBS; do
    echo "Processing job ID: $job_id"
    
    # Check if the output log file exists
    log_file="$LOGS_DIR/${job_id}.out"
    
    if [ -f "$log_file" ]; then
        echo "  Found log file: $log_file"
        
        # Extract the JSON file path from the last line
        json_path=$(tail -n 1 "$log_file" | grep -o '/scratch/ty296/json_data/p_proj0.5/[^[:space:]]*\.json' | head -n 1)
        
        if [ -n "$json_path" ]; then
            echo "  Found JSON file path: $json_path"
            
            # Check if the JSON file exists
            if [ -f "$json_path" ]; then
                json_files+=("$json_path")
                echo "  JSON file exists and added to list"
            else
                echo "  Warning: JSON file does not exist: $json_path"
            fi
        else
            echo "  No JSON file path found in log file"
        fi
    else
        echo "  Warning: Log file not found: $log_file"
    fi
done

echo ""
echo "Summary:"
echo "Total jobs processed: $(echo "$RECENT_JOBS" | wc -l)"
echo "JSON files found: ${#json_files[@]}"

if [ ${#json_files[@]} -eq 0 ]; then
    echo "No JSON files to move."
    exit 0
fi

echo ""
echo "Moving JSON files to $DEST_DIR..."

# Move each JSON file to the destination directory
moved_count=0
for json_file in "${json_files[@]}"; do
    filename=$(basename "$json_file")
    dest_path="$DEST_DIR/$filename"
    
    if [ -f "$dest_path" ]; then
        echo "  Warning: Destination file already exists, skipping: $filename"
    else
        if mv "$json_file" "$dest_path"; then
            echo "  Moved: $filename"
            ((moved_count++))
        else
            echo "  Error: Failed to move $filename"
        fi
    fi
done

echo ""
echo "Operation completed!"
echo "Files moved: $moved_count"
echo "Files in destination directory: $(ls -1 "$DEST_DIR" | wc -l)"
