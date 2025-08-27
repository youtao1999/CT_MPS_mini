#!/bin/bash

JOB_IDS_FILE="${1:-sacct.txt}"
LOGS_DIR="/scratch/ty296/logs"

while read -r jobid; do
    [[ -z "$jobid" ]] && continue
    
    out_file="$LOGS_DIR/${jobid}.out"
    
    echo "Job ID: $jobid"
    if [[ -f "$out_file" ]]; then
        sed -n '2p' "$out_file"
        sed -n '9p' "$out_file"
    fi
    echo
    
    seff $jobid | awk '/Memory:/ {print $3, $4}'
done < "$JOB_IDS_FILE"
