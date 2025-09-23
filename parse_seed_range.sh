#!/bin/bash

# Simple function to parse "initial:final:step" and validate divisibility
# Usage: parse_seed_range "initial:final:step"
# Sets global variables: SEED_INITIAL, SEED_FINAL, SEED_STEP, NUM_JOBS
parse_seed_range() {
    local input="$1"
    
    # Parse input
    IFS=':' read -r SEED_INITIAL SEED_FINAL SEED_STEP <<< "$input"    
    # Calculate number of jobs
    local range=$((SEED_FINAL - SEED_INITIAL))
    NUM_JOBS=$((range / SEED_STEP))
    
    if [[ "$NUM_JOBS" -le 0 ]]; then
        echo "Error: Number of jobs must be positive. Got: $NUM_JOBS" >&2
        return 1
    fi
    
    echo "Will submit $NUM_JOBS jobs, each processing $SEED_STEP seeds (range: $SEED_INITIAL to $SEED_FINAL)"
    return 0
}
