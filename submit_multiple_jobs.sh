#!/bin/bash

# Enhanced script to submit multiple jobs with seed range parsing
# Usage: ./submit_multiple_jobs.sh --SEED_RANGE="0:10:2" --L=8 --P_RANGE="0.5" --P_FIXED_NAME="p_ctrl" --P_FIXED_VALUE=0.0 --ANCILLA=0 --MAXDIM=512 --THRESHOLD=1e-15 --MEMORY=8G

# Source the seed parsing function
source /scratch/ty296/CT_MPS_mini/parse_seed_range.sh

# SLURM script
SLURM_SCRIPT="/scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.slurm"

# Set default values
: ${MEMORY:=4G}
: ${THRESHOLD:=1e-15}
: ${SEED_STEP:=1}
: ${MAXDIM:=64}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --SEED_RANGE=*)
            SEED_RANGE="${1#*=}"
            shift
            ;;
        --L=*)
            L="${1#*=}"
            shift
            ;;
        --P_RANGE=*)
            P_RANGE="${1#*=}"
            shift
            ;;
        --P_FIXED_NAME=*)
            P_FIXED_NAME="${1#*=}"
            shift
            ;;
        --P_FIXED_VALUE=*)
            P_FIXED_VALUE="${1#*=}"
            shift
            ;;
        --ANCILLA=*)
            ANCILLA="${1#*=}"
            shift
            ;;
        --MAXDIM=*)
            MAXDIM="${1#*=}"
            shift
            ;;
        --MEMORY=*)
            MEMORY="${1#*=}"
            shift
            ;;
        --THRESHOLD=*)
            THRESHOLD="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 --SEED_RANGE='initial:final:step' [other options]"
            echo "Example: $0 --SEED_RANGE='0:100:10' --L=20 --P_RANGE='0.5:1.0:20'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$SEED_RANGE" ]]; then
    echo "Error: --SEED_RANGE is required"
    echo "Example: --SEED_RANGE='0:100:10'"
    exit 1
fi

# Parse the seed range
if ! parse_seed_range "$SEED_RANGE"; then
    echo "Failed to parse seed range: $SEED_RANGE"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="/scratch/ty296/hdf5_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "Submitting $NUM_JOBS jobs with seed ranges: $SEED_RANGE"

# Submit jobs based on parsed seed range
for ((job_id=0; job_id<NUM_JOBS; job_id++)); do
    # Create a custom job name
    JOB_NAME="CT_MPS_L${L}_${P_FIXED_NAME}${P_FIXED_VALUE}_seeds${job_id}"
    
    # Submit the job with seed parameters
    sbatch --job-name="$JOB_NAME" \
           --export=ALL,L=$L,P_RANGE="$P_RANGE",P_FIXED_NAME="$P_FIXED_NAME",P_FIXED_VALUE=$P_FIXED_VALUE,ANCILLA=$ANCILLA,MAXDIM=$MAXDIM,THRESHOLD=$THRESHOLD,N_CHUNK_REALIZATIONS=$SEED_STEP,OUTPUT_DIR="$OUTPUT_DIR",JOB_NAME="$JOB_NAME",JOB_COUNTER=$job_id \
           --mem=$MEMORY \
           $SLURM_SCRIPT
done

echo "Submitted $NUM_JOBS jobs total."
