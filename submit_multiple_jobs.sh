#!/bin/bash

# Enhanced script to submit multiple jobs with seed range parsing
# Usage: ./submit_multiple_jobs.sh --SEED_RANGE="0:100:1" --L=20 --P_RANGE="0.6" --P_FIXED_NAME="p_ctrl" --P_FIXED_VALUE=0.0 --ANCILLA=0 --MAXDIM=512 --THRESHOLD=1e-15 --MEMORY=150G --OUTPUT_DIR="/scratch/ty296/hdf5_data/p_ctrl0.4/p_proj0.5"

# Source the seed parsing function
source /scratch/ty296/CT_MPS_mini/parse_seed_range.sh

# SLURM script
SLURM_SCRIPT="/scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.slurm"

# Set default values
: ${MEMORY:=4G}
: ${THRESHOLD:=1e-15}
: ${MAXDIM:=64}
: ${N_CHUNK_REALIZATIONS:=1}

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
        --OUTPUT_DIR=*)
            OUTPUT_DIR="${1#*=}"
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
# OUTPUT_DIR="/scratch/ty296/hdf5_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "SEED_INITIAL: $SEED_INITIAL"
echo "SEED_FINAL: $SEED_FINAL"
echo "SEED_STEP: $SEED_STEP"
echo "NUM_JOBS: $NUM_JOBS"
# echo "Submitting $NUM_JOBS jobs with seed ranges: $SEED_RANGE"
# # echo "N_CHUNK_REALIZATIONS: $SEED_STEP"

# Submit jobs based on parsed seed range
for ((job_id=SEED_INITIAL; job_id<SEED_FINAL; job_id+=SEED_STEP)); do
    # Create a custom job name
    JOB_NAME="L${L}_${P_FIXED_NAME}${P_FIXED_VALUE}_job${job_id}"
    # echo "Submitting job: $JOB_NAME"
    # Submit the job with seed parameters
    # echo "p_range: $P_RANGE"

    # srun --time=01:00:00 --mem=10G julia --sysimage=/scratch/ty296/CT_MPS_mini/ct_with_wrapper.so --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L $L --p_range "$P_RANGE" --p_fixed_name "$P_FIXED_NAME" --p_fixed_value $P_FIXED_VALUE --ancilla $ANCILLA --maxdim $MAXDIM --threshold $THRESHOLD --n_chunk_realizations $SEED_STEP --output_dir "$OUTPUT_DIR" --job_counter $job_id --store_sv

    sbatch --job-name="$JOB_NAME" \
           --export=ALL,L=$L,P_RANGE="$P_RANGE",P_FIXED_NAME="$P_FIXED_NAME",P_FIXED_VALUE=$P_FIXED_VALUE,ANCILLA=$ANCILLA,MAXDIM=$MAXDIM,THRESHOLD=$THRESHOLD,N_CHUNK_REALIZATIONS=$SEED_STEP,OUTPUT_DIR="$OUTPUT_DIR",JOB_NAME="$JOB_NAME",JOB_COUNTER=$job_id \
           --mem=$MEMORY \
           $SLURM_SCRIPT
done

echo "Submitted $NUM_JOBS jobs total."
