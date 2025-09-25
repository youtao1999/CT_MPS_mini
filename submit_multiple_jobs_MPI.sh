#!/bin/bash

# Script to submit multiple run_CT_MPS_1-3_MPI.slurm jobs
# Usage: /scratch/ty296/CT_MPS_mini/submit_multiple_jobs_MPI.sh --SEED_RANGE="100:105:1" --L=8 --P_RANGE="0.5" --P_FIXED_NAME="p_ctrl" --P_FIXED_VALUE=0.0 --ANCILLA=0 --MAXDIM=512 --THRESHOLD=1e-15 --REALIZATIONS_PER_CPU=1 --MEM_PER_CPU=4G --OUTPUT_DIR="/scratch/ty296/hdf5_data/test_MPI"

# SLURM script
SLURM_SCRIPT="/scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_MPI.slurm"

source /scratch/ty296/CT_MPS_mini/parse_seed_range.sh

# Set default values (matching run_CT_MPS_1-3_MPI.slurm)
: ${MEM_PER_CPU:=10G}  # Default memory per cpu if not specified
: ${THRESHOLD:=1e-15}  # Default threshold if not specified
: ${REALIZATIONS_PER_CPU:=1}  # Default chunk realizations if not specified
: ${MAXDIM:=64}  # Default maxdim if not specified (will be recalculated in Julia based on L)


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
        --REALIZATIONS_PER_CPU=*)
            REALIZATIONS_PER_CPU="${1#*=}"
            shift
            ;;
        --MEM_PER_CPU=*)
            MEM_PER_CPU="${1#*=}"
            shift
            ;;
        --THRESHOLD=*)
            THRESHOLD="${1#*=}"
            shift
            ;;
        --N_TASKS=*)
            N_TASKS="${1#*=}"
            shift
            ;;
        --OUTPUT_DIR=*)
            OUTPUT_DIR="${1#*=}"
            shift
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

# Create output directory if it doesn't exist (only if not already set)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/scratch/ty296/hdf5_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi


# Submit number of jobs
for i in $(seq $SEED_INITIAL $SEED_STEP $SEED_FINAL); do
    # Create a custom job name
    JOB_NAME="L${L}_${P_FIXED_NAME}${P_FIXED_VALUE}_job${i}"
    echo "Submitting job: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --export=ALL,L=$L,P_RANGE="$P_RANGE",P_FIXED_NAME="$P_FIXED_NAME",P_FIXED_VALUE=$P_FIXED_VALUE,ANCILLA=$ANCILLA,MAXDIM=$MAXDIM,THRESHOLD=$THRESHOLD,OUTPUT_DIR="$OUTPUT_DIR",REALIZATIONS_PER_CPU=$REALIZATIONS_PER_CPU,JOB_COUNTER=$i,N_CHUNK_REALIZATIONS=$SEED_STEP --ntasks=$SEED_STEP --mem-per-cpu=$MEM_PER_CPU $SLURM_SCRIPT

    # set the N_TASKS to the number of seeds per job
done