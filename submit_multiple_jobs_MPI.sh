#!/bin/bash

# Script to submit multiple run_CT_MPS_1-3_MPI.slurm jobs
# Usage: /scratch/ty296/CT_MPS_mini/submit_multiple_jobs_MPI.sh --L=20 --P_RANGE="0.5:1.0:20" --P_FIXED_NAME="p_ctrl" --P_FIXED_VALUE=0.0 --ANCILLA=0 --MAXDIM=512 --THRESHOLD=1e-15 --REALIZATIONS_PER_CPU=1 --N_JOBS=200 --MEM_PER_CPU=40G --N_TASKS=10

# SLURM script
SLURM_SCRIPT="/scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_MPI.slurm"

# Set default values (matching run_CT_MPS_1-3_MPI.slurm)
: ${MEM_PER_CPU:=10G}  # Default memory per cpu if not specified
: ${THRESHOLD:=1e-15}  # Default threshold if not specified
: ${REALIZATIONS_PER_CPU:=1}  # Default chunk realizations if not specified
: ${MAXDIM:=64}  # Default maxdim if not specified (will be recalculated in Julia based on L)


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --N_JOBS=*)
            N_JOBS="${1#*=}"
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

# Create output directory if it doesn't exist (only if not already set)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/scratch/ty296/hdf5_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi


# Submit N_JOBS number of jobs
for i in $(seq 1 $N_JOBS); do
    # Create a custom job name
    JOB_NAME="CT_MPS_L${L}_${P_FIXED_NAME}${P_FIXED_VALUE}_job${i}"
    
    sbatch --job-name="$JOB_NAME" --export=ALL,L=$L,P_RANGE="$P_RANGE",P_FIXED_NAME="$P_FIXED_NAME",P_FIXED_VALUE=$P_FIXED_VALUE,ANCILLA=$ANCILLA,MAXDIM=$MAXDIM,THRESHOLD=$THRESHOLD,OUTPUT_DIR="$OUTPUT_DIR",REALIZATIONS_PER_CPU=$REALIZATIONS_PER_CPU --ntasks=$N_TASKS --mem-per-cpu=$MEM_PER_CPU $SLURM_SCRIPT
done