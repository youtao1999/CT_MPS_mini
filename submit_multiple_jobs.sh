#!/bin/bash

# Script to submit multiple run_CT_MPS_1-3.slurm jobs
# Usage: /scratch/ty296/CT_MPS_mini/submit_multiple_jobs.sh --L=20 --P_RANGE="0.0:1.0:20" --P_FIXED_NAME="p_ctrl" --P_FIXED_VALUE=0.4 --ANCILLA=0 --MAXDIM=300 --N_CHUNK_REALIZATIONS=10 --N_JOBS=200 --MEMORY=20G

# SLURM script
SLURM_SCRIPT="/scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.slurm"

# Set default values
: ${MEMORY:=4G}  # Default memory if not specified

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
        --N_CHUNK_REALIZATIONS=*)
            N_CHUNK_REALIZATIONS="${1#*=}"
            shift
            ;;
        --N_JOBS=*)
            N_JOBS="${1#*=}"
            shift
            ;;
        --MEMORY=*)
            MEMORY="${1#*=}"
            shift
            ;;
        --CUTOFF=*)
            CUTOFF="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
# OUTPUT_DIR="/scratch/ty296/json_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
OUTPUT_DIR="/scratch/ty296/hdf5_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi


# Submit N_JOBS number of jobs
for i in $(seq 1 $N_JOBS); do
    sbatch --export=ALL,L=$L,P_RANGE=$P_RANGE,P_FIXED_NAME=$P_FIXED_NAME,P_FIXED_VALUE=$P_FIXED_VALUE,ANCILLA=$ANCILLA,MAXDIM=$MAXDIM,CUTOFF=$CUTOFF,N_CHUNK_REALIZATIONS=$N_CHUNK_REALIZATIONS,OUTPUT_DIR=$OUTPUT_DIR --mem=$MEMORY $SLURM_SCRIPT
done