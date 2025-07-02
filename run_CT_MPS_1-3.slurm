#!/bin/bash
#SBATCH --job-name=CT_MPS          # Job name
#SBATCH --ntasks=6                # Number of tasks (processes)
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --time=06:00:00            # Time limit hrs:min:sec
#SBATCH --output=/scratch/ty296/logs/%j.out            # Standard output log
#SBATCH --error=/scratch/ty296/logs/%j.err             # Standard error log
#SBATCH --partition=main


# Default values if not provided via --export
: ${L:=10}
: ${P_RANGE:="0.0:1.0:10"}  # Keep as string due to colon-separated format
: ${P_FIXED_NAME:="p_ctrl"}  # Keep as string - it's a name
: ${P_FIXED_VALUE:=0.5}
: ${ANCILLA:=0}
: ${MAXDIM:=10}
: ${N_CHUNK_REALIZATIONS:=1}

# Load singularity
module load singularity

# Run the command
singularity exec /scratch/ty296/CT_MPS_mini/julia_CT.sif julia --sysimage /scratch/ty296/CT_MPS_mini/ct_with_wrapper.so /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L $L --p_range $P_RANGE --p_fixed_name $P_FIXED_NAME --p_fixed_value $P_FIXED_VALUE --ancilla $ANCILLA --maxdim $MAXDIM --n_chunk_realizations $N_CHUNK_REALIZATIONS --job_id $SLURM_JOB_ID --random

# delete the --random flag when requiring a fixed seed