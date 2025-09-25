#!/bin/bash
#SBATCH --requeue
#SBATCH --cpus-per-task=2        # Number of CPU cores per task
#SBATCH --time=10:00:00            # Time limit hrs:min:sec
#SBATCH --output=/scratch/ty296/logs/%j.out            # Standard output log
#SBATCH --error=/scratch/ty296/logs/%j.err             # Standard error log
#SBATCH --partition=mem
#SBATCH --nodes=1                  # Force single node

# Script to run CT_MPS simulation with MPI workers on a single node
# Usage: ./run_CT_MPS_1-3_MPI.slurm [additional_args...]
# Modeled after run_CT_MPS_1-3.slurm

# Module loading (following SLURM script pattern)
echo "Loading modules..."
module purge
module use /projects/community/modulefiles/
module load julia/1.9-bd387 # julia
module load gcc/14.2.0-cermak # GCC compiler
module load gcc/14.2.0/openmpi/5.0.5-cermak # Base OpenMPI module
module load gcc/14.2.0/openmpi/5.0.5/hdf5/1.14.5-cermak # HDF5 module compiled with GCC 14.2.0

# Set threading environment (following SLURM script pattern)
export JULIA_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Debug: Print all variables
echo "=== DEBUG: Variable values ==="
echo "L=$L"
echo "P_RANGE=$P_RANGE"
echo "P_FIXED_NAME=$P_FIXED_NAME"
echo "P_FIXED_VALUE=$P_FIXED_VALUE"
echo "ANCILLA=$ANCILLA"
echo "MAXDIM=$MAXDIM"
echo "N_CHUNK_REALIZATIONS=$N_CHUNK_REALIZATIONS"
echo "THRESHOLD=$THRESHOLD"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "JOB_NAME=$JOB_NAME"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "JOB_COUNTER=$JOB_COUNTER"
echo "=== END DEBUG ==="

# Run the Julia MPI program with specified number of processes
# Using srun for SLURM native process management
# Temporarily removing sysimage to test MPI configuration
mpirun -n $SLURM_NTASKS julia --heap-size-hint=100G --sysimage /scratch/ty296/CT_MPS_mini/ct_with_wrapper.so --project=/scratch/ty296/CT_MPS_mini/CT \
    /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_MPI.jl \
    --L $L \
    --p_range "$P_RANGE" \
    --p_fixed_name $P_FIXED_NAME \
    --p_fixed_value $P_FIXED_VALUE \
    --ancilla $ANCILLA \
    --maxdim $MAXDIM \
    --threshold $THRESHOLD \
    --n_chunk_realizations $N_CHUNK_REALIZATIONS \
    --job_counter $JOB_COUNTER \
    --output_dir $OUTPUT_DIR \
    --store_sv
