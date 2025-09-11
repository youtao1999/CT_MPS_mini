#!/bin/bash
#SBATCH --job-name=CT_MPS          # Job name
#SBATCH --ntasks=10                # Number of tasks (processes)
#SBATCH --requeue
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --mem-per-cpu=10G          # Memory per CPU core (4GB each)
#SBATCH --time=10:00:00            # Time limit hrs:min:sec
#SBATCH --output=/scratch/ty296/logs/%j.out            # Standard output log
#SBATCH --error=/scratch/ty296/logs/%j.err             # Standard error log
#SBATCH --partition=main
#SBATCH --nodes=1                  # Force single node

# Script to run CT_MPS simulation with MPI workers on a single node
# Usage: ./run_CT_MPS_1-3_MPI.slurm [additional_args...]
# Modeled after run_CT_MPS_1-3.slurm

# Default values if not provided via environment variables (following SLURM script pattern)
: ${L:=10}
: ${P_RANGE:="0.0:1.0:10"}  # Keep as string due to colon-separated format
: ${P_FIXED_NAME:="p_ctrl"}  # Keep as string - it's a name
: ${P_FIXED_VALUE:=0.5}
: ${ANCILLA:=0}
: ${MAXDIM:=512} # julia Typemax(Int)
: ${THRESHOLD:=1e-10}
: ${REALIZATIONS_PER_CPU:=1}
: ${N_CHUNK_REALIZATIONS:=$((REALIZATIONS_PER_CPU * SLURM_NTASKS))}
: ${OUTPUT_DIR:="/scratch/ty296/test_output"}
: ${JOB_ID:=$SLURM_JOB_ID}  # Use SLURM job ID if available, fallback to timestamp

# Module loading (following SLURM script pattern)
echo "Loading modules..."
module purge
module use /projects/community/modulefiles/
module load julia/1.9-bd387 # julia
module load gcc/14.2.0-cermak # GCC compiler
module load gcc/14.2.0/openmpi/5.0.5-cermak # Base OpenMPI module
module load gcc/14.2.0/openmpi/5.0.5/hdf5/1.14.5-cermak # HDF5 module compiled with GCC 14.2.0

# Set threading environment (following SLURM script pattern)
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Run the Julia MPI program with specified number of processes
# Using srun for SLURM native process management
# Temporarily removing sysimage to test MPI configuration
mpirun -n $SLURM_NTASKS julia --sysimage /scratch/ty296/CT_MPS_mini/ct_with_wrapper.so --project=/scratch/ty296/CT_MPS_mini/CT \
    /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_MPI.jl \
    --L $L \
    --p_range "$P_RANGE" \
    --p_fixed_name $P_FIXED_NAME \
    --p_fixed_value $P_FIXED_VALUE \
    --ancilla $ANCILLA \
    --maxdim $MAXDIM \
    --threshold $THRESHOLD \
    --n_chunk_realizations $N_CHUNK_REALIZATIONS \
    --job_id $JOB_ID \
    --random \
    --output_dir $OUTPUT_DIR \
    --store_sv
