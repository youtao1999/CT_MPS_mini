#!/bin/bash
#SBATCH --job-name=CT_MPS          # Job name
#SBATCH --ntasks=10                # Number of tasks (processes)
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G          # Memory per CPU core (4GB each)
#SBATCH --time=06:00:00            # Time limit hrs:min:sec
#SBATCH --output=/scratch/ty296/logs/%j.out            # Standard output log
#SBATCH --error=/scratch/ty296/logs/%j.err             # Standard error log
#SBATCH --partition=main
#SBATCH --nodes=1                  # Force single node
#SBATCH --ntasks-per-node=10       # All tasks on one node

# Script to run CT_MPS simulation with MPI workers on a single node
# Usage: ./run_CT_MPS_1-3_MPI.slurm [additional_args...]
# Modeled after run_CT_MPS_1-3.slurm

# Set number of workers (default 20, can be overridden)
NWORKERS=10

# Default values if not provided via environment variables (following SLURM script pattern)
: ${L:=10}
: ${P_RANGE:="0.0:1.0:10"}  # Keep as string due to colon-separated format
: ${P_FIXED_NAME:="p_ctrl"}  # Keep as string - it's a name
: ${P_FIXED_VALUE:=0.5}
: ${ANCILLA:=0}
: ${MAXDIM:=9223372036854775807} # julia Typemax(Int)
: ${CUTOFF:=1e-10}
: ${N_CHUNK_REALIZATIONS:=10}  # Default 20 realizations for 20 workers
: ${OUTPUT_DIR:="/scratch/ty296/hdf5_data"}
: ${JOB_ID:=$SLURM_JOB_ID}  # Use SLURM job ID if available, fallback to timestamp

# Module loading (following SLURM script pattern)
echo "Loading modules..."
module purge
module use /projects/community/modulefiles/
module load julia
# Use default Intel MPI (since it's the system default)
module load hdf5  # Add HDF5 module for system libraries

# Set threading environment (following SLURM script pattern)
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

echo "Checking MPI configuration..."
which mpiexec
echo "MPI Version:"
mpiexec --version || echo "No version info available"
echo "Library dependencies:"
ldd $(which mpiexec) | grep -E "(mpi|ompi)" || echo "No MPI libraries found"
echo "Environment modules loaded:"
module list
echo ""
echo "Running CT_MPS simulation with $NWORKERS workers on single node"
echo "Job ID: $JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "System size L: $L"
echo "P range: $P_RANGE"
echo "P fixed name: $P_FIXED_NAME"
echo "P fixed value: $P_FIXED_VALUE"
echo "Ancilla: $ANCILLA"
echo "Max bond dimension: $MAXDIM"
echo "Cutoff: $CUTOFF"
echo "Total realizations: $N_CHUNK_REALIZATIONS (distributed across workers)"
echo "Using srun for MPI execution"

# Run the Julia MPI program with specified number of processes
# Using srun for SLURM native process management
# Temporarily removing sysimage to test MPI configuration
mpirun -n $NWORKERS julia --project=/scratch/ty296/CT_MPS_mini/CT \
    run_CT_MPS_1-3_MPI.jl \
    --L $L \
    --p_range "$P_RANGE" \
    --p_fixed_name $P_FIXED_NAME \
    --p_fixed_value $P_FIXED_VALUE \
    --ancilla $ANCILLA \
    --maxdim $MAXDIM \
    --cutoff $CUTOFF \
    --n_chunk_realizations $N_CHUNK_REALIZATIONS \
    --job_id $JOB_ID \
    --random \
    --output_dir $OUTPUT_DIR \
    --store_sv

echo "Simulation completed. Results saved to:"
echo "HDF5 files: ${OUTPUT_DIR}/${P_FIXED_NAME}${P_FIXED_VALUE}_${JOB_ID}_worker*_a${ANCILLA}_L${L}.h5"

# Following SLURM script pattern - comment with example usage
echo ""
echo "Example usage:"
echo "  # Set custom parameters via environment variables"
echo "  L=12 P_RANGE=\"0.5:1.0:11\" N_CHUNK_REALIZATIONS=40 ./run_20_workers.sh"
echo ""
echo "  # Or use defaults (L=10, P_RANGE=\"0.0:1.0:10\", N_CHUNK_REALIZATIONS=20)"
echo "  ./run_CT_MPS_1-3_MPI.slurm"
