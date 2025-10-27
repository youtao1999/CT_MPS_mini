#!/bin/bash
#SBATCH --requeue
#SBATCH --time=30:00:00            # Time limit hrs:min:sec
#SBATCH --output=/scratch/ty296/logs/%j.out            # Standard output log
#SBATCH --error=/scratch/ty296/logs/%j.err             # Standard error log
#SBATCH --partition=main
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

# # Set threading environment (following SLURM script pattern)
# export JULIA_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# export OMP_NUM_THREADS=2
# export OPENBLAS_NUM_THREADS=2

# Run the Julia MPI program with specified number of processes
# Using srun for SLURM native process management
# Temporarily removing sysimage to test MPI configuration
mpirun -n $SLURM_NTASKS julia --heap-size-hint=32G --project=/scratch/ty296/CT_MPS_mini/CT \
    /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_MPI.jl \
    --L $L \
    --job_file $JOB_FILE \
    --p_fixed_name $P_FIXED_NAME \
    --p_fixed_value $P_FIXED_VALUE \
    --ancilla $ANCILLA \
    --maxdim $MAXDIM \
    --threshold $THRESHOLD \

# # example:
# mpirun -n 4 julia --heap-size-hint=10G --project=/scratch/ty296/CT_MPS_mini/CT \
#     /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_MPI.jl \
#     --L 20 \
#     --job_list "[(2, 0.5, 1e-10, '/scratch/ty296/hdf5_data/p_ctrl0.4/p_proj0.5_eps1e-10'), (3, 0.5, 1e-10, '/scratch/ty296/hdf5_data/p_ctrl0.4/p_proj0.5_eps1e-10'), (4, 0.5, 1e-10, '/scratch/ty296/hdf5_data/p_ctrl0.4/p_proj0.5_eps1e-10'), (5, 0.5, 1e-10, '/scratch/ty296/hdf5_data/p_ctrl0.4/p_proj0.5_eps1e-10')]" \
#     --p_fixed_name p_ctrl \
#     --p_fixed_value 0.4 \
#     --ancilla 0 \
#     --maxdim 1024 \
#     --threshold 1e-15 \
#     --eps 0.0

# # First, load the necessary modules
# module purge
# module use /projects/community/modulefiles/
# module load julia/1.9-bd387
# module load gcc/14.2.0-cermak
# module load gcc/14.2.0/openmpi/5.0.5-cermak
# module load gcc/14.2.0/openmpi/5.0.5/hdf5/1.14.5-cermak

# # Then run the benchmark with nohup srun
# nohup srun --time=06:00:00 --mem=50G --cpus-per-task=4 -p mem julia --heap-size-hint=40G --sysimage=/scratch/ty296/CT_MPS_mini/ct_with_wrapper.so --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/benchmark_adder.jl > /scratch/ty296/logs/benchmark_adder_$(date +%Y%m%d_%H%M%S).log 2>&1 &