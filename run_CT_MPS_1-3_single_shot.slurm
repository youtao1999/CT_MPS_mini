#!/bin/bash
#SBATCH --job-name=CT_MPS   # Job name with L parameter
#SBATCH --ntasks=2                # Number of tasks (processes)
#SBATCH --requeue
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --time=20:00:00            # Time limit hrs:min:sec
#SBATCH --output=/scratch/ty296/logs/%j.out            # Standard output log
#SBATCH --error=/scratch/ty296/logs/%j.err             # Standard error log
#SBATCH --partition=mem


# Variables provided via --export (defaults are handled in Julia script if not provided)

# Load singularity
# module load singularity

module purge
module use /projects/community/modulefiles/
module load julia


# Restrict Julia threads to 1
export JULIA_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=2
# export OPENBLAS_NUM_THREADS=2

# Run the command
julia --heap-size-hint=100G --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_single_shot.jl --L $L --p_vary $P_VARY --p_fixed_name "$P_FIXED_NAME" --p_fixed_value $P_FIXED_VALUE --ancilla $ANCILLA --maxdim $MAXDIM --threshold $THRESHOLD --seed $SEED --output_dir "$OUTPUT_DIR" --eps $EPS  

# srun --nodes=1 --ntasks-per-node=1 --mem=100G --partition=mem --time=04:00:00 julia --heap-size-hint=40G --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3_single_shot.jl --L 40 --p_vary 0.5 --p_fixed_name "p_ctrl" --p_fixed_value 0.4 --ancilla 0 --maxdim 64 --threshold 1e-15 --seed 146 --output_dir "/scratch/ty296/test_output" --eps 1e-10

# L24 20G eps 1e-10
# L30 40G eps 1e-10, 1e-5