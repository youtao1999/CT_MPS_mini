#!/bin/bash
#SBATCH --job-name=CT_MPS   # Job name with L parameter
<<<<<<< Updated upstream
#SBATCH --ntasks=4                # Number of tasks (processes)
=======
#SBATCH --ntasks=6                # Number of tasks (processes)
#SBATCH --partition=mem
>>>>>>> Stashed changes
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

# Run the command
# singularity exec /scratch/ty296/CT_MPS_mini/julia_CT.sif julia --sysimage /scratch/ty296/CT_MPS_mini/ct_with_wrapper.so /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L $L --p_range "$P_RANGE" --p_fixed_name "$P_FIXED_NAME" --p_fixed_value $P_FIXED_VALUE --ancilla $ANCILLA --maxdim $MAXDIM --threshold $THRESHOLD --n_chunk_realizations $N_CHUNK_REALIZATIONS --job_id $SLURM_JOB_ID --random --output_dir "$OUTPUT_DIR" --store_sv
julia --heap-size-hint=100G --sysimage=/scratch/ty296/CT_MPS_mini/ct_with_wrapper.so --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L $L --p_range "$P_RANGE" --p_fixed_name "$P_FIXED_NAME" --p_fixed_value $P_FIXED_VALUE --ancilla $ANCILLA --maxdim $MAXDIM --threshold $THRESHOLD --n_chunk_realizations $N_CHUNK_REALIZATIONS --job_counter $JOB_COUNTER --output_dir "$OUTPUT_DIR" --store_sv

# delete the --random flag when requiring a fixed seed

# srun --time=01:00:00 --mem=200G -p mem julia --heap-size-hint=100G --sysimage=/scratch/ty296/CT_MPS_mini/ct_with_wrapper.so --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L 20 --p_range "0.5" --p_fixed_name p_ctrl --p_fixed_value 0.4 --ancilla 0 --maxdim 512 --threshold 1e-15 --n_chunk_realizations 1 --job_counter 1 --store_sv

# srun --time=01:00:00 --mem=10G julia --heap-size-hint=10G --project=/scratch/ty296/CT_MPS_mini/CT /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L 20 --p_range "0.5" --p_fixed_name p_ctrl --p_fixed_value 0.4 --ancilla 0 --maxdim 512 --threshold 1e-15 --n_chunk_realizations 1 --job_counter 1 --store_sv