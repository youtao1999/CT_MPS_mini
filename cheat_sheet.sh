# Basic usage - scan p_ctrl from 0.0 to 1.0 with 20 points, keeping p_proj fixed at 0.5
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --L 20 --p-range "0.0:1.0:20" --p-fixed-name p_proj --p-fixed-value 0.5 --memory 40G

# Scan specific values of p_ctrl
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --L 20 --p-range "0.1,0.3,0.5,0.7,0.9" --p-fixed-name p_proj --p-fixed-value 0.5

# Large system with more memory and time
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --L 20 --maxdim 200 --request-memory 80G --request-time 02:00:00

# Run directly if already on compute node
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --direct --L 8 --n-chunk-realizations 5

/scratch/ty296/CT_MPS_mini/submit_multiple_jobs.sh --L=20 --P_RANGE="0.0:1.0:20" --P_FIXED_NAME="p_proj" --P_FIXED_VALUE=0.5 --ANCILLA=0 --MAXDIM=200 --N_CHUNK_REALIZATIONS=10 --N_JOBS=200 --MEMORY=40G
