#!/bin/bash

###########################################
# SLURM Command Cheat Sheet
# Created: $(date)
# Description: Common SLURM commands for job management and monitoring
###########################################

echo "This is a reference file. Do not execute directly."
exit 0

###########################################
# Basic Job History and Monitoring
###########################################

# View your recent jobs (basic)
sacct

# View jobs with detailed format
sacct --format=JobID,JobName,State,Elapsed,Start,End,MaxRSS,NodeList > sacct_output.txt

# View currently running and pending jobs
squeue -u $USER

# Watch jobs in real-time (refreshes every 1 seconds)
watch -n 1 squeue -u ty296

###########################################
# Time-Based Job History
###########################################

# Last 24 hours
sacct --starttime $(date -d '24 hours ago' +%Y-%m-%d-%R)

# Specific date range
sacct --starttime 2025-07-03 --endtime 2025-07-07

###########################################
# Memory and Resource Usage
###########################################

# View detailed memory usage
sacct --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,State,Elapsed,ExitCode --user=ty296

# Get efficiency report for specific job
seff <jobid>

# View memory usage for multiple jobs
sacct -j $(tr '\n' ',' < job_ids.txt) --format=JobID,JobName%30,State,MaxRSS,MaxVMSize,Elapsed,ExitCode

###########################################
# Job Details and Configuration
###########################################

# Detailed job information
scontrol show job <jobid>

# Node resource information
scontrol show nodes hal[0079,0087]

# View jobs on specific nodes
squeue -w hal[0079,0087] -o "%.18i %.9P %.8j %.8u %.8T %.10M %.6D %R %C"

# View just CPU load
scontrol show nodes hal[0079,0087] | grep -E "NodeName|CPULoad"

###########################################
# Extracting Job IDs
###########################################

# Get main job IDs (exclude .ba+ and .ex+)
sacct | grep -v "\.ba\|\.ex" | awk '{print $1}'

# Get failed job IDs
sacct | grep -v "\.ba\|\.ex" | awk '{print $1}'

# Get out-of-memory job IDs
sacct | grep -v "\.ba\|\.ex" | awk '$6 == "OUT_OF_ME+" {print $1}'

###########################################
# Interactive Sessions
###########################################

# Request interactive session
srun --nodes=1 --tasks-per-node=6 --cpus-per-task=1 --mem-per-cpu=16G --time=04:00:00 --pty bash

# Request specific node
srun --nodelist=hal0079 --tasks-per-node=1 --cpus-per-task=1 --mem-per-cpu=4G --time=01:00:00 --pty bash

###########################################
# Output Management
###########################################

# Save job IDs to file
sacct | awk '!/\.ba|\.ex/ {print $1}' > job_ids.txt

# Append new job IDs with timestamp
echo -e "\nJob IDs from $(date):" >> job_ids.txt
sacct | awk '!/\.ba|\.ex/ {print $1}' >> job_ids.txt

###########################################
# Useful Aliases (Add to ~/.bashrc)
###########################################

# Quick job status
alias myjobs="sacct --format=JobID,JobName%30,State,Elapsed,Start,End,MaxRSS,NodeList"
alias watchjobs="watch -n 0.1 squeue -u $USER"

###########################################
# Common Exit Codes
###########################################
# 0:0    - Successful completion
# 0:125  - Out of memory (OOM)
# 11:0   - General failure
# 7:0    - System/runtime error

###########################################
# Understanding Job States
###########################################
# PENDING     - Job is waiting for resources
# RUNNING     - Job is currently running
# COMPLETED   - Job completed successfully
# FAILED      - Job terminated with non-zero exit code
# TIMEOUT     - Job reached its time limit
# OUT_OF_ME+  - Job terminated due to memory limit
# CANCELLED   - Job was cancelled by user or system

###########################################
# Job Components
###########################################
# Main Job (no suffix) - Overall job status
# .ba+ (batch)        - Actual batch script execution
# .ex+ (extern)       - External SLURM services/monitoring

###########################################
# Memory Units in SLURM Output
###########################################
# K - Kilobytes  (1024 bytes)
# M - Megabytes  (1024K)
# G - Gigabytes  (1024M)
# T - Terabytes  (1024G)

###########################################
# Tips
###########################################
# 1. Always check MaxRSS when jobs fail - it shows actual memory usage
# 2. Use --mem-per-cpu or --mem to set memory limits
# 3. Use seff for quick efficiency analysis
# 4. Watch for OUT_OF_ME+ status - indicates need for more memory
# 5. Check both .out and .err files for error messages

###########################################
# Job Queue and Placement
###########################################

# View your jobs in the queue
squeue -u $USER

# View all jobs with estimated start times
squeue --start

# View detailed queue information with priorities
squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %Q %p"

# View your job's priority and position
sprio -j <jobid>

# View all partitions and their limits
sinfo

# View node status and job placement
sinfo -N -l

# View detailed partition information
scontrol show partition

# View jobs running on specific nodes with resources
squeue -w hal[0079,0087] -o "%.18i %.9P %.8j %.8u %.8T %.10M %.6D %R %C"

# View your pending jobs with reasons
squeue -u $USER -t PENDING -o "%.18i %.9P %.8j %.8u %.8T %.10M %.6D %R %r"

###########################################
# Memory Request and Usage Information
###########################################

# View memory requests and usage for your jobs
squeue -u ty296 -o "%.18i %.9P %.8j %.6C %.8m %.12l %.12L"
# %i = JobID, %P = Partition, %j = JobName, %C = CPUs, %m = Memory, %l = TimeLimit, %L = TimeLeft

# View detailed memory allocation for running jobs
scontrol show job -d $(squeue -u $USER -h -t RUNNING -o %A)

# View memory requests and actual usage for completed jobs
sacct -u ty296 --format=JobID,JobName%30,Partition,AllocCPUS,ReqMem,MaxRSS,Elapsed,State

# View memory efficiency (requested vs used) for a specific job
seff <jobid>

# View memory and CPU requests with node placement
squeue -u $USER -o "%.18i %.9P %.8j %.6C %.8m %.6D %R"

# Detailed memory usage history
sacct -u ty296 --format=JobID,JobName%30,State,AllocCPUS,ReqMem,MaxRSS,AveRSS,MaxVMSize

###########################################
# Job Priority and Queue Position
###########################################

# View priority of all your jobs
sprio -u $USER

# View detailed priority factors for a specific job
sprio -j <jobid> -l

# View all jobs in queue sorted by priority (highest first)
squeue --sort=-p -o "%.18i %.9P %.8j %.8u %.8T %.5Q %.6p %R"
# %Q = Priority, %p = Nice value, %R = Reason for pending

# View your position in queue with priorities
squeue -u $USER -o "%.18i %.9P %.8j %.8T %.5Q %.6p %R"

# View all pending jobs sorted by priority
squeue -t PENDING --sort=-p -o "%.18i %.9P %.8j %.8u %.8T %.5Q %.6p %r"

# View detailed job priority info including age and fairshare
sprio -n -o "%.15i %9r %10y %10f %10a %10j %10p %10q %10N"
# %r=RawPrio, %y=Long/Nice, %f=Fairshare, %a=Age, %j=JobSize
# %p=Partition, %q=QOS, %N=NodeWeight

###########################################
# Queue Position and Overall Queue Status
###########################################

# View entire queue sorted by priority with usernames
squeue --sort=-p -o "%.18i %.9P %.8j %.8u %.8T %.5Q %.6p %.11M %R"

# Count number of jobs ahead of you in queue
squeue -t PENDING --sort=-p -o "%.8u %.5Q" | awk -v user=$USER '
    {if ($1 == user) {exit} else {count++}}
    END {print "Jobs ahead of you: "count}'

# View jobs per user in the queue
squeue --format="%.18i %.9P %.8j %.8u %.8T %.5Q" | awk '
    NR>1 {users[$4]++}
    END {for (user in users) printf "%-20s %d\n", user, users[user]}' | sort -k2nr

# Summary of queue by partition
squeue --format="%.9P %.8T" | awk '
    NR>1 {parts[$1]++; states[$1,$2]++}
    END {
        printf "\n%-12s %-10s %-10s %-10s %-10s\n", "PARTITION", "TOTAL", "PENDING", "RUNNING", "OTHER"
        for (p in parts) {
            printf "%-12s %-10d %-10d %-10d %-10d\n", 
            p, parts[p], states[p,"PENDING"], states[p,"RUNNING"], 
            parts[p]-states[p,"PENDING"]-states[p,"RUNNING"]
        }
    }' | sort

# View detailed queue with estimated start times
squeue --start --format="%.18i %.9P %.8j %.8u %.8T %.11M %.11S %.12e"

###########################################
# Interactive Session Examples
###########################################

# Basic interactive session
srun --nodes=1 --ntasks=1 --mem=40G --time=04:00:00 --pty bash

# Request specific node for interactive session
srun --nodelist=slepner088 --ntasks=1 --mem=1G --time=01:00:00 --pty bash

# Interactive session workflow
# 1. Request resources
# 2. Load necessary modules (e.g., module load singularity)
# 3. Run your test commands
singularity exec /scratch/ty296/julia_itensor.sif julia --sysimage /scratch/ty296/run_CT_MPS_ensemble.so --check-bounds=no -O3 /scratch/ty296/CT_MPS/run_CT_MPS_ensemble.jl --scan_type p_ctrl --p_fixed 0.5 --p_range "0.1,0.2,0.3,0.4,0.5" --L 18 --ancilla 0 --n_realizations 100 --maxdim 500
# 4. Exit when done (exit or Ctrl+D)

###########################################
# Job Monitoring Tips
###########################################

# Watch job queue in real-time (refreshes every 0.1 seconds)
watch -n 1 squeue -u ty296

# Alternative with full command
alias watchjobs="watch -n 0.1 squeue -u $USER"

# Performance Benchmarks
# Example: For MPS calculations with maxdim=100
# - L=8:  ~1.3 secs/realization
# - L=10: ~2.3 secs/realization
# - L=12: ~3.9 secs/realization
# - L=14: ~6.4 secs/realization
# - L=16: ~9.3 secs/realization
# - L=18: ~12.8 secs/realization 

###########################################
# Process Monitoring and Memory Tracking
###########################################

# Monitor residential memory of specific PID every second
# Format: timestamp, RSS (KB) 
nohup bash -c 'while true; do ps -o rss= -p 5376 | awk -v ts=$(date +%s) '{print ts, $1}' >> memory_log.txt; sleep 1; done' &

# Same command but with human readable timestamp
nohup bash -c 'while true; do ps -o rss= -p 20564 | awk -v ts="$(date '+%Y-%m-%d %H:%M:%S')" '{print ts, $1}' >> memory_log_readable.txt; sleep 1; done' &

# Monitor with more details (timestamp, RSS, VSZ, %MEM)
nohup bash -c 'while true; do ps -o rss=,vsz=,%mem= -p 20564 | awk -v ts="$(date '+%Y-%m-%d %H:%M:%S')" '{print ts, $1, $2, $3}' >> memory_log_detailed.txt; sleep 1; done' &

# Run in background with timestamp in filename
nohup bash -c 'while true; do ps -o rss= -p 20564 | awk -v ts=$(date +%s) "{print ts, \$1}" >> memory_log_$(date +%Y%m%d_%H%M%S).txt; sleep 1; done' &


# sometimes that ssh agent would stop working, here is the fix:

# Check if agent is running
ps aux | grep ssh-agent | grep $(whoami)

# Check environment variables
echo $SSH_AUTH_SOCK
echo $SSH_AGENT_PID

# Check loaded keys
ssh-add -l

# Start agent if needed
eval "$(ssh-agent -s)"

# Add keys if needed
ssh-add ~/.ssh/id_rsa ~/.ssh/github_ed25519

# Basic usage - scan p_ctrl from 0.0 to 1.0 with 20 points, keeping p_proj fixed at 0.5
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --L 20 --p-range "0.0:1.0:20" --p-fixed-name p_proj --p-fixed-value 0.5 --memory 40G

# Scan specific values of p_ctrl
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --L 20 --p-range "0.1,0.3,0.5,0.7,0.9" --p-fixed-name p_proj --p-fixed-value 0.5

# Large system with more memory and time
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --L 20 --maxdim 200 --request-memory 80G --request-time 02:00:00

# Run directly if already on compute node
/scratch/ty296/CT_MPS_mini/mini_memory_benchmark.sh --direct --L 8 --n-chunk-realizations 5

/scratch/ty296/CT_MPS_mini/submit_multiple_jobs.sh --L=24 --P_RANGE="0.0:1.0:20" --P_FIXED_NAME="p_ctrl" --P_FIXED_VALUE=0.0 --ANCILLA=0 --N_CHUNK_REALIZATIONS=1 --N_JOBS=200 --MEMORY=40G
