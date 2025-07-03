#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --p-fixed-name NAME    Parameter to keep fixed (p_ctrl or p_proj, default: p_ctrl)"
    echo "  --p-fixed-value VALUE  Fixed parameter value (default: 0.5)"
    echo "  --p-range RANGE        Range of parameter values (default: 0.0:1.0:10)"
    echo "  --L SIZE              System size (default: 16)"
    echo "  --ancilla VALUE       Ancilla setting (default: 0)"
    echo "  --n-chunk-realizations N  Number of realizations per chunk (default: 20)"
    echo "  --maxdim VALUE        Maximum bond dimension (default: 100)"
    echo "  --output-dir DIR      Output directory (default: based on L and maxdim)"
    echo "  --request-memory MEM  Request memory (default: 40G)"
    echo "  --request-time TIME   Request time (default: 1:00:00)"
    echo "  --direct              Run directly (skip srun - use if already on compute node)"
    echo "  -h, --help            Show this help message"
}

# Parse command line arguments
DIRECT_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --p-fixed-name)
            P_FIXED_NAME="$2"
            shift 2
            ;;
        --p-fixed-value)
            P_FIXED_VALUE="$2"
            shift 2
            ;;
        --p-range)
            P_RANGE="$2"
            shift 2
            ;;
        --L)
            L="$2"
            shift 2
            ;;
        --ancilla)
            ANCILLA="$2"
            shift 2
            ;;
        --n-chunk-realizations)
            N_CHUNK_REALIZATIONS="$2"
            shift 2
            ;;
        --maxdim)
            MAXDIM="$2"
            shift 2
            ;;
        --cutoff)
            CUTOFF="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --request-memory)
            REQUEST_MEMORY="$2"
            shift 2
            ;;
        --request-time)
            REQUEST_TIME="$2"
            shift 2
            ;;
        --direct)
            DIRECT_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default values if not provided via command line
: ${L:="10"}
: ${P_RANGE:="0.5"}
: ${P_FIXED_NAME:="p_ctrl"}
: ${P_FIXED_VALUE:="0.5"}
: ${ANCILLA:="0"}
: ${N_CHUNK_REALIZATIONS:="20"}
: ${MAXDIM:="9223372036854775807"}
: ${CUTOFF:="1e-10"}
: ${REQUEST_MEMORY:="40G"}
: ${REQUEST_TIME:="01:00:00"}

# Set output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    if [ "$MAXDIM" -eq "9223372036854775807" ]; then
        OUTPUT_DIR="/scratch/ty296/memory_benchmark_results/L${L}_maxdim_inf_cutoff${CUTOFF}_ancilla${ANCILLA}"
    else
        OUTPUT_DIR="/scratch/ty296/memory_benchmark_results/L${L}_maxdim${MAXDIM}_cutoff${CUTOFF}_ancilla${ANCILLA}"
    fi
fi

# Function to run the actual computation
run_computation() {
    # Convert HH:MM:SS to seconds (for timeout, though not used with /usr/bin/time approach)
    TIMEOUT_SECS=$(echo $REQUEST_TIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')

    # Step 0: Create output directories if they don't exist
    mkdir -p "$OUTPUT_DIR"
    # Create temporary directory for benchmark JSON output (separate from main computation data)
    BENCHMARK_JSON_DIR="$OUTPUT_DIR/benchmark_data"
    mkdir -p "$BENCHMARK_JSON_DIR"

    # Step 1: Load singularity module
    module load singularity

    # Step 2: Run the Julia program with /usr/bin/time to capture memory usage
    # Handle case where SLURM_JOB_ID is not defined (e.g., direct runs or non-SLURM environments)
    if [ -z "$SLURM_JOB_ID" ]; then
        JOB_ID="$(date +%Y%m%d_%H%M%S)_$$"  # Use timestamp and process ID with benchmark prefix
    else
        JOB_ID="$SLURM_JOB_ID"  # Add benchmark prefix to SLURM job ID
    fi
    
    # Ensure the main json_data directory exists (Julia program expects this)
    mkdir -p "/scratch/ty296/json_data/${P_FIXED_NAME}${P_FIXED_VALUE}"
    
    JULIA_CMD="singularity exec /scratch/ty296/CT_MPS_mini/julia_CT.sif julia --sysimage /scratch/ty296/CT_MPS_mini/ct_with_wrapper.so /scratch/ty296/CT_MPS_mini/run_CT_MPS_1-3.jl --L $L --p_range $P_RANGE --p_fixed_name $P_FIXED_NAME --p_fixed_value $P_FIXED_VALUE --ancilla $ANCILLA --maxdim $MAXDIM --cutoff $CUTOFF --n_chunk_realizations $N_CHUNK_REALIZATIONS --job_id $JOB_ID --random"

    echo "Running command: /usr/bin/time $JULIA_CMD"

    # Use /usr/bin/time to capture both output and memory usage
    # Format: %e = elapsed time, %M = maxresident memory in KB, %U = user time, %S = system time
    /usr/bin/time -f "MEMORY_STATS: elapsed_time=%e maxresident_kb=%M user_time=%U system_time=%S" \
        $JULIA_CMD 2>&1 | tee "$OUTPUT_DIR/julia_output.log"

    # Extract memory statistics from the output
    echo "Extracting memory statistics..."
    grep "MEMORY_STATS:" "$OUTPUT_DIR/julia_output.log" | tail -1 > "$OUTPUT_DIR/memory_stats.txt"

    # Parse and save individual metrics
    if [ -f "$OUTPUT_DIR/memory_stats.txt" ]; then
        MEMORY_LINE=$(cat "$OUTPUT_DIR/memory_stats.txt")
        MAX_MEMORY_KB=$(echo "$MEMORY_LINE" | sed 's/.*maxresident_kb=\([0-9]*\).*/\1/')
        ELAPSED_TIME=$(echo "$MEMORY_LINE" | sed 's/.*elapsed_time=\([0-9.]*\).*/\1/')
        USER_TIME=$(echo "$MEMORY_LINE" | sed 's/.*user_time=\([0-9.]*\).*/\1/')
        SYSTEM_TIME=$(echo "$MEMORY_LINE" | sed 's/.*system_time=\([0-9.]*\).*/\1/')
        
        # Convert memory to MB and GB for convenience
        MAX_MEMORY_MB=$(echo "scale=2; $MAX_MEMORY_KB / 1024" | bc)
        MAX_MEMORY_GB=$(echo "scale=3; $MAX_MEMORY_KB / 1048576" | bc)
        
        # Calculate total number of realizations (p_range points * n_chunk_realizations)
        # First, count the number of points in p_range
        if [[ "$P_RANGE" == *":"* ]]; then
            # Format: "start:stop:num"
            N_P_POINTS=$(echo "$P_RANGE" | cut -d':' -f3)
        else
            # Format: "0.1,0.2,0.3"
            N_P_POINTS=$(echo "$P_RANGE" | tr ',' '\n' | wc -l)
        fi
        TOTAL_REALIZATIONS=$((N_P_POINTS * N_CHUNK_REALIZATIONS))
        
        # Calculate average time per realization
        if [ "$TOTAL_REALIZATIONS" -gt 0 ]; then
            AVG_TIME_PER_REAL=$(echo "scale=2; $ELAPSED_TIME / $TOTAL_REALIZATIONS" | bc)
        else
            AVG_TIME_PER_REAL="N/A"
        fi
        
        # Check if JSON output file exists and move it to benchmark directory
        SOURCE_JSON_FILE="/scratch/ty296/json_data/${P_FIXED_NAME}${P_FIXED_VALUE}/${JOB_ID}_a${ANCILLA}_L${L}.json"
        BENCHMARK_JSON_FILE="$BENCHMARK_JSON_DIR/results.json"
        
        if [ -f "$SOURCE_JSON_FILE" ]; then
            JSON_LINE_COUNT=$(wc -l < "$SOURCE_JSON_FILE")
            # Move the JSON file to our benchmark output directory (separate from main computation data)
            mv "$SOURCE_JSON_FILE" "$BENCHMARK_JSON_FILE"
            JSON_INFO="Benchmark JSON results moved to: $BENCHMARK_JSON_FILE ($JSON_LINE_COUNT lines)"
            echo "JSON file moved from main computation directory to benchmark directory"
        else
            JSON_INFO="Warning: Expected JSON output file not found at $SOURCE_JSON_FILE"
        fi
        
        # Create summary file
        cat > "$OUTPUT_DIR/summary.txt" << EOF
=== Memory and Performance Summary ===
System Size (L): $L
Max Bond Dimension: $MAXDIM
Cutoff: $CUTOFF
Chunk Realizations: $N_CHUNK_REALIZATIONS
Parameter Points: $N_P_POINTS
Total Realizations: $TOTAL_REALIZATIONS
Fixed Parameter: $P_FIXED_NAME = $P_FIXED_VALUE
Parameter Range: $P_RANGE
Ancilla: $ANCILLA

=== Memory Usage ===
Max Resident Memory: $MAX_MEMORY_KB KB ($MAX_MEMORY_MB MB, $MAX_MEMORY_GB GB)

=== Timing ===
Total Elapsed Time: $ELAPSED_TIME seconds
User Time: $USER_TIME seconds
System Time: $SYSTEM_TIME seconds
Average Time per Realization: $AVG_TIME_PER_REAL seconds

=== Output ===
$JSON_INFO
Benchmark Data Directory: $BENCHMARK_JSON_DIR/
Shell Output Log: $OUTPUT_DIR/julia_output.log

=== Job Info ===
Job ID: $JOB_ID
Slurm Job ID: ${SLURM_JOB_ID:-"N/A (direct run)"}
Output Directory: $OUTPUT_DIR
Timestamp: $(date)
EOF

        echo "=== Summary ==="
        cat "$OUTPUT_DIR/summary.txt"
        
        # Also save just the key numbers for easy parsing
        echo "$MAX_MEMORY_KB" > "$OUTPUT_DIR/max_memory_kb.txt"
        echo "$AVG_TIME_PER_REAL" > "$OUTPUT_DIR/avg_time_per_realization.txt"
        echo "$TOTAL_REALIZATIONS" > "$OUTPUT_DIR/total_realizations.txt"
        
    else
        echo "Error: Could not find memory statistics in output"
        exit 1
    fi

    # Cleanup: Remove any temporary files that might have been created
    rm -f "/tmp/run_CT_MPS_benchmark_${JOB_ID}.jl" 2>/dev/null || true
    
    echo "Memory benchmarking completed. Results saved to $OUTPUT_DIR/"
    echo "Benchmark JSON data kept separate from main computation data in: $BENCHMARK_JSON_DIR/"
}

# Main execution logic
echo "=== Benchmarking Parameters ==="
echo "  P_FIXED_NAME: $P_FIXED_NAME"
echo "  P_FIXED_VALUE: $P_FIXED_VALUE"
echo "  P_RANGE: $P_RANGE"
echo "  L: $L"
echo "  ANCILLA: $ANCILLA"
echo "  N_CHUNK_REALIZATIONS: $N_CHUNK_REALIZATIONS"
echo "  MAXDIM: $MAXDIM"
echo "  CUTOFF: $CUTOFF"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  Note: Benchmark JSON data will be kept separate from main computation data"
echo ""

if [ "$DIRECT_RUN" = true ]; then
    echo "Running computation directly..."
    run_computation
else
    echo "Running job with srun for continuous output..."
    # Export variables so they're available in the srun environment
    export P_FIXED_NAME P_FIXED_VALUE P_RANGE L ANCILLA N_CHUNK_REALIZATIONS MAXDIM CUTOFF OUTPUT_DIR REQUEST_TIME
    
    # Run the job with srun, using bash -c to execute the function
    srun --nodes=1 --ntasks-per-node=1 --mem=$REQUEST_MEMORY --time=$REQUEST_TIME \
         --output="$OUTPUT_DIR/slurm_%j.out" --error="$OUTPUT_DIR/slurm_%j.err" \
         bash -c "$(declare -f run_computation); run_computation"
fi

echo "Session completed" 