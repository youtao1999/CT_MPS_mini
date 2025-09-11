# println("initial memory: ", Sys.maxrss() / 1024^2, " MB")
using MPI
using Pkg
using ITensors
using Random
using LinearAlgebra
using MKL
using JSON
using HDF5
if !isdefined(Main, :CT)
    using CT
end
using Printf

using ArgParse
using Serialization

include("run_CT_MPS_1-3.jl")

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    println("Worker $rank of $size on node $(gethostname())")
    println("Uses threads: ",BLAS.get_num_threads())
    println("Uses backends: ",BLAS.get_config())
    
    args = parse_my_args()
    p_range = parse_p_range(args["p_range"])
    p_fixed_name = args["p_fixed_name"]
    p_fixed_value = args["p_fixed_value"]
    
    # Only master process prints the full configuration
    if rank == 0
        println("p_range: ", p_range)
        println("p_fixed_name: ", p_fixed_name)
        println("p_fixed_value: ", p_fixed_value)
        println("L: ", args["L"])
        println("ancilla: ", args["ancilla"])
        println("maxdim: ", args["maxdim"])
        println("threshold: ", args["threshold"])
        println("n_chunk_realizations: ", args["n_chunk_realizations"])
        println("Number of p values: ", length(p_range))
        println("Expected workers: ", min(size, length(p_range)))
    end

    # Distribute realizations across workers instead of p values
    # Each worker computes all p values but for different realizations
    total_realizations = args["n_chunk_realizations"]
    
    # Assert that number of realizations is divisible by number of workers
    if total_realizations % size != 0
        error("Number of realizations ($total_realizations) must be divisible by number of workers ($size)")
    end
    
    realizations_per_worker = div(total_realizations, size)
    start_realization = rank * realizations_per_worker + 1
    end_realization = (rank + 1) * realizations_per_worker
    
    println("Worker $rank assigned realizations: $start_realization to $end_realization ($realizations_per_worker total)")
    println("Worker $rank will compute all p values: ", p_range)
    println("Worker $rank job_id: $(args["job_id"])")
    println("Worker $rank output_dir: $(args["output_dir"])")
    
    # Choose storage format based on command line argument
    store_singular_values = args["store_sv"]

    if store_singular_values
        # Use HDF5 format for large singular value arrays
        # Each worker writes to its own file to avoid conflicts
        # Add hostname and process ID for extra uniqueness
        hostname = gethostname()
        pid = getpid()
        filename = "$(args["output_dir"])/$(args["p_fixed_name"])$(args["p_fixed_value"])_$(args["job_id"])_worker$(rank)_$(hostname)_$(pid)_L$(args["L"]).h5"
        println("Worker $rank will write to file: $filename")
        result_count = 0
        
        # Process assigned realizations, then sweep through all p values for each realization
        for i in start_realization:end_realization
            # Set seed once per realization to ensure consistency across all p values
            if args["random"]
                # Use realization number for seed to ensure reproducibility
                # All workers use the same seed for the same realization number
                seed = rand(1:10000)
            else
                seed = i  # Use realization number as seed
            end
            
            println("Worker $rank processing realization $i with seed $seed")
            
            # Now sweep through all p values with the same seed
            for p in p_range
                # Initialize parameters for this iteration
                p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p
                p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p
                
                # Get results as tuple with singular values
                @time O, entropy_data, max_bond = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["threshold"],seed;sv=store_singular_values)
                
                result_count += 1
                
                # Store result directly to HDF5
                store_result_hdf5(filename, result_count, O, entropy_data, max_bond, 
                                p_ctrl, p_proj, i, args, seed)
                
                println("Worker $rank stored result $result_count (realization=$i, p=$p) to HDF5")
            end
        end
        
        println("Worker $rank saved $result_count results to $filename (HDF5 format)")
        
    else
        # Use JSON format for scalar entropy values only
        # Each worker writes to its own file to avoid conflicts
        filename = "$(args["output_dir"])/$(args["job_id"])_worker$(rank)_a$(args["ancilla"])_L$(args["L"]).json"
        result_count = 0
        
        open(filename, "w") do f
            # Process assigned realizations, then sweep through all p values for each realization
            for i in start_realization:end_realization
                # Set seed once per realization to ensure consistency across all p values
                if args["random"]
                    # Use realization number for seed to ensure reproducibility
                    # All workers use the same seed for the same realization number
                    seed = rand(1:10000)
                else
                    seed = i  # Use realization number as seed
                end
                
                println("Worker $rank processing realization $i with seed $seed")
                
                # Now sweep through all p values with the same seed
                for p in p_range
                    # Initialize parameters for this iteration
                    p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p
                    p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p
                    
                    # Get results as dictionary (scalar entropy only)
                    results = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["threshold"],seed;sv=store_singular_values)
                    data_to_serialize = merge(results, Dict("args" => args, "p_value" => p, "realization number" => i, "worker_rank" => rank))
                    
                    # Write each result as a separate line (JSON Lines format)
                    println(f, JSON.json(data_to_serialize))
                    result_count += 1
                    
                    println("Worker $rank stored result for realization=$i, p=$p")
                end
            end
        end
        
        println("Worker $rank saved $result_count results to $filename (JSON format)")
    end
    
    # Synchronize all workers before finishing
    MPI.Barrier(comm)
    
    if rank == 0
        println("All workers completed their tasks")
    end
    
    # Explicitly finalize MPI (following working script pattern)
    MPI.Finalize()
end

main()