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

function main_interactive(L::Int,p_ctrl::Float64,p_proj::Float64,ancilla::Int,maxdim::Int,cutoff::Float64,seed::Int;sv::Bool=false,n::Int=0)
    ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=maxdim,_cutoff=cutoff, _maxdim0=maxdim)
    initial_state = copy(ct_f.mps)
    initial_maxdim = CT.max_bond_dim(ct_f.mps)
    println("initial_maxdim: ",initial_maxdim)
    i=1
    # T_max = 100
    T_max = ancilla ==0 ? 2*(ct_f.L^2) : div(ct_f.L^2,2)
    # println("memory after CT_MPS: ", Sys.maxrss() / 1024^2, " MB")
    for idx in 1:T_max
        # println(idx)
        # before = Sys.maxrss()
        i=CT.random_control!(ct_f,i,p_ctrl,p_proj)
        after = Sys.maxrss()
        # println(Base.summarysize(ct_f.adder))
        # println(Base.summarysize(ct_f.mps))
        println(idx, " maxrss: ", after / 1024^2, " MB")
        # println(varinfo())
    end
    O=CT.order_parameter(ct_f)
    max_bond= CT.max_bond_dim(ct_f.mps)
    if ancilla ==0 
        if sv
            sv_arr=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2);sv=sv)
            return O, sv_arr, max_bond
        else
            EE=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2);n=n)
            ct_f.mps=initial_state # resetting the mps for memory benchmarking purposes
            return Dict("O" => O, "EE" => EE, "max_bond" => max_bond, "p_ctrl" => p_ctrl, "p_proj" => p_proj, "n" => n)
        end
    else
        if sv
            SA=CT.von_Neumann_entropy(ct_f.mps,1;sv=sv)
            return O, SA, max_bond
        else
            SA=CT.von_Neumann_entropy(ct_f.mps,1;n=n)
            return Dict("O" => O, "SA" => SA, "max_bond" => max_bond, "p_ctrl" => p_ctrl, "p_proj" => p_proj, "n" => n)
        end
    end
end

"""
Parse the p_range argument into a list of values
"""
function parse_p_range(p_range_str::String)
    if contains(p_range_str, ":")
        # Format: "start:stop:num"
        parts = split(p_range_str, ":")
        if length(parts) != 3
            error("Invalid range format. Expected start:stop:num")
        end
        start = parse(Float64, parts[1])
        stop = parse(Float64, parts[2])
        num = parse(Int, parts[3])
        return range(start, stop, length=num) |> collect
    else
        # Format: "0.1,0.2,0.3"
        return parse.(Float64, split(p_range_str, ","))
    end
end

"""
Store a single result directly to HDF5 file, optimized for large singular value arrays.
Uses compression and efficient chunking for large arrays.
Appends results to existing file or creates new file.
"""
function store_result_hdf5(filename::String, result_idx::Int, O::Float64, entropy_data, max_bond::Int, 
                          p_ctrl::Float64, p_proj::Float64, p_value::Float64, realization::Int, 
                          args::Dict, ancilla::Int)
    
    # Each worker writes multiple results to their own unique file
    # First call creates file and groups, subsequent calls append to same file
    file_mode = isfile(filename) ? "r+" : "cw"
    h5open(filename, file_mode) do file
        # Initialize file structure if new file
        if file_mode == "cw"
            metadata_group = create_group(file, "metadata")
            sv_arrays_group = create_group(file, "singular_values")
        else
            metadata_group = file["metadata"]
            sv_arrays_group = file["singular_values"]
        end
        
        result_name = "result_$(result_idx)"
        
        # Store metadata
        meta_group = create_group(metadata_group, result_name)
        meta_group["O"] = O
        meta_group["max_bond"] = max_bond
        meta_group["p_ctrl"] = p_ctrl
        meta_group["p_proj"] = p_proj
        meta_group["p_value"] = p_value
        meta_group["realization"] = realization
        meta_group["ancilla"] = ancilla
        
        # Store args
        args_group = create_group(meta_group, "args")
        for (key, value) in args
            try
                args_group[key] = value
            catch
                args_group[key] = string(value)
            end
        end
        
        # Store singular values with compression (always present in HDF5 files)
        sv_data = Float64.(entropy_data)
        
        sv_dataset = create_dataset(sv_arrays_group, result_name, 
        datatype(Float64), dataspace(sv_data),
        chunk=(min(1000, length(sv_data)),), 
        shuffle=true, deflate=6)
        write(sv_dataset, sv_data)        
        # No global metadata stored
    end
end

function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--p_range", "-p"
        arg_type = String
        default = "0.0:1.0:10"
        help = "range of p_scan"
        "--p_fixed_name", "-f"
        arg_type = String
        default = "p_ctrl"
        help = "fixed p value name"
        "--p_fixed_value", "-v"
        arg_type = Float64
        default = 0.0
        help = "fixed p value"
        "--L", "-L"
        arg_type = Int
        default = 8
        help = "system size"
        "--random", "-r"
        action = :store_true
        help = "use random seed"
        "--ancilla", "-a"
        arg_type = Int
        default = 0
        help = "number of ancilla"
        "--maxdim", "-m"
        arg_type = Int
        default = 10
        help = "set the maximal bond dim"
        "--cutoff", "-c"
        arg_type = Float64
        default = 1e-15
        help = "set the cutoff"
        "--n_chunk_realizations", "-n"
        arg_type = Int
        default = 1
        help = "number of realizations handled per cpu worker"
        "--job_id", "-j"
        arg_type = Int
        default = 0
        help = "job id"
        "--output_dir", "-o"
        arg_type = String
        default = "/scratch/ty296/json_data"
        help = "output directory"
        "--store_sv"
        action = :store_true
        help = "store singular values (uses HDF5 format), otherwise store scalar entropy only (JSON format)"
    end
    return parse_args(s)
end

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
        println("cutoff: ", args["cutoff"])
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
        filename = "$(args["output_dir"])/$(args["p_fixed_name"])$(args["p_fixed_value"])_$(args["job_id"])_worker$(rank)_$(hostname)_$(pid)_a$(args["ancilla"])_L$(args["L"]).h5"
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
                @time O, entropy_data, max_bond = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["cutoff"],seed;sv=store_singular_values)
                
                result_count += 1
                
                # Store result directly to HDF5
                store_result_hdf5(filename, result_count, O, entropy_data, max_bond, 
                                p_ctrl, p_proj, p, i, args, args["ancilla"])
                
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
                    results = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["cutoff"],seed;sv=store_singular_values)
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