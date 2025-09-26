# println("initial memory: ", Sys.maxrss() / 1024^2, " MB")
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

function sv_check(mps::MPS, eps::Float64, L::Int)
    mps_ = orthogonalize(copy(mps), div(L,2))
    _, S = svd(mps_[div(L,2)], (linkind(mps_, div(L,2)),); cutoff=eps)
    return array(diag(S))
end

function main_interactive(L::Int,p_ctrl::Float64,p_proj::Float64,ancilla::Int,maxdim::Int,threshold::Float64,seed::Int;sv::Bool=false,n::Int=0)
    println("seed: ", seed)
    ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=maxdim, _maxdim0=maxdim, builtin=true)
    i=1
    # T_max = 1
    T_max = ancilla ==0 ? 2*(ct_f.L^2) : div(ct_f.L^2,2)
    for idx in 1:T_max
        i =CT.random_control!(ct_f,i,p_ctrl,p_proj)
        # println(sv_check_dict)
        println(idx, " heap memory usage: ", Base.gc_live_bytes()/ 1024^2, " MB")
    end
    O=CT.order_parameter(ct_f)
    max_bond= CT.max_bond_dim(ct_f.mps)
    if ancilla ==0 
        if sv
            sv_arr=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2),threshold;sv=sv,positivedefinite=false,n=n)
            # println(length(sv_arr), "lower bound sv: ", sv_arr[end])
            return O, sv_arr, max_bond
        else
            EE=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2),threshold;n=n)
            # ct_f.mps=initial_state # resetting the mps for memory benchmarking purposes
            return Dict("O" => O, "EE" => EE, "max_bond" => max_bond, "p_ctrl" => p_ctrl, "p_proj" => p_proj, "n" => n)
        end
    else
        if sv
            SA=CT.von_Neumann_entropy(ct_f.mps,1,threshold;sv=sv)
            return O, SA, max_bond
        else
            SA=CT.von_Neumann_entropy(ct_f.mps,1,threshold;n=n)
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
        return range(start, stop, length=num) |> collect |> reverse
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
                          p_ctrl::Float64, p_proj::Float64, realization::Int, 
                          args::Dict, seed::Int)
    
    # # Delete file if it exists to ensure clean overwrite
    if isfile(filename)
        rm(filename)
    end

    # Determine if we need to create or append to file
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
        meta_group["realization"] = realization
        meta_group["seed"] = seed

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



"""
Read results from HDF5 file with optimized structure. Returns a vector of dictionaries containing the results.
Handles both metadata and large singular value arrays efficiently.
Example usage: 
    results = read_results_hdf5("output.h5")
    singular_values = results[1]["sv_arr"]
    metadata_only = read_results_hdf5("output.h5", load_sv_arrays=false)  # Skip large arrays
"""
function read_results_hdf5(filename::String; load_sv_arrays::Bool=true)
    results = Vector{Dict{String,Any}}()
    
    h5open(filename, "r") do file        
        if haskey(file, "metadata") && haskey(file, "singular_values")
            # Optimized format with separate metadata and singular_values groups
            metadata_group = file["metadata"]
            sv_arrays_group = file["singular_values"]
            
            # Get all result groups from metadata
            result_groups = filter(x -> startswith(x, "result_"), keys(metadata_group))
            
            for group_name in sort(result_groups, by=x->parse(Int, split(x, "_")[2]))
                result_dict = Dict{String,Any}()
                
                # Load metadata
                meta_group = metadata_group[group_name]
                for key in keys(meta_group)
                    if meta_group[key] isa HDF5.Dataset
                        result_dict[key] = read(meta_group[key])
                    elseif meta_group[key] isa HDF5.Group
                        # Handle nested groups (like args)
                        subdict = Dict{String,Any}()
                        subgroup = meta_group[key]
                        for subkey in keys(subgroup)
                            subdict[subkey] = read(subgroup[subkey])
                        end
                        result_dict[key] = subdict
                    end
                end
                
                # Load singular value arrays if requested (always present in HDF5 files)
                if load_sv_arrays && haskey(sv_arrays_group, group_name)
                    result_dict["sv_arr"] = read(sv_arrays_group[group_name])
                end
                
                push!(results, result_dict)
            end
            
        else
            # Legacy format (version 1.0) - maintain backward compatibility
            result_groups = filter(x -> startswith(x, "result_"), keys(file))
            
            for group_name in sort(result_groups, by=x->parse(Int, split(x, "_")[2]))
                result_dict = Dict{String,Any}()
                group = file[group_name]
                
                for key in keys(group)
                    if group[key] isa HDF5.Dataset
                        # Skip large arrays if requested
                        if !load_sv_arrays && key == "sv_arr"
                            continue
                        end
                        result_dict[key] = read(group[key])
                    elseif group[key] isa HDF5.Group
                        # Handle nested groups (like args)
                        subdict = Dict{String,Any}()
                        subgroup = group[key]
                        for subkey in keys(subgroup)
                            subdict[subkey] = read(subgroup[subkey])
                        end
                        result_dict[key] = subdict
                    end
                end
                
                push!(results, result_dict)
            end
        end
    end
    
    return results
end

"""
Inspect HDF5 file structure and get summary information without loading large arrays.
Useful for checking file contents and sizes before loading.
"""
function inspect_hdf5_file(filename::String)
    h5open(filename, "r") do file
        println("=== HDF5 File Inspection: $filename ===")
        
        if haskey(file, "metadata") && haskey(file, "singular_values")
            # Format with separate metadata and singular_values groups
            metadata_group = file["metadata"]
            sv_arrays_group = file["singular_values"]
            
            println("\nMetadata Groups: $(length(keys(metadata_group)))")
            println("Singular Value Arrays: $(length(keys(sv_arrays_group)))")
            
            # Sample a few results to show structure
            result_groups = filter(x -> startswith(x, "result_"), keys(metadata_group))
            sample_size = min(3, length(result_groups))
            
            println("\nSample Results (first $sample_size):")
            for (i, group_name) in enumerate(sort(result_groups)[1:sample_size])
                println("  $group_name:")
                meta_group = metadata_group[group_name]
                for key in keys(meta_group)
                    if key == "sv_array_length"
                        sv_length = read(meta_group[key])
                        println("    $key: $sv_length")
                        if haskey(sv_arrays_group, group_name)
                            sv_size_bytes = sizeof(sv_arrays_group[group_name])
                            println("    sv_array_size: $(round(sv_size_bytes/1024, digits=2)) KB")
                        end
                    else
                        value = read(meta_group[key])
                        if isa(value, AbstractArray) && length(value) > 5
                            println("    $key: Array of length $(length(value))")
                        else
                            println("    $key: $value")
                        end
                    end
                end
            end
            
        else
            # Legacy format
            result_groups = filter(x -> startswith(x, "result_"), keys(file))
            println("\nLegacy format with $(length(result_groups)) result groups")
            
            # Sample a few results
            sample_size = min(3, length(result_groups))
            println("\nSample Results (first $sample_size):")
            for (i, group_name) in enumerate(sort(result_groups)[1:sample_size])
                println("  $group_name:")
                group = file[group_name]
                for key in keys(group)
                    if key == "sv_arr" && group[key] isa HDF5.Dataset
                        sv_dataset = group[key]
                        sv_size = size(sv_dataset)
                        sv_size_bytes = sizeof(sv_dataset)
                        println("    $key: Array of size $sv_size ($(round(sv_size_bytes/1024, digits=2)) KB)")
                    else
                        value = read(group[key])
                        if isa(value, AbstractArray) && length(value) > 5
                            println("    $key: Array of length $(length(value))")
                        else
                            println("    $key: $value")
                        end
                    end
                end
            end
        end
        
        println("\n=== End Inspection ===")
    end
end

function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--p_range", "-p"
        arg_type = String
        help = "range of p_scan"
        "--p_fixed_name", "-f"
        arg_type = String
        default = "p_ctrl"
        help = "fixed p value name"
        "--p_fixed_value", "-v"
        arg_type = Float64
        help = "fixed p value"
        "--L", "-L"
        arg_type = Int
        help = "system size"
        "--random", "-r"
        action = :store_true
        help = "use random seed"
        "--ancilla", "-a"
        arg_type = Int
        help = "number of ancilla"
        "--maxdim", "-m"
        arg_type = Int
        default = 64  # Will be calculated as 2^(L/2) if not provided
        help = "set the maximal bond dim"
        "--threshold", "-t"
        arg_type = Float64
        help = "set the cutoff"
        "--n_chunk_realizations", "-n"
        arg_type = Int
        help = "number of realizations handled per cpu worker"
        "--job_counter", "-c"
        arg_type = Int
        help = "job counter"
        "--output_dir", "-o"
        arg_type = String
        default = "/scratch/ty296/test_output"
        help = "output directory"
        "--store_sv"
        action = :store_true
        help = "store singular values (uses HDF5 format), otherwise store scalar entropy only (JSON format)"
    end
    return parse_args(s)
end

function main()
    println("Uses threads: ",BLAS.get_num_threads())
    println("Uses backends: ",BLAS.get_config())
    args = parse_my_args()
    
    # Calculate maxdim = 2^(L/2) if default value is being used
    if args["maxdim"] == 64  # Default value we set
        args["maxdim"] = 2^div(args["L"], 2)
    end
    
    p_range = parse_p_range(args["p_range"])
    p_fixed_name = args["p_fixed_name"]
    p_fixed_value = args["p_fixed_value"]
    println("p_range: ", p_range)
    println("p_fixed_name: ", p_fixed_name)
    println("p_fixed_value: ", p_fixed_value)
    println("L: ", args["L"])
    println("ancilla: ", args["ancilla"])
    println("maxdim: ", args["maxdim"])
    println("threshold: ", args["threshold"])
    println("n_chunk_realizations: ", args["n_chunk_realizations"])
    
    
    # Choose storage format based on command line argument
    store_singular_values = args["store_sv"]
    # println("store_singular_values: ", store_singular_values)
    # println("random: ", args["random"])
    if store_singular_values
        # Use HDF5 format for large singular value arrays
        filename = "$(args["output_dir"])/$(args["job_counter"])_a$(args["ancilla"])_L$(args["L"]).h5"
        result_count = 0
        
        #scan over p_range
        for p in p_range
            # Initialize parameters for this iteration
            p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p
            p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p
            
            for i in 1:(args["n_chunk_realizations"])
                if args["random"]
                    seed = rand(1:10000)
                else
                    seed = i - 1 + args["job_counter"]  * args["n_chunk_realizations"]
                end
                # Get results as tuple with singular values
                @time O, entropy_data, max_bond = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["threshold"],seed;sv=store_singular_values)
                
                result_count += 1
                
                # Store result directly to HDF5
                store_result_hdf5(filename, result_count, O, entropy_data, max_bond, 
                                p_ctrl, p_proj, i, args, seed)
                
                println("Stored result $result_count to HDF5")
            end
        end
        
        println("Saved $result_count results to $filename (HDF5 format)")
        
    else
        # Use JSON format for scalar entropy values only
        filename = "$(args["output_dir"])/$(args["job_counter"])_a$(args["ancilla"])_L$(args["L"]).json"
        result_count = 0
        
        open(filename, "w") do f
            #scan over p_range
            for p in p_range
                # Initialize parameters for this iteration
                p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p
                p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p
                
                for i in 1:(args["n_chunk_realizations"])
                    if args["random"]
                        seed = rand(1:10000)
                    else
                        seed = i - 1 + args["job_counter"]  * args["n_chunk_realizations"]
                    end 
                    
                    # Get results as dictionary (scalar entropy only)
                    results = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["threshold"],seed;sv=store_singular_values)
                    data_to_serialize = merge(results, Dict("args" => args, "realization number" => i))
                    
                    # Write each result as a separate line (JSON Lines format)
                    println(f, JSON.json(data_to_serialize))
                    result_count += 1
                end
            end
        end
        
        println("Saved $result_count results to $filename (JSON format)")
    end
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end




# julia --project=CT run_CT_MPS_1-3.jl --p_range "0.2" --p_fixed_name "p_ctrl" --p_fixed_value 0.0 --L 8 --threshold 1e-15 --n_chunk_realizations 1 --random --ancilla 0 --maxdim 64 --output_dir "/scratch/ty296/debug_sv" --store_sv