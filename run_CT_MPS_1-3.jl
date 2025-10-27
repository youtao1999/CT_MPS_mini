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

"""
Store a single result directly to HDF5 file, optimized for large singular value arrays.
Uses compression and efficient chunking for large arrays.
Appends results to existing file or creates new file.
"""
function store_result_hdf5_single_shot(filename::String, singular_values::Union{Vector{Float64},Vector{Vector{Float64}}}, max_bond::Int, O::Float64,
                          p_ctrl::Float64, p_proj::Float64, 
                          args::Dict, seed::Int, eps::Float64)
    
    # # Delete file if it exists to ensure clean overwrite
    if isfile(filename)
        rm(filename)
    end

    # create filename
    h5open(filename, "cw") do file
        # Handle different data types for chunking
        if isa(singular_values, Vector{Float64})
            # 1D case
            chunk_size = (min(1000, length(singular_values)),)
            sv_data = singular_values
        else
            # Vector{Vector{Float64}} case - convert to 2D matrix
            max_len = maximum(length.(singular_values))
            # the first dimension is the number of time steps, the second dimension is the number of singular values
            sv_matrix = zeros(Float64, length(singular_values), max_len)
            for (i, sv_vec) in enumerate(singular_values)
                sv_matrix[i, 1:length(sv_vec)] = sv_vec
            end
            chunk_size = (min(1000, size(sv_matrix, 1)), min(1000, size(sv_matrix, 2)))
            sv_data = sv_matrix
        end
        
        sv_dataset = create_dataset(file, "singular_values", datatype(Float64), dataspace(sv_data),
        chunk=chunk_size, 
        shuffle=true, deflate=6)
        write(sv_dataset, sv_data)
        
        # Add metadata as attributes to the compressed dataset
        attributes(sv_dataset)["L"] = args["L"]
        attributes(sv_dataset)["ancilla"] = args["ancilla"]
        attributes(sv_dataset)["maxdim"] = args["maxdim"]
        attributes(sv_dataset)["max_bond"] = max_bond
        attributes(sv_dataset)["O"] = O
        attributes(sv_dataset)["seed"] = seed
        attributes(sv_dataset)["p_ctrl"] = p_ctrl
        attributes(sv_dataset)["p_proj"] = p_proj
        attributes(sv_dataset)["eps"] = eps
    end
end

"""
Read back exactly what was stored by store_result_hdf5_single_shot.
Returns a tuple: (singular_values, metadata_dict)
"""
function read_hdf5_single_shot(filename::String)
    if !isfile(filename)
        error("File $filename does not exist")
    end
    
    h5open(filename, "r") do file
        # Read the singular values dataset
        sv_dataset = file["singular_values"]
        raw_data = read(sv_dataset)

        # Convert back to appropriate format
        if length(size(raw_data)) == 1
            # Single time step case
            singular_values = raw_data
            println("Not time averaged. 1D singular values")
        else
            # Multiple time steps case - convert back to Vector{Vector{Float64}}
            # Remove zero-padding by finding the last non-zero element in each row
            singular_values = Vector{Vector{Float64}}()
            for i in 1:size(raw_data, 1)
                row = raw_data[i, :]
                # Find last non-zero element (or use all if no zeros)
                last_nonzero = findlast(x -> x != 0.0, row)
                if last_nonzero === nothing
                    # All zeros - this shouldn't happen but handle gracefully
                    push!(singular_values, Float64[])
                else
                    push!(singular_values, row[1:last_nonzero])
                end
            end
            println("Time averaged. Converted back to Vector{Vector{Float64}} format")
        end
        
        # Read all attributes (metadata)
        attrs = attributes(sv_dataset)
        metadata = Dict{String, Any}()
        
        # Extract all the stored attributes
        metadata["L"] = read(attrs["L"])
        metadata["ancilla"] = read(attrs["ancilla"])
        metadata["maxdim"] = read(attrs["maxdim"])
        metadata["max_bond"] = read(attrs["max_bond"])
        metadata["O"] = read(attrs["O"])
        metadata["seed"] = read(attrs["seed"])
        metadata["p_ctrl"] = read(attrs["p_ctrl"])
        metadata["p_proj"] = read(attrs["p_proj"])
        metadata["eps"] = read(attrs["eps"])
        return singular_values, metadata
    end
end


function main_interactive(L::Int,p_ctrl::Float64,p_proj::Float64,ancilla::Int,maxdim::Int,threshold::Float64, eps::Float64,seed::Int;n::Int=0,time_average::Union{Int,Nothing}=nothing)
    ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1 // 3, 2 // 3]),_maxdim=maxdim, builtin=false,_eps=eps, passthrough=true)
    # println("ct_f memory usage: ", Base.summarysize(ct_f) / 2^20, " MB")
    i=1
    T_max = ancilla ==0 ? 2*(ct_f.L^2) : div(ct_f.L^2,2)
    for idx in 1:T_max
        i =CT.random_control!(ct_f,i,p_ctrl,p_proj)
        # println("norm: ", norm(ct_f.mps))
        # @show maxlinkdim(ct_f.mps)
        heap_memory_usage = Base.gc_live_bytes() / 1024^2
        max_rss = Sys.maxrss() / 1024^2
        println("$(idx) heap memory usage: $(heap_memory_usage) MB, Max RSS: $(max_rss) MB")
    end
    O=CT.order_parameter(ct_f)
    max_bond= CT.max_bond_dim(ct_f.mps)
    if ancilla ==0 
        if time_average !== nothing && time_average > 1
            # Multiple time steps - return 2D data (list of arrays)
            sv_arr_list = Vector{Vector{Float64}}()
            for _ in 1:time_average
                sv_arr=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2),threshold, eps;positivedefinite=false,n=n,sv=true)
                # println(sum(abs2.(sv_arr)))
                push!(sv_arr_list, sv_arr)
                # println(size(sv_arr))
                i =CT.random_control!(ct_f,i,p_ctrl,p_proj)
            end
            # implement time averaging: store the sv_arrs for multiple time steps
            # println(size(sv_arr_list))
            return O, sv_arr_list, max_bond, ct_f._eps
        else
            # Single time step - return 1D data
            sv_arr=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2),threshold, eps;positivedefinite=false,n=n,sv=true)
            return O, sv_arr, max_bond, ct_f._eps
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

function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--L", "-L"
        arg_type = Int
        help = "system size"
        "--p_ctrl", "-c"
        arg_type = Float64
        help = "set the p_ctrl"
        "--p_proj", "-p"
        arg_type = Float64
        help = "set the p_proj"
        "--ancilla", "-a"
        arg_type = Int
        help = "number of ancilla"
        "--maxbond", "-b"
        arg_type = Int
        help = "set the maximal bond dim"
        "--threshold", "-t"
        arg_type = Float64
        help = "set the cutoff"
        "--eps", "-e"
        arg_type = Float64
        help = "set the svd cutoff"
        "--seed", "-s"
        arg_type = Int
        help = "set the seed"
        "--output_dir", "-o"
        arg_type = String
        default = "/scratch/ty296/test_output"
        help = "output directory"
    end
    return parse_args(s)
end

function main()
    println("Uses threads: ",BLAS.get_num_threads())
    println("Uses backends: ",BLAS.get_config())
    args = parse_my_args()
    
    println("L: ", args["L"])
    println("ancilla: ", args["ancilla"])
    println("maxbond: ", args["maxbond"])
    println("threshold: ", args["threshold"])
    println("eps: ", args["eps"])
    println("seed: ", args["seed"])
    println("output_dir: ", args["output_dir"])
    
    # Use HDF5 format for large singular value arrays
    filename = "$(args["output_dir"])/L$(args["L"])_p_ctrl$(args["p_ctrl"])_p_proj$(args["p_proj"])_ancilla$(args["ancilla"])_maxbond$(args["maxbond"])_threshold$(args["threshold"])_eps$(args["eps"])_seed$(args["seed"]).h5"
    
    # Get results as tuple with singular values
    @time O, entropy_data, max_bond, _eps = main_interactive(args["L"], args["p_ctrl"], args["p_proj"], args["ancilla"],args["maxbond"],args["threshold"],args["eps"],args["seed"];)
    
    # Store result directly to HDF5
    store_result_hdf5_single_shot(filename, entropy_data, max_bond, O, args["p_ctrl"], args["p_proj"], args, args["seed"], args["eps"])
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end




# julia --project=CT run_CT_MPS_1-3.jl --p_range "0.2" --p_fixed_name "p_ctrl" --p_fixed_value 0.0 --L 8 --threshold 1e-15 --n_chunk_realizations 1 --random --ancilla 0 --maxdim 64 --output_dir "/scratch/ty296/debug_sv" --store_sv