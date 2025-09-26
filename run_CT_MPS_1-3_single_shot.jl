# run_CT_MPS_1-3.jl for single seed and single p_ctrl

include("run_CT_MPS_1-3.jl")

"""
Store a single result directly to HDF5 file, optimized for large singular value arrays.
Uses compression and efficient chunking for large arrays.
Appends results to existing file or creates new file.
"""
function store_result_hdf5(filename::String, singular_values::Array{Float64, 1}, max_bond::Int, O::Float64,
                          p_ctrl::Float64, p_proj::Float64, 
                          args::Dict, seed::Int, _eps::Float64)
    
    # # Delete file if it exists to ensure clean overwrite
    if isfile(filename)
        rm(filename)
    end

    # create filename
    h5open(filename, "cw") do file
        sv_dataset = create_dataset(file, "singular_values", datatype(Float64), dataspace(singular_values),
        chunk=(min(1000, length(singular_values)),), 
        shuffle=true, deflate=6)
        write(sv_dataset, singular_values)
        
        # Add metadata as attributes to the compressed dataset
        attributes(sv_dataset)["L"] = args["L"]
        attributes(sv_dataset)["ancilla"] = args["ancilla"]
        attributes(sv_dataset)["maxdim"] = args["maxdim"]
        attributes(sv_dataset)["max_bond"] = max_bond
        attributes(sv_dataset)["O"] = O
        attributes(sv_dataset)["seed"] = seed
        attributes(sv_dataset)["p_ctrl"] = p_ctrl
        attributes(sv_dataset)["p_proj"] = p_proj
    end
end

"""
Read back exactly what was stored by store_result_hdf5.
Returns a tuple: (singular_values, metadata_dict)
"""
function read_hdf5(filename::String)
    if !isfile(filename)
        error("File $filename does not exist")
    end
    
    h5open(filename, "r") do file
        # Read the singular values dataset
        sv_dataset = file["singular_values"]
        singular_values = read(sv_dataset)
        
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
        
        return singular_values, metadata
    end
end

function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--p_vary", "-p"
        arg_type = Float64
        help = "vary p value"
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
        "--seed", "-s"
        arg_type = Int
        help = "seed"
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
    
    # Calculate maxdim = 2^(L/2) if default value is being used
    if args["maxdim"] == 64  # Default value we set
        args["maxdim"] = 2^div(args["L"], 2)
    end
    
    p_vary = args["p_vary"]
    p_fixed_name = args["p_fixed_name"]
    p_vary_name = p_fixed_name == "p_ctrl" ? "p_proj" : "p_ctrl"
    p_fixed_value = args["p_fixed_value"]
    println(p_vary_name, p_vary, p_fixed_name, p_fixed_value)
    println("L: ", args["L"])
    println("ancilla: ", args["ancilla"])
    println("maxdim: ", args["maxdim"])
    println("threshold: ", args["threshold"])
    println("seed: ", args["seed"])
    
    
    # Choose storage format based on command line argument
    # Use HDF5 format for large singular value arrays
    filename = "$(args["output_dir"])/$(args["seed"])_a$(args["ancilla"])_L$(args["L"])_$(p_vary_name)$(p_vary).h5"
    
    # Initialize parameters for this iteration
    p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p_vary
    p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p_vary
    # Get results as tuple with singular values
    @time O, sv_array, max_bond, _eps = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["threshold"],args["seed"];sv=true)
    
    
    # Store result directly to HDF5
    store_result_hdf5(filename, sv_array, max_bond, O,
                    p_ctrl, p_proj, args, args["seed"], _eps)
    
    println("Stored result to HDF5")    
end

main()
