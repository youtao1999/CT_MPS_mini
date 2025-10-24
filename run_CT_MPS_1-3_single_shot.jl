# run_CT_MPS_1-3.jl for single seed and single p_ctrl

include("run_CT_MPS_1-3.jl")


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
        "--eps", "-e"
        arg_type = Float64
        help = "set the eps"
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
    println(p_vary_name, " ", p_vary, " ", p_fixed_name, " ", p_fixed_value)
    println("L: ", args["L"])
    println("ancilla: ", args["ancilla"])
    println("maxdim: ", args["maxdim"])
    println("seed: ", args["seed"])
    println("eps: ", args["eps"])
    
    # Choose storage format based on command line argument
    # Use HDF5 format for large singular value arrays
    filename = "$(args["output_dir"])/$(args["seed"])_a$(args["ancilla"])_L$(args["L"])_$(p_fixed_name)$(p_fixed_value)_$(p_vary_name)$(p_vary)_eps$(args["eps"]).h5"
    
    # Initialize parameters for this iteration
    p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p_vary
    p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p_vary
    # Get results as tuple with singular values
    @time O, sv_array, max_bond, eps = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["threshold"],args["eps"],args["seed"];time_average=10)
    # Store result directly to HDF5
    store_result_hdf5_single_shot(filename, sv_array, max_bond, O,
                    p_ctrl, p_proj, args, args["seed"], eps)
    
    println("Stored result to HDF5")    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# apptainer exec --bind /ospool:/ospool julia_CT_OSG.sif julia --project=CT run_CT_MPS_1-3_single_shot.jl --p_vary 0.7 --p_fixed_name p_ctrl --p_fixed_value 0.4 --L 8 --seed 42 --ancilla 0 --threshold 1e-15 --eps 1e-15 --output_dir "/ospool/ap20/data/ty296/test"
