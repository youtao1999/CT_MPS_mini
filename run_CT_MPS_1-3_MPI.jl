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
include("run_CT_MPS_1-3_single_shot.jl")

function parse_my_args_MPI()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--job_file"
        "--p_fixed_name"
        "--p_fixed_value"
        "--ancilla"
        "--maxdim"
        "--threshold"
        "--L"
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
    
    args = parse_my_args_MPI()
    
    # Read job_list from file to avoid command line length limits
    job_file = args["job_file"]
    println("Reading jobs from file: ", job_file)
    job_list = JSON.parsefile(job_file)
    println("Loaded ", length(job_list), " jobs")
    
    # Convert other arguments to proper types
    p_fixed_name = args["p_fixed_name"]
    p_fixed_value = parse(Float64, args["p_fixed_value"])
    ancilla = parse(Int, args["ancilla"])
    maxdim = parse(Int, args["maxdim"])
    threshold = parse(Float64, args["threshold"])
    L = parse(Int, args["L"])
    
    # Only master process prints the full configuration
    if rank == 0
        println("job_list: ", job_list)
        println("p_fixed_name: ", p_fixed_name)
        println("p_fixed_value: ", p_fixed_value)
        println("L: ", L)
        println("ancilla: ", ancilla)
        println("maxdim: ", maxdim)
        println("threshold: ", threshold)
        println("Expected workers: ", min(size, length(job_list)))        
    end

    # Extract job for this worker
    seed, p, eps, output_dir = job_list[rank+1]  # Julia is 1-indexed
    
    # Convert job parameters to proper types
    seed = Int(seed)
    p = Float64(p)
    eps = Float64(eps)
    output_dir = String(output_dir)

    println("Worker $rank assigned job: seed=$seed, p=$p, eps=$eps")
    println("Worker $rank output_dir: $(output_dir)")

    # Setup parameters for computation
    p_vary_name = p_fixed_name == "p_ctrl" ? "p_proj" : "p_ctrl"
    p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p
    p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p
    
    filename = "$(output_dir)/$(seed)_a$(ancilla)_L$(L)_$(p_vary_name)$(p)_eps$(eps).h5"
    println("Worker $rank will write to file: $filename")
    
    O, sv_array, max_bond, eps = main_interactive(L, p_ctrl, p_proj, ancilla, maxdim,threshold,eps,seed;time_average=10, builtin=false)
    # Store result directly to HDF5
    store_result_hdf5_single_shot(filename, sv_array, max_bond, O, p_ctrl, p_proj, args, seed, eps)
    println("Worker $rank stored result (seed=$seed, p=$p, eps=$eps) to HDF5")
    
    # Synchronize all workers before finishing
    MPI.Barrier(comm)
    
    if rank == 0
        println("All workers completed their tasks")
    end
    
    # Explicitly finalize MPI (following working script pattern)
    MPI.Finalize()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
