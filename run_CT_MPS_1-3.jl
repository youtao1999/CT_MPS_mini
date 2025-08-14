using Pkg
using ITensors
using Random
using LinearAlgebra
using MKL
using JSON
if !isdefined(Main, :CT)
    using CT
end
using Printf

using ArgParse
using Serialization

function main_interactive(L::Int,p_ctrl::Float64,p_proj::Float64,ancilla::Int,maxdim::Int,cutoff::Float64,seed::Int)

    ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=maxdim,_cutoff=cutoff, _maxdim0=maxdim)
    initial_maxdim = CT.max_bond_dim(ct_f.mps)
    println("initial mps: ", ct_f.mps[10])
    println("initial maxdim: ", initial_maxdim)
    i=1
    # T_max = ancilla ==0 ? 2*(ct_f.L^2) : div(ct_f.L^2,2)
    T_max = 10

    for idx in 1:T_max
        println(idx)
        i=CT.random_control!(ct_f,i,p_ctrl,p_proj)
    end
    O=CT.order_parameter(ct_f)
    max_bond= CT.max_bond_dim(ct_f.mps)
    if ancilla ==0 
        EE=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2); n=0)
        return Dict("O" => O, "EE" => EE, "max_bond" => max_bond, "p_ctrl" => p_ctrl, "p_proj" => p_proj)
    else
        SA=CT.von_Neumann_entropy(ct_f.mps,1; n=0)
        return Dict("O" => O, "SA" => SA, "max_bond" => max_bond, "p_ctrl" => p_ctrl, "p_proj" => p_proj)
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
        default = 1e-10
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
    end
    return parse_args(s)
end

function main()
    println("Uses threads: ",BLAS.get_num_threads())
    println("Uses backends: ",BLAS.get_config())
    args = parse_my_args()
    p_range = parse_p_range(args["p_range"])
    p_fixed_name = args["p_fixed_name"]
    p_fixed_value = args["p_fixed_value"]
    
    # Open file once for writing all results
    filename = "$(args["output_dir"])/$(args["job_id"])_a$(args["ancilla"])_L$(args["L"]).json"
    result_count = 0
    
    open(filename, "w") do f
        #scan over p_range
        for p in p_range
            # Initialize parameters for this iteration
            p_ctrl = p_fixed_name == "p_ctrl" ? p_fixed_value : p
            p_proj = p_fixed_name == "p_proj" ? p_fixed_value : p
            
            for i in 1:args["n_chunk_realizations"]
                if args["random"]
                    seed = rand(1:10000)
                else
                    seed = 0
                end
                results = main_interactive(args["L"], p_ctrl, p_proj, args["ancilla"],args["maxdim"],args["cutoff"],seed)
                data_to_serialize = merge(results, Dict("args" => args))
                
                # Write each result as a separate line (JSON Lines format)
                println(f, JSON.json(data_to_serialize))
                result_count += 1
            end
        end
    end
    
    println("Saved $result_count results to $filename")
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end




# julia --sysimage ~/.julia/sysimages/sys_itensors.so run_CT_MPS.jl --p 1 --L 8 --seed 0 --ancilla 0