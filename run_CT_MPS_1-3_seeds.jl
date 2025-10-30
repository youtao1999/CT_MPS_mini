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

function main_interactive(L::Int,p_ctrl::Float64,p_proj::Float64,ancilla::Int,maxdim::Int,cutoff::Float64,seed::Int;sv::Bool=false,n::Int=0)
    ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=maxdim,_cutoff=cutoff, _maxdim0=maxdim)
    # println(CT.sv_check(ct_f.mps, cutoff, L))
    i=1
    T_max = ct_f.L
    # T_max = ancilla ==0 ? 2*(ct_f.L^2) : div(ct_f.L^2,2)
    for idx in 1:T_max
        i, sv_check_dict =CT.random_control!(ct_f,i,p_ctrl,p_proj)
        store_sv_check_json(sv_check_dict, seed, "/scratch/ty296/debug_sv")
        # println(CT.sv_check(ct_f.mps, cutoff, L))
    end
    O=CT.order_parameter(ct_f)
    max_bond= CT.max_bond_dim(ct_f.mps)
    sv_arr=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2);sv=sv,threshold=ct_f._cutoff,positivedefinite=false,n=n)
    # println(length(sv_arr), "lower bound sv: ", sv_arr[end])
    return O, sv_arr, max_bond
end

function store_sv_check_json(sv_check_dict::Dict{String, Any}, seed::Int, output_dir::String)
    filename = "$output_dir/sv_check_$(seed).json"
    if isfile(filename)
        append = true
    else
        append = false
    end
    open(filename, append=append) do f
        JSON.print(f, sv_check_dict)
    end
end

function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--seeds"
        help = "Seeds to run (range 'start:end' or comma-separated list '1,2,3,4,5')"
        arg_type = String
        default = "1"
    end
    return parse_args(s)
end

function parse_seeds(seeds_str::String)
    if contains(seeds_str, ':')
        parts = split(seeds_str, ':')
        if length(parts) == 2
            start_seed = parse(Int, strip(parts[1]))
            end_seed = parse(Int, strip(parts[2]))
            collect(start_seed:end_seed)
        else
            error("Invalid range format. Use 'start:end' (e.g., '20:30')")
        end
    else
        [parse(Int, strip(s)) for s in split(seeds_str, ',')]
    end
end

function main()
    args = parse_my_args()
    seeds_str = args["seeds"]
    seeds = parse_seeds(seeds_str)
    
    println("Running with seeds: ", seeds)
    
    results = []
    for seed in seeds
        # check if json file exists; if so, delete it 
        if isfile("/scratch/ty296/debug_sv/sv_check_$(seed).json")
            rm("/scratch/ty296/debug_sv/sv_check_$(seed).json")
        end
        println("Running seed: ", seed)
        O, sv_arr, max_bond = main_interactive(24, 0.0, 0.2, 0, Int(2^(9)), 1e-15, seed; sv=true, n=0)
        push!(results, (seed=seed, O=O, sv_arr=sv_arr, max_bond=max_bond))
        println("Seed $seed completed: O=$O, max_bond=$max_bond")
    end
    
    return results
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end