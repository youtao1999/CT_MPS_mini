using Pkg
Pkg.activate("CT")
using CT
# benchmark adder_MPO
using ITensors
# initialize random state
using .CT: CT_MPS, all_bond_dim
using Random
using ProgressMeter
using JSON
using Statistics
include("run_CT_MPS_1-3.jl")

L = 12
ancilla = 0
folded = true
seed_vec = 123457
xj = Set([1//3, 2//3])
_maxdim = 2^(div(L,2))
_maxdim0 = 50
seed = 123457
x0 = nothing
rng = MersenneTwister(seed_vec)
rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
# ct_h = CT_MPS(L=L,xj=xj,folded=folded,_maxdim=_maxdim,ancilla=ancilla,seed_vec=seed_vec,seed=seed,x0=x0,debug=false);
# ct_t = CT_MPS(L=L,xj=xj,folded=folded,_maxdim=_maxdim,ancilla=ancilla,seed_vec=seed_vec,seed=seed,x0=x0,debug=false,passthrough=true);

# for i1 in 1:L
#     println(i1, " iterative: ", all_bond_dim(ct_h.adder[i1]))
#     println(i1, " passthrough: ", all_bond_dim(ct_t.adder[i1]))
# end
global i, j
i = 1
j = 1
p_ctrl = 0.4
p_proj = 0.7
n=0
threshold = 1e-15
_eps = 0.0
ancilla = 0
T_max = ancilla == 0 ? 2*(L^2) : div(L^2,2)
ensemble_size = 50
# Write a single string
seed_vec_arr = [seed_vec + i for i in 1:ensemble_size]
entropy_t_arr_list = [zeros(T_max) for _ in 1:ensemble_size]
@showprogress "Ensemble trajectories: " for (idx, seed_vec_i) in enumerate(seed_vec_arr)
    # O, sv_arr, max_bond, _eps_out = main_interactive(L,p_ctrl,p_proj,ancilla,_maxdim,threshold, _eps,seed_vec_i;n=n,time_average=nothing)
    # entropy_t_arr_list[idx] = sv_arr

    ct_t = CT_MPS(L=L,xj=xj,folded=folded,_maxdim=_maxdim,ancilla=ancilla,seed_vec=seed_vec_i,seed=seed,x0=x0,_eps=_eps,debug=false,passthrough=true);
    j_local = 1
    for t in 1:T_max
        j_local = CT.random_control!(ct_t,j_local,p_ctrl,p_proj)
        entropy_t_arr_list[idx][t] = CT.von_Neumann_entropy(ct_t.mps, div(ct_t.L,2), threshold, _eps;positivedefinite=false,n=n)
    end
end
# Compute mean and std across ensemble
entropy_t_arr = mean(hcat(entropy_t_arr_list...), dims=2)[:] 
entropy_t_std = std(hcat(entropy_t_arr_list...), dims=2)[:]

# Save entropy data to JSON
output_data = Dict(
    "time_steps" => collect(1:T_max),
    "entropy_mean" => entropy_t_arr,
    "entropy_std" => entropy_t_std,
    "entropy_trajectories" => entropy_t_arr_list,
    "parameters" => Dict(
        "L" => L,
        "p_ctrl" => p_ctrl,
        "p_proj" => p_proj,
        "ancilla" => ancilla,
        "maxdim" => _maxdim,
        "seed_vec" => seed_vec,
        "seed" => seed,
        "folded" => folded,
        "ensemble_size" => ensemble_size
    )
)

output_filename = "entropy_evolution_data.json"
open(output_filename, "w") do f
    JSON.print(f, output_data, 4)
end

println("Entropy data saved to $output_filename")
println("Final entropy: $(entropy_t_arr[end])")
println("Max entropy: $(maximum(entropy_t_arr))")
println("Min entropy: $(minimum(entropy_t_arr))")