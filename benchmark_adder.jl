using Pkg
Pkg.activate("CT")
using CT
# benchmark adder_MPO
using ITensors
# initialize random state
using .CT: CT_MPS
using Random
using ProgressMeter

L = 24
ancilla = 0
folded = true
seed_vec = 123457
xj = Set([1//3, 2//3])
_maxdim = 2^(div(L,2))
_maxdim0 = 50
_eps = 0.0
seed = 123457
x0 = nothing
rng = MersenneTwister(seed_vec)
rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
ct_h = CT_MPS(L=L,xj=xj,folded=folded,_maxdim=_maxdim,ancilla=ancilla,seed_vec=seed_vec,seed=seed,x0=x0,debug=false);
ct_t = CT_MPS(L=L,xj=xj,folded=folded,_maxdim=_maxdim,ancilla=ancilla,seed_vec=seed_vec,seed=seed,x0=x0,debug=false,passthrough=true);

for i1 in 1:L
    println(all_bond_dim(ct_h.adder[i1]))
    pirntln(all_bond_dim(ct_t.adder[i1]))
end
# global i, j
# i = 1
# j = 1
# p_ctrl = 0.4
# p_proj = 0.7

# ancilla = 0
# T_max = ancilla == 0 ? 2*(ct_h.L^2) : div(ct_h.L^2,2)
# # Write a single string
# for _ in 1:T_max
#     global i, j
#     @time i = CT.random_control!(ct_h,i,p_ctrl,p_proj)
#     @time j = CT.random_control!(ct_t,j,p_ctrl,p_proj)
# end