include("run_CT_MPS_1-3.jl")
using Profile, PProf

L = 20
seed = 42
ancilla = 0
maxdim = 2^(div(L,2))
threshold = 1e-15
eps = 0.0
p_ctrl = 0.4
p_proj = 0.7
i = 1
ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1 // 3, 2 // 3]),_maxdim=maxdim, builtin=true,_eps=eps, passthrough=true)
# Profile.Allocs.@profile CT.random_control!(ct_f,i,p_ctrl,p_proj)

# main_interactive(L,p_ctrl,p_proj,ancilla,maxdim,threshold,eps,seed;n=0,time_average=nothing, builtin=true);

Profile.Allocs.@profile main_interactive(L,p_ctrl,p_proj,ancilla,maxdim,threshold,eps,seed;n=0,time_average=nothing, builtin=false)
PProf.Allocs.pprof()

# @code_warntype CT.apply_op!(ct_f.mps, CT.projector(ct_f, [0], [i]), eps, maxdim)

