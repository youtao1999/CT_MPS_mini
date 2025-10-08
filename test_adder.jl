using Pkg
Pkg.activate("CT")
using Revise
using CT

# using Random
using ITensors
# using Plots
# using Statistics
# using Profile
# using BenchmarkTools



ct=CT.CT_MPS(L=12,seed=0,seed_C=0,seed_m=0,folded=true,store_op=true,store_vec=false,ancilla=0,_maxdim0=60,xj=Set([1 // 3, 2 // 3]),debug=true,simplified_U=true, x0=8//2^12,)

mps_new = apply(ct.adder[1], ct.mps)

println(CT.mps_element(ct.mps,"000000001000"[ct.ram_phy]))

println(CT.mps_element(mps_new,"001010110011"[ct.ram_phy]))

000000001000
001010110011
001010101011

# result = parse(Int, "001010101011", base=2) + parse(Int, "000000001000", base=2)
# string(result, base=2)