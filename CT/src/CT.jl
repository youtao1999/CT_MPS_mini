module CT

using ITensors
using Random
using LinearAlgebra
import TensorCrossInterpolation as TCI
using TCIITensorConversion

# using InteractiveUtils

function __init__()
    ITensors.disable_warn_order()
end

# include("~/CT_MPS_mini/global_adder_passthrough.jl")

mutable struct CT_MPS
    L::Int
    store_vec::Bool
    store_op::Bool
    store_prob::Bool
    seed::Union{Int,Nothing}
    seed_vec::Union{Int,Nothing}
    seed_C::Union{Int,Nothing} # random seed for "circuit", i.e., choice of unitary (Bernoulli) and position of measurement (control)
    seed_m::Union{Int,Nothing} # random seed for "measurement outcome".
    x0::Union{Rational{Int},Rational{BigInt},Nothing}
    xj::Set
    ancilla::Int
    folded::Bool
    rng::Random.AbstractRNG
    rng_vec::Random.AbstractRNG
    rng_C::Random.AbstractRNG
    rng_m::Random.AbstractRNG
    qubit_site::Vector{Index{Int64}}
    phy_ram::Vector{Int}
    ram_phy::Vector{Int}
    phy_list::Vector{Int}
    _maxdim0::Int
    _eps::Float64
    _maxdim::Int
    mps::MPS
    vec_history::Vector{MPS}
    op_history::Vector{Vector{Any}}
    adder::Vector{Union{MPO,Nothing}}
    dw::Vector{Vector{Union{MPO,Nothing}}}
    debug::Bool
    simplified_U::Bool
    builtin::Bool
end

function CT_MPS(
    ; L::Int,
    store_vec::Bool=false,
    store_op::Bool=false,
    store_prob::Bool=false,
    seed::Union{Nothing,Int}=nothing,
    seed_vec::Union{Nothing,Int}=nothing,
    seed_C::Union{Nothing,Int}=nothing,
    seed_m::Union{Nothing,Int}=nothing,
    x0::Union{Rational{Int},Rational{BigInt},Nothing}=nothing,
    xj::Set=Set([1 // 3, 2 // 3]),
    ancilla::Int=0,
    folded::Bool=false,
    _maxdim0::Int=50,
    _eps::Float64=1e-10,
    _maxdim::Int=2^10,
    debug::Bool=false,
    simplified_U::Bool=false,
    builtin::Bool=false,
    passthrough::Bool=false,
    )
    rng = MersenneTwister(seed)
    rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
    rng_C = seed_C === nothing ? rng : MersenneTwister(seed_C)
    rng_m = seed_m === nothing ? rng : MersenneTwister(seed_m)
    qubit_site, ram_phy, phy_ram, phy_list = _initialize_basis(L,ancilla,folded)
    # println("initialize basis done")
    @time mps=_initialize_vector(L,ancilla,x0,folded,qubit_site,ram_phy,phy_ram,phy_list,rng_vec,_eps,_maxdim0)
    # println("initialize vector done")
    adder=[] # initialize empty adder
    if passthrough
        adder=adder_mpo_vec(L, xj, qubit_site, ancilla, folded, phy_list, phy_ram);
    else
        adder=[adder_MPO(i1,xj,qubit_site,L,phy_ram,phy_list) for i1 in 1:L];
    end
    # println("initialize adder done ", Base.summarysize(adder) / 2^20, " MB")
    # dw=[[dw_MPO(i1,xj,qubit_site,L,phy_ram,phy_list,order) for i1 in 1:L] for order in 1:2]
    dw=[]
    # println('initialize dw done')
    ct = CT_MPS(L, store_vec, store_op, store_prob, seed, seed_vec, seed_C, seed_m, x0, xj, ancilla, folded, rng, rng_vec, rng_C, rng_m, qubit_site, phy_ram, ram_phy, phy_list, _maxdim0, _eps, _maxdim, mps, [],[],adder,dw,debug,simplified_U,builtin)
    return ct
end

function _initialize_basis(L::Int,ancilla::Int,folded::Bool)

    qubit_site = siteinds("Qubit", L+ancilla) # RAM site index
    # ram_phy[actual in ram] = physical 
    if ancilla ==0
        ram_phy = folded ? [i for pairs in zip(1:(L÷2), reverse((L÷2+1):L)) for i in pairs] : collect(1:L)
    elseif ancilla ==1
        ram_phy = folded ? vcat(L+1,[i for pairs in zip(1:(L÷2), reverse((L÷2+1):L)) for i in pairs]) : vcat(collect(1:L),L+1)
        # if folded : [L+1, 1,L, ..] else [1,2,..,L,L+1]
    elseif ancilla ==2
        error("Not implemented yet")
    end

    # phy_ram[physical] = actual in ram
    phy_ram = fill(0, L+ancilla)
    for (ram, phy) in enumerate(ram_phy)
        phy_ram[phy] = ram
    end

    # phy_list = Dict(0 => 1:L,1=>1:L)[ancilla]
    phy_list = collect(1:L)
    return qubit_site, ram_phy, phy_ram, phy_list
end

function _initialize_vector(L::Int,ancilla::Int,x0::Union{Rational{Int},Rational{BigInt},Nothing},folded::Bool,qubit_site::Vector{Index{Int64}},ram_phy::Vector{Int},phy_ram::Vector{Int},phy_list::Vector{Int},rng_vec::Random.AbstractRNG,_eps::Float64,_maxdim0::Int)
    if ancilla == 0
        if x0 !== nothing
            vec_int = dec2bin(x0, L)
            vec_int_pos = [string(s) for s in lpad(string(vec_int, base=2), L, "0")] # physical index
            return MPS(ComplexF64, qubit_site, [vec_int_pos[ram_phy[i]] for i in 1:L])
        else
            println("initializing random MPS with linkdims $_maxdim0")
            return randomMPS(rng_vec, qubit_site, linkdims=_maxdim0)
        end
    elseif ancilla ==1
        if x0 !== nothing
            # This branch is only for debug purpose
            vec_int = dec2bin(x0, L)
            vec_int_pos = [string(s) for s in lpad(string(vec_int, base=2), L, "0")] # physical index
            push!(vec_int_pos,"0")
            return MPS(ComplexF64,qubit_site, [vec_int_pos[ram_phy[i]] for i in 1:(L+ancilla)])
        else
            # Create two orthogonal MPS and couple them to |0> and |1> ancilla, respectively
            # The physical qubit of ancilla is always the last one
            # The ram position, depending on the folded or not, is either the first (folded) or the last (unfolded)
            
            mps1 = randomMPS(rng_vec, ComplexF64, qubit_site[sort(phy_ram[phy_list])], linkdims=_maxdim0)
            mps0 = randomMPS(rng_vec, ComplexF64, qubit_site[sort(phy_ram[phy_list])], linkdims=_maxdim0)
            mps1 -= inner(mps0, mps1) .* mps0
            mps1 = mps1 ./ norm(mps1)
            anc0 = MPS(qubit_site[[(phy_ram[L+1])]], ["0"])
            anc1 = MPS(qubit_site[[(phy_ram[L+1])]], ["1"])
            if folded
                return add(attach_mps(anc0,mps0) , attach_mps(anc1,mps1),cutoff=_eps)/sqrt(2)
            else 
                return (attach_mps(mps0,anc0) + attach_mps(mps1,anc1))/sqrt(2)
            end
        end
    end
end

"""
apply the operator `op` to `mps`, the `op` should have indices of (i,j,k.. i',j',k')
the orthogonalization center is at i
index should be ram index
"""

function find_matching_site_indices(tensor1, tensor2)
    matches = []
    
    for ind1 in inds(tensor1)
        if hastags(ind1, "Site")  # Check if it's a Site index
            for ind2 in inds(tensor2)
                if hastags(ind2, "Site") && tags(ind1) == tags(ind2)
                    # Found matching Site indices with identical tags
                    push!(matches, (ind1, ind2))
                    # println("Match found: $(tags(ind1)) - IDs: $(id(ind1)) <-> $(id(ind2))")
                end
            end
        end
    end
    
    return matches
end

function apply_op!(mps::MPS, op::ITensor, eps::Float64, maxdim::Int)
    i_list = [parse(Int, replace(string(tags(inds(op)[i])[length(tags(inds(op)[i]))]), "n=" => "")) for i in 1:div(length(op.tensor.inds), 2)]
    sort!(i_list)
    orthogonalize!(mps, i_list[1])
    mps_ij = mps[i_list[1]]
    for idx in i_list[1]+1:i_list[end]
        mps_ij *= mps[idx]
    end

    matches = find_matching_site_indices(op, mps_ij)
    for match in matches
        if plev(match[1]) == 0
            op = replaceinds(op, [match[1]], [match[2]])
        else
            op = replaceinds(op, [match[1]], [prime(match[2])])
        end
    end
    mps_ij *= op  
    noprime!(mps_ij)
    if length(i_list) == 1
        mps[i_list[1]] = mps_ij
    else
        lefttags= (i_list[1]==1) ? nothing : tags(linkind(mps,i_list[1]-1))
        for idx in i_list[1]:i_list[end]-1
            inds1 = (idx ==1) ? [siteind(mps,1)] : [findindex(mps[idx-1],lefttags), findindex(mps[idx],"Site")]
            lefttags=tags(linkind(mps,idx))
            U, S, V = svd(mps_ij, inds1, cutoff=eps,lefttags=lefttags,maxdim=maxdim)
            mps[idx] = U
            mps_ij = S * V
        end
        mps[i_list[end]] = mps_ij
    end
    return
end

""" apply Bernoulli_map to physical site i
"""
function Bernoulli_map!(ct::CT_MPS, i::Int)
    S!(ct, i, ct.rng_C)
end

""" apply scrambler (Haar random unitary) to site (i,i+1) [physical index]
"""
function S!(ct::CT_MPS, i::Int, rng::Union{Nothing, Int, Random.AbstractRNG}; theta=nothing)
    # println("theta = ", theta)
    if ct.simplified_U
        # print("simplified U!!")
        if i==ct.L
            U_4_mat = U_simp(false, nothing, theta)
            println("i == L, U_4_mat = ", U_4_mat)

        else
            U_4_mat = U_simp(true, nothing, theta)
            println("i != L, U_4_mat = ", U_4_mat)
        end
    else
        U_4_mat=U(4,rng)
    end
    U_4 = reshape(U_4_mat, 2, 2, 2, 2)
    if ct.ancilla == 0 || ct.ancilla ==1
        ram_idx = ct.phy_ram[[ct.phy_list[i], ct.phy_list[(i)%(ct.L)+1]]]
        U_4_tensor = ITensor(U_4, ct.qubit_site[ram_idx[1]], ct.qubit_site[ram_idx[2]], ct.qubit_site[ram_idx[1]]', ct.qubit_site[ram_idx[2]]')
        # U_4_tensor = ITensor(U_4, ct_f.qubit_site[ram_idx[1]], ct_f.qubit_site[ram_idx[2]], ct_f.qubit_site[ram_idx[1]]', ct_f.qubit_site[ram_idx[2]]')

        
        if ct.builtin
            ct.mps=apply(U_4_tensor,ct.mps,[ram_idx[1], ram_idx[2]];cutoff=ct._eps,maxdim=ct._maxdim)
        else
            apply_op!(ct.mps, U_4_tensor,ct._eps,ct._maxdim)
        end

        if ct.debug
            println("U $(U_4_mat) apply to $(i)")
        end
    elseif ct.ancilla ==2
        nothing
    end
end

function S_fixed!(ct::CT_MPS, i::Int, U_4_mat::Matrix{ComplexF64})
    # println("U_4_mat: ", U_4_mat)
    U_4 = reshape(U_4_mat, 2, 2, 2, 2)
    if ct.ancilla == 0 || ct.ancilla ==1
        ram_idx = ct.phy_ram[[ct.phy_list[i], ct.phy_list[(i)%(ct.L)+1]]]
        U_4_tensor = ITensor(U_4, ct.qubit_site[ram_idx[1]], ct.qubit_site[ram_idx[2]], ct.qubit_site[ram_idx[1]]', ct.qubit_site[ram_idx[2]]')
        
        if ct.builtin
            ct.mps=apply(U_4_tensor,ct.mps,[ram_idx[1], ram_idx[2]];cutoff=ct._eps,maxdim=ct._maxdim)
        else
            apply_op!(ct.mps, U_4_tensor,ct._eps,ct._maxdim)
        end

        if ct.debug
            println("U $(U_4_mat) apply to $(i)")
        end
    elseif ct.ancilla ==2
        nothing
    end
end

""" apply control_map to physical sites i, i[1] happens to be the leading bit
"""
function control_map(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    R!(ct, n, i)
    if ct.xj == Set([1 // 3, 2 // 3])
        ct.mps=apply(ct.adder[i[1]],ct.mps;cutoff=ct._eps,maxdim=ct._maxdim)
        normalize!(ct.mps)
        truncate!(ct.mps, cutoff=ct._eps,maxdim=ct._maxdim)
    end
end

"""Reset: apply Projection to physical site list i with outcome list n, then reset to 1
"""
function R!(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    P!(ct, n, i)
    if ct.xj in Set([Set([1 // 3, 2 // 3]),Set([0])])
        # println("Resetting $(i) with outcome $(n)")
        if n[1]==1
            X!(ct,i[1])
            # print("X")
        end
    elseif ct.xj in [Set([1 // 3, -1 // 3])]
        if n[1]^n[2]==0
            print("i[end] needs to be checked")
            X!(ct,i[end])
        end
    end
end
"""generate projector for physical site (will convert to RAM site)
"""
function projector(ct::CT_MPS,n::Vector{Int}, i::Vector{Int})
    if ct.debug
        println("Get projector $(n) at Phy $(i)")
    end
    proj_op=emptyITensor(ct.qubit_site[ct.phy_ram[ct.phy_list[i]]],ct.qubit_site[ct.phy_ram[ct.phy_list[i]]]')
    idx=n.+1
    proj_op[ idx...,idx... ]=1+0im
    return proj_op
end
""" perform projection for physical site
"""
function P!(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    @assert length(n) == length(i) "length of n $(n) is not equal to length of i $(i)"
    if ct.debug
        println("Projecting $(n) at Phy $(i)")
    end
    proj_op= projector(ct,n, i)
    apply_op!(ct.mps, proj_op, ct._eps, ct._maxdim)
    if ct.debug
        println("norm is $(norm(ct.mps))")
    end
    normalize!(ct.mps)
    truncate!(ct.mps, cutoff=ct._eps)
end
""" perform Sigma_X for physical site
"""
function X!(ct::CT_MPS, i::Int)
    X_op=ITensor([0 1+0im; 1+0im 0],ct.qubit_site[ct.phy_ram[ct.phy_list[i]]],ct.qubit_site[ct.phy_ram[ct.phy_list[i]]]')
    apply_op!(ct.mps, X_op, ct._eps, ct._maxdim)
    # normalize!(ct.mps)
end

"""compute the 2nd Renyi entropy from definition  by tracing out the `region` (physical sites). """
function Renyi_entropy(ct::CT_MPS,  region::Vector{Int})
    # construct the density matrix from mps
    rho = get_DM(ct.mps)
    ram_idx=ct.phy_ram[region]
    # tracing the other par of region
    partial_trace!(rho, ram_idx)
    # compute rho^2
    rho2=rho_square(rho)
    # take the trace of rho2
    return -real(log(trace(rho2)))
end

"""compute mutual information based on 2nd Renyi entropy for region1 and region2 (physical sites)"""
function mutual_information(ct::CT_MPS,region1::Vector{Int},region2::Vector{Int})
    @assert isempty(intersect(region1, region2)) "Region 1 and 2 have overlapping elements!"

    return Renyi_entropy(ct,region1)+Renyi_entropy(ct,region2)-Renyi_entropy(ct,vcat(region1,region2))
end

"""compute the mutual information from definition"""
function mps_to_tensor(mps; array=false, vector=false, column_first=true)
    # psi = mps[1]
    # for i = 2:length(mps)
    #     psi = psi * mps[i]
    # end

    psi=prod(mps)
    # Convert the resulting tensor to a dense array
    if array
        psi = array(psi)
        if vector
            if column_first
                return vec(psi)
            else
                return vec(permutedims(psi, reverse(1:ndims(psi))))
            end
        else
            return psi
        end
    else
        return psi
    end
end

# function encoding!(ct::CT_MPS)
#     i=1
#     for idx in 1:2*ct.L^2
#         Bernoulli_map!(ct, i)
#         # println(i)
#         i=mod(((i+1) - 1),ct.L+ct.ancilla )+ 1
#     end
# end

function sv_check(mps::MPS, eps::Float64, L::Int)
    mps_ = orthogonalize(copy(mps), div(L,2))
    _, S = svd(mps_[div(L,2)], (linkind(mps_, div(L,2)),); cutoff=eps)
    return array(diag(S))
end

"""randomly apply control or Bernoulli map to physical site i (the left leg of op)
"""
function random_control!(ct::CT_MPS, i::Int, p_ctrl::Float64, p_proj::Float64)
    op_l=[]
    # sv_check_dict = Dict{String, Any}()
    p_0=-1.  # -1 for not applicable because of Bernoulli map
    if rand(ct.rng_C) < p_ctrl
        # control map
        if ct.xj in Set([Set([1 // 3, 2 // 3]),Set([0])])
            p_0= inner_prob(ct, [0], [i])
            if ct.debug
                println("Born prob for measuring 0 at phy site $i is $p_0")
            end
            n =  rand(ct.rng_m) < p_0 ?  0 : 1
            control_map(ct, [n], [i])
            # sv_check_dict = Dict("Type"=>"Control","sv"=>sv_check(ct.mps, ct._eps, ct.L))
            # println(Dict("Type"=>"Control","sv"=>sv_check_dict["Control"]))
            push!(op_l,Dict("Type"=>"Control","Site"=>[i],"Outcome"=>[n]))
        elseif ct.xj in [Set([1 // 3, -1 // 3])]
            # p_00= ...
            # p_01= ...
            # p_10= ...
            # p_11= ...
            # n = ...
            # control_map(ct, n, [i, i+1])
            nothing
        end
        if ct.debug
            print("Control with $(i)")
        end

        i=mod(((i-1) - 1),(ct.L)) + 1
        if ct.debug
            println("=> Next i is $(i)")
        end
    else
        # chaotic map
        Bernoulli_map!(ct, i)
        # sv_check_dict = Dict("Type"=>"Bernoulli","sv"=>sv_check(ct.mps, ct._eps, ct.L))
        # println(Dict("Type"=>"Bernoulli","sv"=>sv_check_dict["Bernoulli"]))
        push!(op_l,Dict("Type"=>"Bernoulli","Site"=>[i,((i+1) - 1)%(ct.L) + 1],"Outcome"=>nothing))
        i=mod(((i+1) - 1),(ct.L) )+ 1
    end

    if op_l[end]["Type"] == "Bernoulli"
        # projection
        for pos in [i-1,i]
            if rand(ct.rng_C) < p_proj
                pos=mod((pos-1),ct.L)+1
                p2=inner_prob(ct, [0], [pos])
                n= rand(ct.rng_m) < p2 ? 0 : 1
                P!(ct,[n],[pos])
                # sv_check_dict = Dict("Type"=>"Projection","sv"=>sv_check(ct.mps, ct._eps, ct.L))
                # println(Dict("Type"=>"Projection","sv"=>sv_check_dict["Projection"]))
                push!(op_l,Dict("Type"=>"Projection","Site"=>[pos],"Outcome"=>[n]))
            end
        end
    end
    update_history(ct,op_l,p_0)
    # println(varinfo())
    return i #sv_check_dict
end

function update_history(ct::CT_MPS,op::Vector{Any},p_0::Float64)
    if ct.store_vec
        push!(ct.vec_history,copy(ct.mps)) 
    end
    if ct.store_op
        push!(ct.op_history,op)
    end
    if ct.store_prob
        push!(ct.op_history,p_0)
    end
end

function random_control_fixed_circuit!(ct::CT_MPS, i::Int, cir::Vector{Any})
    """
    circuit is a vector of vectors, each vector contains the type of operation, the site, the outcome, and the rng_m
    for control map: cir = ["C"]
    for projection map: cir = ["P", outcome, rng_m]
    for Bernoulli map: cir = ["B", theta]
    """
    op_l=[]
    p_0=-1.  # -1 for not applicable because of Bernoulli map
    # control map
    if cir[1] == "C"
        if ct.xj in Set([Set([1 // 3, 2 // 3]),Set([0])])
            p_0= inner_prob(ct, [0], [i])
            if ct.debug
                println("Born prob for measuring 0 at phy site $i is $p_0")
            end
            n =  cir[2] < p_0 ?  0 : 1
            control_map(ct, [n], [i])
            push!(op_l,Dict("Type"=>"Control","Site"=>[i],"Outcome"=>[n]))
        elseif ct.xj in [Set([1 // 3, -1 // 3])]
            # p_00= ...
            # p_01= ...
            # p_10= ...
            # p_11= ...
            # n = ...
            # control_map(ct, n, [i, i+1])
            nothing
        end
        if ct.debug
            print("Control with $(i)")
        end
        i=mod(((i-1) - 1),(ct.L)) + 1
        if ct.debug
            println("=> Next i is $(i)")
        end
    elseif cir[1] == "P"
        for pos in [i-1,i]
            pos=mod((pos-1),ct.L)+1
            p2=inner_prob(ct, [cir[2]], [pos])
            n= cir[3] < p2 ? 0 : 1
            P!(ct,[n],[pos])
        end
        # P!(ct, [cir[2]], [i])
        push!(op_l,Dict("Type"=>"Projection","Site"=>[i],"Outcome"=>[cir[2]],"rng_m"=>cir[3]))
        i=mod(((i-1) - 1),(ct.L)) + 1
        if ct.debug
            println("=> Next i is $(i)")
        end
    elseif cir[1] == "B"
        # bernoulli map
        # println(cir[2:end])
        i=mod(((i+1) - 1),(ct.L) )+ 1
        S_fixed!(ct, i, cir[2])
        push!(op_l,Dict("Type"=>"Bernoulli","Site"=>[i,((i+1) - 1)%(ct.L) + 1],"Outcome"=>nothing))
    else    
        error("Unknown operation $(cir[1]) in circuit")
    end
    update_history(ct,op_l,p_0)
    return i
end


""" compute the Born probability through the inner product at physical site list i (will convert to RAM site internally)
"""
function inner_prob(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    @assert length(n) == length(i) "length of n $(n) is not equal to length of i $(i)"
    ram_idx=ct.phy_ram[ct.phy_list[i]]
    if length(i)==1
        if ct.debug
            println("Get projector for inner_prob $(n) at Phy $(i) at RAM $(ram_idx)")
        end
        proj_op= array(projector(ct,n, i))
        return only(expect(ct.mps, proj_op,sites=ram_idx))
    else
        # proj_op= projector(ct,n, i,tensor=false)
        # os=OpSum()
        error("Not implemented yet")
    end

end

function order_parameter(ct::CT_MPS)
    if ct.xj in Set([Set([1 // 3, 2 // 3])])
        O = ZZ(ct)
        return (O)
    elseif ct.xj in [Set([0])]
        O = Z(ct)
        return (O)
    end
end

"""return each Z_i at physical site i, <Z_i>, here no need to have the second moment because <Z_i^2> = 1"""
function Zi(ct::CT_MPS)
    sZ = expect(ct.mps, "Sz")[ct.phy_ram][ct.phy_list]
    return sZ*2
end

"""note that here ZiZj is not put to the correct position because it would be taken a sum in the end anyway""" 
function ZiZj(ct::CT_MPS)
    sZZ=correlation_matrix(ct.mps, "Sz", "Sz")
    return sZZ*4
end

raw""" return the sum of Z_i, i.e., <O>, where O=1/L \sum_i Z_i"""
function Z(ct::CT_MPS)
    sZ = Zi(ct)
    if ct.debug
        println("Z are $(2 * sZ)")
    end
    return real(sum(sZ)) / ct.L
end

raw""" return the square sum of Z_i, i.e., <O^2>, where O=1/L \sum_i Z_i. Thus, <O^2> = 1/L^2<\sum_{ij}Z_iZ_j>"""
function Z_sq(ct::CT_MPS)
    sZZ = ZiZj(ct)
    return real(sum(sZZ)) / ct.L^2
end

function ZZ(ct::CT_MPS)
    os = OpSum()
    for i in ct.phy_list[1:end-1]
        os += "Sz", ct.phy_ram[i], "Sz", ct.phy_ram[i+1]
    end
    os += "Sz", ct.phy_ram[ct.phy_list[end]], "Sz", ct.phy_ram[ct.phy_list[1]]
    zz = MPO(os, ct.qubit_site)
    return real(-inner(ct.mps', zz, ct.mps)) * 4 / ct.L
end

raw"""return the bitstring in the "unfolded" order"""
function bitstring_sample(ct::CT_MPS)
    bitstring = sample!(ct.rng_m,ct.mps)
    return bitstring[ct.phy_ram]
end
function bin2dec(v::Vector{Int})
    return parse(BigInt, join(v), base=2)
end

function Z_bitstring(bitstring::Vector{Int})
    return sum(1 .-2*(bitstring.-1))/length(bitstring)
end

function mps_element(mps::MPS, x::String)
    @assert length(x) == length(mps)
    x = [string(s) for s in x]
    V = ITensor(1.0)
    for i = 1:length(mps)
        V *= (mps[i] * state(siteind(mps, i), x[i]))
    end
    return scalar(V)
end

function display_mps_element(ct::CT_MPS; mps::MPS=ct.mps)
    println(rpad("RAM",ct.L+ct.ancilla), "=>", "Physical")
    vec=zeros(Complex{Float64},2^(ct.L+ct.ancilla))
    for i in 1:2^(ct.L+ct.ancilla)
        bitstring=lpad(string(i-1,base=2),ct.L+ct.ancilla,"0")
        matel=CT.mps_element(mps,bitstring)
        println(bitstring, "=>", bitstring[ct.phy_ram],": ",matel)
        vec[parse(Int,bitstring[ct.phy_ram];base=2)+1]=matel
    end
    return vec
end

function dec2bin(x::Real, L::Int)
    @assert 0 <= x < 1 "$x is not in [0,1)"

    return BigInt(floor(x * (BigInt(1) << L)))
end

"""create Haar random unitary
"""
function U(n::Int, rng::Random.AbstractRNG=MersenneTwister(nothing))
    z = randn(rng, n, n) + randn(rng, n, n) * im
    Q, R = qr(z)
    r_diag = diag(R)
    Lambda = Diagonal(r_diag ./ abs.(r_diag))
    Q *= Lambda
    return Q
end

CZ_mat = [1.0 0.0 0.0 0.0;
0.0 1.0 0.0 0.0;
0.0 0.0 1.0 0.0;
0.0 0.0 0.0 -1.0+0im]

"""create Rx gate"""
function Rx(theta::Float64)
    return [cos(theta / 2) -im*sin(theta / 2); 
             -im*sin(theta / 2) cos(theta / 2)]
end

"""create Rz gate"""
function Rz(theta::Float64)
    return [exp(-im * theta / 2) 0; 
            0 exp(im * theta / 2)]
end

"""create a simplified Haar random unitary.
The unitary is defined as 
---Rx(θ1)---Rz(θ2)---Rx(θ3)---CZ---Rx(θ7)---Rz(θ8)---Rx(θ9)---
                                             |
---Rx(θ4)---Rz(θ5)---Rx(θ6)---CZ---Rx(θ10)---Rz(θ11)---Rx(θ12)---
If `CZ` is true, applied a CZ gate, otherwise, it is skipped.
Here, 12 θ's are independently chosen as a random number in [0,2pi), and Rx and Rz are single qubit rotation gates along the x and z axes, respectively.
For simplicity, we denote θ as θ[1], θ[2], ..., θ[6] on the top qubit, and θ[7], θ[8], ..., θ[12] on the bottom qubit. 
"""
function U_simp(CZ::Bool, rng::Union{Nothing, Int, Random.AbstractRNG}, 
                theta::Vector{Any}=nothing)
    if theta === nothing
        theta = rand(rng, 12) * 2 * pi
    end
    # println("Random θ: ", theta)

    # Layer 1 (Left)
    U1 = kron(Rx(theta[1]), Rx(theta[4]))
    # Layer 2
    U2 = kron(Rz(theta[2]), Rz(theta[5]))
    # Layer 3
    U3 = kron(Rx(theta[3]), Rx(theta[6]))
    # Layer 4 (CZ)
    U4 = CZ ? CZ_mat : Matrix{ComplexF64}(I, 4, 4)
    # Layer 5
    U5 = kron(Rx(theta[7]), Rx(theta[10]))
    # Layer 6
    U6 = kron(Rz(theta[8]), Rz(theta[11]))
    # Layer 7 (Right)
    U7 = kron(Rx(theta[9]), Rx(theta[12]))

    # Combine layers (matrix multiplication from right to left)
    U_final = U7 * U6 * U5 * U4 * U3 * U2 * U1

    # Transpose is important to ensure that is consistent with Qiskit's convention (which should also be the state vector)
    return collect(transpose(U_final))
end

"""physically same as U_simp, but use single qubit rotation SU(2) instead of the 
three Euler angles decomposition. [The global phase does not matter here]
---U_11---CZ---U_12
          |
---U_21---CZ---U_22
"""
function U2(CZ::Bool,rng::Random.AbstractRNG=MersenneTwister(nothing))
    U_11=U(2,rng)
    U_12=U(2,rng)
    U_21=U(2,rng)
    U_22=U(2,rng)
    CZ_ = CZ ? CZ_mat : Matrix{ComplexF64}(I, 4, 4)
    return kron(U_11,U_21)*CZ_*kron(U_12,U_22)
end
    
"""attach mps2 to mps1 at the last site. The returned mps is |mps1>|mps2>
The physical index of mps2 should be given 
"""
function attach_mps(mps1::MPS,mps2::MPS,eps::Float64=1e-30;replace_link=true)
    n1=length(mps1)
    n2=length(mps2)
    if replace_link
        advance_link_tags!(mps2,n1)
    end
    orthogonalize!(mps1,n1)
    mps_=mps1[n1]*mps2[1]
    lefttags=replace(string(tags(findinds(inds(mps_),"Link")[1])),r"l=[\d]+"=>"l=$(n1)",'"'=>"")
    U,S,V=svd(mps_,inds(mps1[n1]),cutoff=eps,lefttags=lefttags)
    mps_new=MPS(vcat(siteinds(mps1),siteinds(mps2)))
    for i in 1:(n1+n2)
        if i<n1
            mps_new[i]=mps1[i]
        elseif i==n1
            mps_new[i]=U
        elseif i==n1+1
            mps_new[i]=S*V
        else
            mps_new[i]=mps2[i-n1]
        end
    end
    return mps_new
end

function advance_link_tags!(mps::MPS,n::Int64)
    for i in 1:length(mps)
        links=findinds(mps[i],"Link")
        for link in links
            cur_i=parse(Int,match(r"l=(\d+)",string(tags(link)[length(tags(link))]))[1])
            new_link=replacetags(link,"l=$(cur_i)","l=$(n+cur_i)")
            mps[i]*=delta(link,new_link)
        end
    end
end

function max_bond_dim(mps::MPS)
    max_dim = 0
    for i in 1:length(mps)-1
        dim = commonind(mps[i], mps[i+1])
        max_dim = max(max_dim, space(dim))
    end
    return max_dim
end

function all_bond_dim(mp::Union{MPO,MPS})
    dim_list = []
    for i in 1:length(mp)-1
        dim = commonind(mp[i], mp[i+1])
        push!(dim_list, space(dim))
    end
    return dim_list
end

"""add one, i1 is the leading (physical) qubit"""
function add1(i1::Int,L::Int,phy_ram::Vector{Int},phy_list::Vector{Int})
    A1=OpSum()
    A1+=tuple(([((i==i1) ? "X" : "S+",phy_ram[phy_list[i]]) for i in 1:L]...)...)

    for j in 2:L
        i_list = phy_list[mod.(collect(j:L).+(i1-1).-1,L).+1]
        A1+=tuple(([((i==i_list[1]) ? "S-" : "S+",phy_ram[i]) for i in i_list]...)...)
    end
    return A1
end

"""compute mpo^n
convert n to binary rep, and then square it each step, if the binary bit is 1, add the current mpo to the sum
the space overhead is O(2) forone n,
time complexity is O(L): (L-2)+(L/2)'s multiplication for 1/6, and (L-1)+ (L/2)'s multiplication for 1/3
"""
function power_mpo(mpo::MPO,n_list::Vector{Int})
    # mpo_sum=1
    L=length(mpo)
    n_list=mod.(n_list,2^L)
    mpo_sum=fill(copy(mpo),length(n_list))
    mpo_sum_visted=fill(false,L)
    # print(n_list)
    while sum(n_list)>0
        for (n_idx,n) in enumerate(n_list)
            if n%2==1
                # println(n_idx,':',n," in ",n_list)
                if !mpo_sum_visted[n_idx]
                    mpo_sum_visted[n_idx]=true
                    mpo_sum[n_idx]=mpo
                else
                    mpo_sum[n_idx]=apply(mpo_sum[n_idx],mpo)
                end
            end
        end
        mpo=apply(mpo,mpo)
        n_list.>>=1
    end
    return mpo_sum
end

"""return the MPO for {1/6,1/3}, here site indices are all physical indices"""
function adder_MPO(i1::Int,xj::Set,qubit_site::Vector{Index{Int64}},L::Int,phy_ram::Vector{Int},phy_list::Vector{Int})
    if xj == Set([1 // 3, 2 // 3])
        add1_mpo=MPO(add1(i1,L,phy_ram,phy_list),qubit_site)
        # print(add1_mpo)
        add1_6,add1_3=power_mpo(add1_mpo,[div(2^L,6)+1,div(2^L,3)])
        # println("bond dims of add1_3 iterative: ", all_bond_dim(add1_3))
        # println("bond dims before fix spurs: ", all_bond_dim(add1_3))
        i2=phy_list[mod(i1,L)+1]    # 2
        add_condition=apply(add1_6,P_MPO([phy_ram[i2]],[0],qubit_site)) + apply(add1_3,P_MPO([phy_ram[i2]],[1],qubit_site))
        iLm2=phy_list[mod(i1+L-4,L)+1]  # L-2
        iLm1=phy_list[mod(i1+L-3,L)+1 ]   # L-1
        iL=phy_list[mod(i1+L-2,L)+1] # L i1+(L-1) -> L, (x-1)%L+1
        P2=(P_MPO([phy_ram[i1],phy_ram[iLm2],phy_ram[iL]],[1,0,1],qubit_site)+P_MPO([phy_ram[i1],phy_ram[iLm2],phy_ram[iL]],[0,1,0],qubit_site))    # fix spurs
        XI=XI_MPO([phy_ram[iLm1]],qubit_site)

        fix_spurs = apply(XI,P2) + I_MPO([phy_ram[iLm1]],qubit_site)
        # println(all_bond_dim(apply(fix_spurs,add_condition)))
        return apply(fix_spurs,add_condition)
    else
        return nothing
    end
end

"""return the MPO for a joint projection at sites `pos_list` with outcome `n_list`"""
function P_MPO(pos_list::Vector{Int},n_list::Vector{Int},qubit_site::Vector{Index{Int64}})
    P_sum=OpSum()
    P_sum+=tuple(([("Proj$n",pos) for (pos,n) in zip(pos_list,n_list)]...)...)
    return MPO(P_sum,qubit_site)
end

"""return the oerpator of X-I at sites `pos_list`"""
function XI_MPO(pos_list::Vector{Int},qubit_site::Vector{Index{Int64}})
    X_sum=OpSum()
    X_sum+=tuple(([("X",pos) for pos in pos_list]...)...)
    X_sum+=tuple(([(-1,"I",pos) for pos in pos_list]...)...)
    return MPO(X_sum,qubit_site)
end

function I_MPO(pos_list::Vector{Int},qubit_site::Vector{Index{Int64}})
    I_sum=OpSum()
    I_sum+=tuple(([("I",pos) for pos in pos_list]...)...)
    return MPO(I_sum,qubit_site)
end

function dw_FM(i1::Int,L::Int,phy_ram::Vector{Int},phy_list::Vector{Int},order::Int)
    dw=OpSum()
    for j in 1:L
        i_list = phy_list[mod.(collect(1:j).+(i1-1).-1,L).+1]
        dw+=tuple((L-j+1)^order,([((i==i_list[end]) ? "Proj1" : "Proj0",phy_ram[i]) for i in i_list]...)...)
    end
    return dw
end

function dw_MPO(i1::Int,xj::Set,qubit_site::Vector{Index{Int64}},L::Int,phy_ram::Vector{Int},phy_list::Vector{Int},order::Int)
    if xj == Set([1 // 3, 2 // 3])
        return nothing
    elseif xj == Set([0])
        return MPO(dw_FM(i1,L,phy_ram,phy_list,order),qubit_site)
    end
end

function von_Neumann_entropy(mps::MPS, i::Int, threshold::Float64, eps::Float64; n::Int=1,positivedefinite=false,sv=false)
    # println("from SvN, n=$n, threshold=$threshold")
    mps_ = orthogonalize(mps, i)
    _, S = svd(mps_[i], (linkind(mps_, i),); cutoff=eps)
    if sv
        return array(diag(S))
    end
    if positivedefinite
        p=max.(diag(S),threshold)
    else
        p=max.(diag(S),threshold) .^ 2
    end
    if n==1
        SvN = -sum(p .* log.(p))
    elseif n==0
        SvN = log(length(p))
    else
        SvN = log(sum(p.^n)) / (1 - n)
    end
    return SvN
end

###### generalizing to arbitrary leading bits on folded geometry #########

function fraction_to_binary_shift(numerator::Int, denominator::Int, L::Int)
    total_states = 2^L
    if numerator == 1 && denominator == 6
        shift_amount = div(total_states * numerator, denominator) + 1
    else
        shift_amount = div(total_states * numerator, denominator)
    end

    # Convert to L-bit binary vector
    binary_shift = zeros(Int, L)
    temp = shift_amount
    for i in L:-1:1
        binary_shift[i] = temp % 2
        temp ÷= 2
    end

    return binary_shift, shift_amount
end

function create_identity_tensor_4d(s::Index, s_prime::Index, c_in::Index, c_out::Index)
    id_op = ITensor(c_out, s, s_prime, c_in)
    for outer in 1:2  # first and last index values
        for inner in 1:2  # diagonal of the 2×2 identity
            id_op[c_out => outer, s => inner, s_prime => inner, c_in => outer] = 1.0
        end
    end
    return id_op
end

function create_addition_tensor_with_carry(shift_bit::Int, s::Index, s_prime::Index, c_in::Index, c_out::Index)
    T = ITensor(s, s_prime, c_in, c_out)
    
    # Fill tensor according to binary addition logic
    for s_val in 1:2, c_in_val in 1:2  # ITensor uses 1-indexing
        input_bit = s_val - 1          # Convert to 0/1
        carry_in_bit = c_in_val - 1    # Convert to 0/1
        
        # Binary addition: input + shift + carry_in
        sum = input_bit + shift_bit + carry_in_bit
        output_bit = sum % 2
        carry_out_bit = sum ÷ 2
        
        # Set tensor element (convert back to 1-indexing)
        T[s => s_val, s_prime => output_bit + 1, c_in => c_in_val, c_out => carry_out_bit + 1] = 1.0
    end
    
    return T
end

function lower_prime_level(tensor::ITensor, _tags::String)
    # this function lower the double prime level indices to single prime level
    for index in filterinds(inds(tensor), _tags)
        if plev(index) > 1
            tensor = prime(tensor, -(plev(index)-1), index)
        end
    end
    return tensor
end

function conn_pairs(L::Int)
    # Generate list of right translations of 1:L more efficiently
    # Using array comprehension and avoiding modulo operations where possible

    translations = [circshift(1:L, shift) for shift in 0:L-1]
    conn_pairs_dict = Dict()
    folded_ram_phy_dict = Dict()
    for (i1, translation) in enumerate(translations)
        # println(fold(translation, 0))
        ram_phy, phy_ram = fold(translation, 0, L)
        folded_ram_phy_dict[i1] = ram_phy
        # println(ram_phy)
        pairs = []
        for i in reverse(collect(1:L)[2:end])
            current_pos = findfirst(x -> x == i, ram_phy)
            next_pos = findfirst(x -> x == (i-1)%L, ram_phy)
            pair = (current_pos, next_pos)
            push!(pairs, pair)
            # println(i, " is in position $currest_pos carry into $(i-1) which is in position $next_pos")
        end

        conn_pairs_dict[i1] = pairs
    end
    return conn_pairs_dict, folded_ram_phy_dict
end

function fold(translation, ancilla::Int, L::Int)
    if ancilla ==0
        ram_phy = [i for pairs in zip(translation[1:(L÷2)], reverse(translation[(L÷2+1):L])) for i in pairs]
    elseif ancilla ==1
        ram_phy = vcat(L+1,[i for pairs in zip(translations[1:(L÷2)], reverse(translations[(L÷2+1):L])) for i in pairs])
    elseif ancilla ==2
        error("Not implemented yet")
    end

    # phy_ram[physical] = actual in ram
    phy_ram = fill(0, L+ancilla)
    for (ram, phy) in enumerate(ram_phy)
        phy_ram[phy] = ram
    end

    return ram_phy, phy_ram
end

function initialize_gate_vec(pairings_fixed_i1, shift_bits, qubit_site, L, ram_phy)

    carry_links = [Index(2, "Carry,c=$(ram_pos-1)") for ram_pos in 1:L+1]

    gate_vec = ITensor[]

    for (i, _pair) in enumerate(pairings_fixed_i1)
        if _pair[2] - _pair[1] == 1
            # forward one
            T_tensor = create_addition_tensor_with_carry(shift_bits[ram_phy[_pair[1]]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[_pair[2]])
            push!(gate_vec, T_tensor)
        elseif _pair[2] - _pair[1] == 2
            # next nearest neighbor case, need to create two qubit gates
            mid_pos = (_pair[1] + _pair[2]) ÷ 2
            T_tensor = create_addition_tensor_with_carry(shift_bits[ram_phy[_pair[1]]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[mid_pos])
            id_tensor = create_identity_tensor_4d(qubit_site[mid_pos], prime(qubit_site[mid_pos]), carry_links[mid_pos], carry_links[_pair[2]])
            gate = T_tensor * id_tensor
            push!(gate_vec, gate)
        elseif _pair[2] - _pair[1] == -2
            # next nearest neighbor case, need to create two qubit gates
            mid_pos = (_pair[1] + _pair[2]) ÷ 2
            T_tensor = create_addition_tensor_with_carry(shift_bits[ram_phy[_pair[1]]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[mid_pos])
            id_tensor = create_identity_tensor_4d(qubit_site[mid_pos], prime(qubit_site[mid_pos]), carry_links[mid_pos], carry_links[_pair[2]])
            gate = T_tensor * id_tensor
            push!(gate_vec, gate)
        elseif _pair[2] - _pair[1] == -1
            # backward one
            T_tensor = create_addition_tensor_with_carry(shift_bits[ram_phy[_pair[1]]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[_pair[2]])
            push!(gate_vec, T_tensor)
        else
            # more than next nearest neighbor, not implemented yet
            error("Not implemented yet")
        end
    end

    # locate the lsb link
    lsb_link = setdiff(filterinds(inds(gate_vec[1]), "Carry"), filterinds(inds(gate_vec[2]), "Carry"))
    lsb_tensor = onehot(lsb_link...=>1)

    # include the msb tensor
    msb_pos = pairings_fixed_i1[end][2]
    # pairings[i1][end][1]
    mid_pos = (pairings_fixed_i1[end][1] + pairings_fixed_i1[end][2]) ÷ 2
    msb_link = Index(2, "Carry,msb")
    msb_carry_in = setdiff(filterinds(inds(gate_vec[end]), "Carry"), filterinds(inds(gate_vec[end-1]), "Carry"))
    msb_tensor = ITensor(msb_link)
    msb_tensor[msb_link=>1] = 1.0
    msb_tensor[msb_link=>2] = 1.0

    msb_T_tensor = create_addition_tensor_with_carry(shift_bits[ram_phy[msb_pos]], qubit_site[msb_pos], prime(qubit_site[msb_pos]), msb_carry_in..., msb_link)
    msb_T_tensor = msb_T_tensor * msb_tensor
    push!(gate_vec, msb_T_tensor)

    # set least significant bit tensor boundary condition
    gate_vec[1] = gate_vec[1] * lsb_tensor

    return gate_vec
end

function qubit_site_number(gate; plev = 0)
    qubit_legs = filterinds(inds(gate), tags = "Qubit,Site", plev = plev)
    tag_strs = string.(tags.(qubit_legs))
    numbers = [parse(Int, match(r"n=(\d+)", tag).captures[1]) for tag in tag_strs]
    return numbers
end

function svd_chain(gate::ITensor; eps = 1e-5, maxbonddim = 10)
    mpo_vec = Vector{ITensor}()
    sites = filterinds(inds(gate), tags = "Qubit,Site", plev = 0)
    for i in 1:length(sites) - 1
        site_to_left = sites[i]
        site_number = qubit_site_number(gate, plev = 0)[1]
        # specify the left indices
        link_inds = filterinds(gate, "Link")
        left_indices = Vector{Index{Int64}}([site_to_left, prime(site_to_left)])
        if !isempty(link_inds)
            append!(left_indices, link_inds)
        end

        U, S, V = svd(gate, left_indices, lefttags = "Link,l=$site_number"; cutoff = eps, maxdim = maxbonddim)

        push!(mpo_vec, U)
        gate = S * V
        # println(inds(gate))
    end
    return mpo_vec, gate
end


function find_collapse_pairs(gate_vec_sorted)
    # first collapse the folded gate 
    collapse_pairs = []
    # identify the pairs
    i = 1
    while i <= length(gate_vec_sorted)
        current_group = [i]
        sites = filterinds(inds(gate_vec_sorted[i]), tags = "Qubit,Site", plev = 0)
        
        # Keep checking the next gate for overlap with current group
        j = i + 1
        while j <= length(gate_vec_sorted)
            next_sites = filterinds(inds(gate_vec_sorted[j]), tags = "Qubit,Site", plev = 0)
            # Check if next gate overlaps with any sites in current group
            if intersect(sites, next_sites) != []
                push!(current_group, j)
                # Update sites to include sites from the newly added gate
                sites = union(sites, next_sites)
                j += 1
            else
                break
            end
        end
        
        push!(collapse_pairs, current_group)
        i = j  # Move to the next ungrouped gate
    end
    return collapse_pairs
end

function tower_two_qubit_gates(gate_1::ITensor, gate_2::ITensor)
    overlapping_site_numbers = intersect(qubit_site_number(gate_1), qubit_site_number(gate_2))
    primed_second_gate = gate_2
    for n in overlapping_site_numbers
        primed_second_gate = prime(primed_second_gate, "Qubit,Site,n=$n")
    end
    return lower_prime_level(gate_1 * primed_second_gate, "Qubit,Site") # lower_prime_level only lowers the double primed 
end

function collapse_gate_vec(gate_vec_sorted)
    collapse_pairs = find_collapse_pairs(gate_vec_sorted)
    gate_vec_collapsed = ITensor[]
    for _pair in collapse_pairs
        if length(_pair) == 1
            push!(gate_vec_collapsed, gate_vec_sorted[_pair[1]])
        else
            tmp = tower_two_qubit_gates(gate_vec_sorted[_pair[1]], gate_vec_sorted[_pair[2]])
            push!(gate_vec_collapsed, tmp)
        end
    end
    return gate_vec_collapsed
end

function create_mpo(gate_vec_sorted)
    gate_vec_collapsed = collapse_gate_vec(gate_vec_sorted)
    mpo_vec = ITensor[]
    tmp = gate_vec_collapsed[1]
    for num_gate in 1:length(gate_vec_collapsed) - 1
        # contract the gates
        tmp = tmp * gate_vec_collapsed[num_gate+1]

        # svd
        U_vec, remainder = svd_chain(tmp; eps = 1e-10, maxbonddim = 1000);
        for U in U_vec
            push!(mpo_vec, U)
        end

        # propagate the indices to the remainder
        tmp = remainder
    end

    push!(mpo_vec, tmp);

    return MPO(mpo_vec)
end

function adder_mpo_vec_single_number(i1::Int, L, qubit_site, ancilla, folded, numerator, denominator)
    conn_pairs_dict, folded_ram_phy_dict = conn_pairs(L)
    shift_bits, _ = fraction_to_binary_shift(numerator, denominator, L)
    # qubit_site, _, _, _ = _initialize_basis(L, ancilla, folded)
    ram_phy = folded_ram_phy_dict[i1]
    pairing_fixed_i1 = conn_pairs_dict[i1]

    # initialize the gate vector
    gate_vec = initialize_gate_vec(pairing_fixed_i1, shift_bits, qubit_site, L, ram_phy)

    # sort the gate vector according to the qubit site number
    gate_sites = [qubit_site_number(gate) for gate in gate_vec]
    gate_vec_sorted = gate_vec[sortperm(gate_sites)];

    # create the mpo
    adder_mpo = create_mpo(gate_vec_sorted);
    return adder_mpo
end

function adder_mpo_vec(L::Int64, xj::Set{Rational{Int64}}, qubit_site::Vector{Index{Int64}}, ancilla::Int64, folded::Bool, phy_list::Vector{Int64}, phy_ram::Vector{Int64})
    adder_mpo_vec = MPO[]
    if xj == Set([1 // 3, 2 // 3])
        for i1 in 1:L
            add_1_3 = adder_mpo_vec_single_number(i1, L, qubit_site, ancilla, folded, 1, 3)
            add_1_6 = adder_mpo_vec_single_number(i1, L, qubit_site, ancilla, folded, 1, 6)
            # println("bond dims of add1_3 passthrough: ", all_bond_dim(add_1_3))
            # println("bond dims before fix spurs: ", all_bond_dim(add_1_3), " ", all_bond_dim(add_1_6))
            i2=phy_list[mod(i1,L)+1]    # 2
            add_condition=apply(add_1_6,P_MPO([phy_ram[i2]],[0],qubit_site)) + apply(add_1_3,P_MPO([phy_ram[i2]],[1],qubit_site))
            iLm2=phy_list[mod(i1+L-4,L)+1]  # L-2
            iLm1=phy_list[mod(i1+L-3,L)+1 ]   # L-1
            iL=phy_list[mod(i1+L-2,L)+1] # L i1+(L-1) -> L, (x-1)%L+1
            P2=(P_MPO([phy_ram[i1],phy_ram[iLm2],phy_ram[iL]],[1,0,1],qubit_site)+P_MPO([phy_ram[i1],phy_ram[iLm2],phy_ram[iL]],[0,1,0],qubit_site))    # fix spurs
            XI=XI_MPO([phy_ram[iLm1]],qubit_site)

            fix_spurs = apply(XI,P2) + I_MPO([phy_ram[iLm1]],qubit_site)
            # println("bond dims after fix spurs: ", all_bond_dim(apply(fix_spurs,add_condition)))
            push!(adder_mpo_vec, apply(fix_spurs,add_condition))
        end
    else
        error("Not implemented yet")
        push!(adder_mpo_vec, nothing)
    end
    return adder_mpo_vec
end

greet() = print("Hello World! How are 121?")


end # module CT
