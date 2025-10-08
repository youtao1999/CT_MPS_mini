include("global_adder_passthrough.jl")

using Pkg
Pkg.activate("CT")
using CT

# benchmark adder_MPO
using ITensors
# initialize random state
using .CT: _initialize_basis, _initialize_vector, P_MPO, XI_MPO, I_MPO, adder_MPO, add1, power_mpo
using Random

L = 12
ancilla = 0
folded = true
seed = 123457
xj = Set([1//3, 2//3])
i1 = 1
_maxdim = 2^(div(L,2))
_maxdim0 = 500
_eps = 1e-10
seed_vec = 123457
x0 = nothing
qubit_site, ram_phy, phy_ram, phy_list = _initialize_basis(L, ancilla, folded)
rng = MersenneTwister(seed_vec)
rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
shift_1_3_bits, shift_1_3_amount = fraction_to_binary_shift(1, 3, L)
shift_1_6_bits, shift_1_6_amount = fraction_to_binary_shift(1, 6, L)
initial_state = _initialize_vector(L, ancilla, x0, folded, qubit_site, ram_phy, phy_ram, phy_list, rng_vec, _eps, _maxdim0);

# initialize haining adder mpo
add1_mpo=MPO(add1(i1,L,phy_ram,phy_list),qubit_site)
add1_6,add1_3=power_mpo(add1_mpo,[div(2^L,6)+1,div(2^L,3)])

# create_addition_tensor_with_carry(shift_bit::Int, s::Index, s_prime::Index, c_in::Index, c_out::Index)
carry_links, T_vec, id_vec, gate_vec = initialize_links(L, qubit_site, shift_1_3_bits, ram_phy);

using .CT: all_bond_dim, max_bond_dim, maxlinkdim

mpo_test = build_adder_mpo(qubit_site,L,carry_links,gate_vec,_eps,_maxdim);

# mps1 = apply(mpo_test,initial_state, cutoff=0.0, maxdim=_maxdim);
@time mps1 = apply(mpo_test,initial_state, cutoff=0.0, maxdim=_maxdim);
# mps2 = apply(add1_3,initial_state, cutoff=0.0, maxdim=_maxdim);
@time mps2 = apply(add1_3,initial_state, cutoff=0.0, maxdim=_maxdim);

println(norm(vec(array(contract(mps1))) - vec(array(contract(mps2)))))