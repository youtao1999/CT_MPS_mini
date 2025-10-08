using Pkg
Pkg.activate("CT")
using CT
include("global_adder_passthrough.jl")

# benchmark adder_MPO
using ITensors
# initialize random state
using .CT: _initialize_basis, _initialize_vector, P_MPO, XI_MPO, I_MPO, adder_MPO, add1, power_mpo
using Random

L = 12
ancilla = 0
folded = true
xj = Set([1//3, 2//3])
i1 = 1
_maxdim = 2^(div(L,2))
_maxdim0 = 50
_eps = 0.0
seed = 123457
x0 = nothing
qubit_site, ram_phy, phy_ram, phy_list = _initialize_basis(L, ancilla, folded)
rng = MersenneTwister(seed_vec)
rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
shift_1_3_bits, shift_1_3_amount = fraction_to_binary_shift(1, 3, L)
shift_1_6_bits, shift_1_6_amount = fraction_to_binary_shift(1, 6, L)
# initial_state = _initialize_vector(L, ancilla, x0, folded, qubit_site, ram_phy, phy_ram, phy_list, rng_vec, _eps, _maxdim0);

# initialize haining adder mpo
add1_mpo=MPO(add1(i1,L,phy_ram,phy_list),qubit_site)
add1_6,add1_3=power_mpo(add1_mpo,[div(2^L,6)+1,div(2^L,3)])

function display_state(state)
    contracted = contract(state)
    state_vec = vec(Array(contracted, inds(contracted)...))
    # @show argmax(state_vec)
    loc = findall(x->abs(x)>1e-1, state_vec)
    # println(state_vec[loc])
    if length(loc) > 0
        pos = loc[1] - 1
        # ITensors native ordering: site 1 = rightmost bit in binary position
        # We reversed input, so reverse output to match our convention
        binary = string(pos, base=2, pad=L)
        return reverse(binary)  # site 1 = leftmost bit in our convention
    else
        return "0"^L
    end
end
# create_addition_tensor_with_carry(shift_bit::Int, s::Index, s_prime::Index, c_in::Index, c_out::Index)
carry_links, T_vec, id_vec, gate_vec = initialize_links(L, qubit_site, shift_1_3_bits, ram_phy);
adder_passthrough = build_adder_mpo(qubit_site,L,carry_links,gate_vec,_eps,_maxdim);

# prod(gate_vec)

dict_correct = Dict{String, String}()

for state_index in 1:2^L
    string_vec = lpad(string(state_index-1,base=2),L,"0")
    correct_unfolded = lpad(string((state_index - 1 + shift_1_3_amount) % 2^L,base=2),L,"0")
    string_vec_folded = join([string_vec[ram_phy[i]] for i in 1:L])
    correct_folded = join([correct_unfolded[ram_phy[i]] for i in 1:L])
    dict_correct[join(string_vec)] = join(correct_folded)
end


for state_index in 1:2^L
    string_vec = lpad(string(state_index-1,base=2),L,"0")
    string_vec_folded = join([string_vec[ram_phy[i]] for i in 1:L])
    # Convention: RAM position i stores physical bit ram_phy[i]
    # productMPS assigns to RAM positions, so use folded representation
    vec = 1 .+ parse.(Int, [string(string_vec_folded[i]) for i in 1:L])
    initial_state = productMPS(qubit_site, vec)

    # @show CT.mps_element(initial_state, "000000001000"[ram_phy])
    final_state_1 = copy(initial_state)
    final_state_2 = copy(initial_state)
    passthrough_state = copy(initial_state)
    final_state_1 = apply(adder_passthrough,final_state_1; cutoff=_eps, maxdim=_maxdim);
    passthrough_state = global_adder(passthrough_state, carry_links, T_vec, gate_vec, qubit_site; cutoff=_eps, maxdim=_maxdim);
    final_state_2 = apply(add1_3,final_state_2; cutoff=_eps, maxdim=_maxdim);
    # @show CT.mps_element(final_state_2, "001010110011"[ram_phy])
    # @show display_state(final_state_2)[phy_ram]
    final_state_1_unfolded = [ display_state(final_state_1)[phy_ram[i]] for i in 1:L ]
    final_state_2_unfolded = [ display_state(final_state_2)[phy_ram[i]] for i in 1:L ]
    passthrough_state_unfolded = [ display_state(passthrough_state)[phy_ram[i]] for i in 1:L ]
    # println(string_vec, "=>", join(final_state_1_unfolded))
    # println(string_vec, "=>", join(final_state_2_unfolded))
    if join(final_state_1_unfolded) != join(final_state_2_unfolded)
        println(string_vec)
        println(join(final_state_1_unfolded))
        println(join(final_state_2_unfolded))
        println(join(passthrough_state_unfolded))
        println("wrong")
        println("correct: ", dict_correct[join(string_vec)])

    end

    # println(string_vec, "=>", display_state(final_state_1), "<=>", display_state(final_state_2))
    # println(display_state(final_state_1) == display_state(final_state_2))
end

# # warm up
# initial_state = randomMPS(qubit_site, L)
# final_state_1 = copy(initial_state)
# final_state_2 = copy(initial_state)
# final_state_1 = global_adder(final_state_1, carry_links, T_vec, gate_vec, qubit_site; cutoff=_eps, maxdim=_maxdim);
# final_state_2 = apply(add1_6,final_state_2; cutoff=_eps, maxdim=_maxdim);

# # benchmark
# @time final_state_2 = apply(add1_6,final_state_2; cutoff=_eps, maxdim=_maxdim);

# @time final_state_1 = global_adder(final_state_1, carry_links, T_vec, gate_vec, qubit_site; cutoff=_eps, maxdim=_maxdim);
