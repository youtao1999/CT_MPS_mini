function create_identity_tensor_4d(s::Index, s_prime::Index, c_in::Index, c_out::Index)
    id_op = ITensor(c_out, s, s_prime, c_in)
    for outer in 1:2  # first and last index values
        for inner in 1:2  # diagonal of the 2×2 identity
            id_op[c_out => outer, s => inner, s_prime => inner, c_in => outer] = 1.0
        end
    end
    return id_op
end

function single_site_add_folded!(shift_bit::Int64, ram_pos::Int64, initial_state::MPS)
    """
    This function adds [shift_bit::Int64] (0 or 1) to the ram site [ram_pos::Int64] with the state being [initial_state::MPS] on the folded structure. The "Carry,c=ram_pos" stems from site [ram_pos] and typically ends on site [ram_pos+2] jumping over [ram_pos+1]. This function automatically passes this leg through the middle site [ram_pos+1] by attaching an identity tensor to the gate on [ram_pos+1]. Then performs the two-site contraction before SVD-ing it back into MPS form. 
    """
    qubit_site = siteinds(initial_state)
    T_left_ind = Index(2, "Carry,c=$(ram_pos-1)")
    T_right_ind = Index(2, "Carry,c=$(ram_pos)")
    T = create_addition_tensor_with_carry(shift_bit, qubit_site[ram_pos], prime(qubit_site[ram_pos]), T_right_ind, T_left_ind);
    # println("T", inds(T))
    # println(T.tensor[:,:,1,1])
    id_op_right_ind = Index(2, "Carry,c=$(ram_pos+1)")
    id_op = create_identity_tensor_4d(qubit_site[ram_pos+1], prime(qubit_site[ram_pos+1]), id_op_right_ind, T_right_ind);
    # println("T", inds(T))
    # println("id_op", inds(id_op))
    # println(id_op.tensor[1,:,:,1])
    gate = T * id_op
    # println("gate inds", inds(gate))
    # contract the first two qubit sites out of the three
    # println("upon receiving the gate, ", inds(initial_state[ram_pos]), " and ", inds(initial_state[ram_pos+1]))

    # figure out the left and right indices for the SVD
    left_original_inds = inds(initial_state[ram_pos]) # bookkeep some of the original indices from the left site for SVD
    link_right = filterinds(initial_state[ram_pos+1], tags="Link,l=$(ram_pos)")
    left_original_inds = setdiff(left_original_inds, link_right)
    # println("left_original_inds ", left_original_inds)

    # contract the gate with the two qubit sites
    # println(inds(initial_state[ram_pos]))
    # println(inds(initial_state[ram_pos+1]))
    psi = gate * (initial_state[ram_pos] * initial_state[ram_pos+1])
    noprime!(psi)
    carry_left = filterinds(psi, tags="Carry,c=$(ram_pos-1)")
    # println("carry_left", carry_left)
    link_left = filterinds(psi, tags="Link,l=$(ram_pos-1)")
    # println("link_left ", link_left)
    qubit_left = filterinds(psi, tags="Site,n=$(ram_pos)")
    # println(qubit_left)
    left_inds = unioninds(carry_left, link_left, qubit_left, left_original_inds)
    # println("left_inds ", left_inds)
    # println(typeof(psi))
    U, S, V = svd(psi, left_inds; cutoff=1e-12, lefttags = "Link,l=$(ram_pos)")
    # println(inds(initial_state))
    # new_state = copy(initial_state)
    initial_state[ram_pos] = U
    initial_state[ram_pos+1] = S*V
end

function global_adder_folded(initial_state::MPS, ram_phy::Vector{Int}, shift_bits::Vector{Int})
    L = length(shift_bits)
    for ram_pos in 1:L-2
        shift_bit = shift_bits[ram_phy[ram_pos]]
        println(shift_bit)
        single_site_add_folded!(shift_bit, ram_pos, initial_state)
        println("after adding ", ram_pos, " ", inds(initial_state))
    end
    # check_entanglement_spectrum(initial_state, ram_phy, shift_bits)
    # no carry into the lsb tensor
    lsb_tensor = ITensor(filterinds(initial_state[2], tags="Carry,c=1"))
    lsb_tensor[inds(lsb_tensor)[1]=>1] = 1.0
    # print(inds(lsb_tensor))
    initial_state[2] = initial_state[2] * lsb_tensor;
    # println(inds(initial_state[2]))

    # discard the carry out of the msb tensor
    msb_tensor = ITensor(filterinds(initial_state[1], tags="Carry,c=0"))
    msb_tensor[inds(msb_tensor)[1]=>1] = 1.0
    msb_tensor[inds(msb_tensor)[1]=>2] = 1.0
    initial_state[1] = initial_state[1] * msb_tensor;
    normalize!(initial_state)
    # println(inds(initial_state[1]))
    # orthogonalize!(initial_state, 1)
    # println(initial_state[1])
    # # now take care of the final boundary contraction of 0 and 1
    # c_mid_left = filterinds(initial_state[L-2], tags = "Carry")[1]
    # c_mid_right = filterinds(initial_state[L-1], tags = "Carry")[1]
    # c_mid = Index(2, "Carry,c=$(L)")
    # # print(c_Lm1)
    # # print(c_L)
    # # println(inds(initial_state[L-2]))
    # T_Lm1 = create_addition_tensor_with_carry(1, qubit_site[L-1], prime(qubit_site[L-1]), c_mid_left, c_mid)
    # # println(inds(T_Lm1))
    # T_L = create_addition_tensor_with_carry(0, qubit_site[L], prime(qubit_site[L]), c_mid, c_mid_right)

    # initial_state[L-2] = initial_state[L-2] * T_Lm1
    # # println(inds(initial_state[L-2]))
    # tmp_combined_tensor = initial_state[L-2] * initial_state[L-1]
    # # println(inds(tmp_combined_tensor))

    # # svd the combined tensor
    # left_inds = union(
    #     filterinds(tmp_combined_tensor, tags="Link,l=$(L-3)"),
    #     filterinds(tmp_combined_tensor, tags="Qubit,n=$(L-2)")
    # )
    # # println(left_inds)
    # U, S, V = svd(tmp_combined_tensor, left_inds; cutoff=1e-12, lefttags = "Link,l=$(L-2)")
    # initial_state[L-2] = U
    # initial_state[L-1] = S*V

    # # act the last addition tensor
    # tmp_combined_tensor = initial_state[L-1] * initial_state[L]
    # tmp_combined_tensor = tmp_combined_tensor * T_L
    # left_inds = union(
    #     filterinds(tmp_combined_tensor, tags="Link,l=$(L-2)"),
    #     filterinds(tmp_combined_tensor, tags="Qubit,n=$(L-1)")
    # )
    # U, S, V = svd(tmp_combined_tensor, left_inds; cutoff=1e-12, lefttags = "Link,l=$(L-1)")
    # initial_state[L-1] = U
    # initial_state[L] = S*V
    # noprime!(initial_state)
    # orthogonalize!(initial_state, L)
    # check_entanglement_spectrum(initial_state, ram_phy, shift_bits)
    return initial_state
end

function entanglement_spectrum(psi::MPS, cut_position::Int)
    # Make a copy and move orthogonality center
    psi_copy = copy(psi)
    orthogonalize!(psi_copy, cut_position)
    
    # Get the bond connecting cut_position and cut_position+1
    bond_idx = commonind(psi_copy[cut_position], psi_copy[cut_position+1])
    # println(bond_idx)
    # println(psi_copy[cut_position])
    # SVD the tensor at cut_position
    U, S, V = svd(psi_copy[cut_position], bond_idx)
    
    return diag(S)
end

function check_entanglement_spectrum(initial_state::MPS, ram_phy::Vector{Int}, shift_bits::Vector{Int})
    L = length(shift_bits)
    for cut in 1:L-1
        spectrum = entanglement_spectrum(initial_state, cut)
        println("Cut between sites $cut and $(cut+1): ", spectrum)
    end
end