using ITensors

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
        ram_phy, phy_ram = fold(translation, 0)
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

function fold(translation, ancilla::Int)
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

function adder_mpo_vec(L::Int64, xj::Set{Rational{Int64}}, qubit_site::Vector{Index{Int64}}, ancilla::Int64, folded::Bool, phy_list::Vector{Int64}, phy_ram::Vector{Int64}, _maxdim::Int64)
    adder_mpo_vec = MPO[]
    if xj == Set([1 // 3, 2 // 3])
        for i1 in 1:L
            add_1_3 = adder_mpo_vec_single_number(i1, L, qubit_site, ancilla, folded, 1, 3)
            add_1_6 = adder_mpo_vec_single_number(i1, L, qubit_site, ancilla, folded, 1, 6)
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
######### helper functions for testing #########

function all_bond_dim(mp::Union{MPO,MPS})
    dim_list = []
    for i in 1:length(mp)-1
        dim = commonind(mp[i], mp[i+1])
        push!(dim_list, space(dim))
    end
    return dim_list
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

function print_nonzero_coordinates(T)
    # Get the dimensions of the tensor
    dims = size(T)
    
    # Iterate through all possible indices
    for idx in CartesianIndices(dims)
        # Check if element is nonzero
        if T[idx] != 0
            # Convert CartesianIndex to tuple for cleaner output
            coords = Tuple(idx)
            println("Nonzero at coordinates $coords: $(T[idx])")
        end
    end
end


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