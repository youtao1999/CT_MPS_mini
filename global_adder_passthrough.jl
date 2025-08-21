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

function initialize_links(L, qubit_site, shift_1_3_bits, ram_phy)
    carry_links = [Index(2, "Carry,c=$(ram_pos-1)") for ram_pos in 1:L+1]
    T_vec = [ram_pos % 2 == 1 ? create_addition_tensor_with_carry(shift_1_3_bits[ram_phy[ram_pos]], qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos+1], carry_links[ram_pos]) : create_addition_tensor_with_carry(shift_1_3_bits[ram_phy[ram_pos]], qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos], carry_links[ram_pos+1]) for ram_pos in 1:L];
    id_vec = [ram_pos % 2 == 1 ? create_identity_tensor_4d(qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos], carry_links[ram_pos+1]) : create_identity_tensor_4d(qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos], carry_links[ram_pos+1]) for ram_pos in 2:L];
    gate_vec = [T_vec[ram_pos] * id_vec[ram_pos] for ram_pos in 1:L-1];
    return carry_links, T_vec, id_vec, gate_vec
end

function global_adder(L, qubit_site, shift_1_3_bits, ram_phy, initial_state)
    carry_links, T_vec, id_vec, gate_vec = initialize_links(L, qubit_site, shift_1_3_bits, ram_phy);
    for i1 in 1:length(gate_vec)
        gate = gate_vec[i1]
        if i1 <= 2
            # psi_tensor = contract(initial_state[i1], initial_state[i1+1], gate)
            psi_tensor = gate * (initial_state[i1] * initial_state[i1+1])
            left_inds = filter_highest_tag_number(inds(psi_tensor))
            U, S, V = svd(psi_tensor, left_inds, lefttags = "Link,l=$(i1)")
            initial_state[i1] = U
            initial_state[i1+1] = S * V
            noprime!(initial_state)
        else
            # psi_tensor = contract(initial_state[i1-1], initial_state[i1], initial_state[i1+1], gate)
            psi_tensor = gate * (initial_state[i1-1] * initial_state[i1] * initial_state[i1+1])
            all_3_inds = inds(psi_tensor)
            mid_inds = filter(idx -> occursin(string(i1), string(tags(idx))), all_3_inds)
            right_inds = filter(idx -> occursin(string(i1+1), string(tags(idx))), all_3_inds)
            U1, S1, V1 = svd(psi_tensor, setdiff(all_3_inds, right_inds, mid_inds), lefttags = "Link,l=$(i1-1)")
            left_inds_2 = union(mid_inds, filterinds(S1, tags="Link,l=$(i1-1)"))
            U2, S2, V2 = svd(S1 * V1, left_inds_2, lefttags = "Link,l=$(i1)")
            initial_state[i1-1] = U1
            initial_state[i1] = U2
            initial_state[i1+1] = S2 * V2
            noprime!(initial_state)
        end
    end
    
    lsb_tensor = ITensor(filterinds(initial_state[2], tags="Carry,c=1"))
    lsb_tensor[inds(lsb_tensor)[1]=>1] = 1.0
    initial_state[2] = initial_state[2] * lsb_tensor;
    
    # discard the carry out of the msb tensor
    msb_tensor = ITensor(filterinds(initial_state[1], tags="Carry,c=0"))
    msb_tensor[inds(msb_tensor)[1]=>1] = 1.0
    msb_tensor[inds(msb_tensor)[1]=>2] = 1.0
    initial_state[1] = initial_state[1] * msb_tensor;
    
    psi = contract(initial_state[L-1], initial_state[L], T_vec[L])
    U, S, V = svd(psi, setdiff(inds(psi), filterinds(inds(psi), tags="Qubit,Site,n=$L")), lefttags = "Link,l=$(L-1)")
    initial_state[L-1] = U
    initial_state[L] = S * V
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

function sort_indices_by_tag_number(indices)
    # Extract number from tag string
    function extract_tag_number(idx)
        tag_str = string(tags(idx))
        m = match(r"[a-zA-Z]=(\d+)", tag_str)
        return m !== nothing ? parse(Int, m.captures[1]) : 0
    end
    
    return sort(collect(indices), by=extract_tag_number)
end

function filter_highest_tag_number(indices)
    # Extract all tag numbers from indices
    tag_numbers = Int[]
    
    for idx in indices
        tag_str = string(tags(idx))
        # Extract numbers from the tag string using regex
        numbers = [parse(Int, m.match) for m in eachmatch(r"[0-9]+", tag_str)]
        append!(tag_numbers, numbers)
    end
    # println(tag_numbers)
    if isempty(tag_numbers)
        return indices
    end
    
    max_number = maximum(tag_numbers)
    
    # Filter out indices that contain the maximum number in their tag string
    return filter(idx -> !occursin(string(max_number), string(tags(idx))), indices)
end
