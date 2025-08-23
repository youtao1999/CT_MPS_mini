using ITensors
# const NUMBER_REGEX = r"[0-9]+"

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

function initialize_links(L, qubit_site, shift_1_3_bits, ram_phy)
    carry_links = [Index(2, "Carry,c=$(ram_pos-1)") for ram_pos in 1:L+1]
    T_vec = [ram_pos % 2 == 1 ? create_addition_tensor_with_carry(shift_1_3_bits[ram_phy[ram_pos]], qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos+1], carry_links[ram_pos]) : create_addition_tensor_with_carry(shift_1_3_bits[ram_phy[ram_pos]], qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos], carry_links[ram_pos+1]) for ram_pos in 1:L];
    id_vec = [ram_pos % 2 == 1 ? create_identity_tensor_4d(qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos], carry_links[ram_pos+1]) : create_identity_tensor_4d(qubit_site[ram_pos], prime(qubit_site[ram_pos]), carry_links[ram_pos], carry_links[ram_pos+1]) for ram_pos in 2:L];
    gate_vec = [T_vec[ram_pos] * id_vec[ram_pos] for ram_pos in 1:L-1];
    return carry_links, T_vec, id_vec, gate_vec
end

function global_adder(initial_state, carry_links, T_vec, id_vec, gate_vec, qubit_site, shift_1_3_bits, ram_phy; cutoff::Float64=1e-10, maxdim::Int=typemax(Int))
    for i1 in 1:length(gate_vec)
        gate = gate_vec[i1]
        # println(i1)
        if i1 == 1
            # psi_tensor = contract(initial_state[i1], initial_state[i1+1], gate)
            psi_tensor = gate * (initial_state[i1] * initial_state[i1+1])
            # @time left_inds = filter_highest_tag_number(inds(psi_tensor))
            # println("left_inds: ", left_inds)
            # println(prime(qubit_site[i1]), carry_links[i1])
            U, S, V = svd(psi_tensor, [prime(qubit_site[i1]), carry_links[i1]], lefttags = "Link,l=$(i1)", cutoff=cutoff, maxdim=maxdim)
            initial_state[i1] = U
            initial_state[i1+1] = S * V
            noprime!(initial_state)
        elseif i1 == 2
            psi_tensor = gate * (initial_state[i1] * initial_state[i1+1])
            U, S, V = svd(psi_tensor, [prime(qubit_site[i1]), carry_links[i1], inds(initial_state[i1])[1], inds(initial_state[i1])[2]], lefttags = "Link,l=$(i1)", cutoff=cutoff, maxdim=maxdim)
            initial_state[i1] = U
            initial_state[i1+1] = S * V
            noprime!(initial_state)
        else
            # psi_tensor = contract(initial_state[i1-1], initial_state[i1], initial_state[i1+1], gate)
            psi_tensor = gate * (initial_state[i1-1] * initial_state[i1] * initial_state[i1+1])
            all_3_inds = inds(psi_tensor)
            # @time mid_inds = filter(idx -> occursin(string(i1), string(tags(idx))), all_3_inds)
            # println([prime(qubit_site[i1]), carry_links[i1+1]])
            # println("mid_inds: ", mid_inds)
            # @time right_inds = filter(idx -> occursin(string(i1+1), string(tags(idx))), all_3_inds)
            left_inds_1 = i1 == 3 ? [qubit_site[i1-1], carry_links[i1-1], inds(initial_state[i1-1])[3]] : [inds(initial_state[i1-1])[1], qubit_site[i1-1]]
            # println(left_inds_1)
            # println(setdiff(all_3_inds, right_inds, mid_inds))
            # U1, S1, V1 = svd(psi_tensor, setdiff(all_3_inds, right_inds, mid_inds), lefttags = "Link,l=$(i1-1)")
            U1, S1, V1 = svd(psi_tensor, left_inds_1, lefttags = "Link,l=$(i1-1)", cutoff=cutoff, maxdim=maxdim)
            # left_inds_2 = union(mid_inds, filterinds(S1, tags="Link,l=$(i1-1)"))
            # println(left_inds_2)
            # println([prime(qubit_site[i1]), carry_links[i1+1], inds(U1)[end]])
            # U2, S2, V2 = svd(S1 * V1, left_inds_2, lefttags = "Link,l=$(i1)")
            U2, S2, V2 = svd(S1 * V1, [prime(qubit_site[i1]), carry_links[i1+1], inds(U1)[end]], lefttags = "Link,l=$(i1)", cutoff=cutoff, maxdim=maxdim)
            initial_state[i1-1] = U1
            initial_state[i1] = U2
            initial_state[i1+1] = S2 * V2
            noprime!(initial_state)
        end
    end
    
    lsb_tensor = ITensor(carry_links[2])
    # println(inds(lsb_tensor))
    # println(carry_links[2])
    lsb_tensor[carry_links[2]=>1] = 1.0
    initial_state[2] = initial_state[2] * lsb_tensor;
    
    # discard the carry out of the msb tensor
    msb_tensor = ITensor(carry_links[1])
    # println(inds(msb_tensor))
    # println(carry_links[1])
    msb_tensor[carry_links[1]=>1] = 1.0
    msb_tensor[carry_links[1]=>2] = 1.0
    initial_state[1] = initial_state[1] * msb_tensor;
    
    psi = contract(initial_state[L-1], initial_state[L], T_vec[L])
    # println(setdiff(inds(psi), filterinds(inds(psi), tags="Qubit,Site,n=$L")))
    # println([inds(initial_state[L-1])[1], qubit_site[L-1]])
    U, S, V = svd(psi, [inds(initial_state[L-1])[1], qubit_site[L-1]], lefttags = "Link,l=$(L-1)", cutoff=cutoff, maxdim=maxdim)
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

# function filter_highest_tag_number(indices)
#     # Extract all tag numbers from indices
#     tag_numbers = Int[]
#     println("filter_highest_tag_numberasd")
#     for idx in indices
#         tag_str = string(tags(idx))
#         # Extract numbers from the tag string using regex
#         numbers = [parse(Int, m.match) for m in eachmatch(r"[0-9]+", tag_str)]
#         append!(tag_numbers, numbers)
#     end
#     # println(tag_numbers)
#     if isempty(tag_numbers)
#         return indices
#     end
    
#     max_number = maximum(tag_numbers)
    
#     # Filter out indices that contain the maximum number in their tag string
#     return filter(idx -> !occursin(string(max_number), string(tags(idx))), indices)
# end

function filter_highest_tag_number(indices)
    
    # Pre-allocate with reasonable size hint
    begin
        tag_numbers = Vector{Int}()
        sizehint!(tag_numbers, length(indices) * 2)
    end
    
    # Pre-compute tag strings to avoid repeated string() calls
    begin
        tag_strings = Vector{String}(undef, length(indices))
        for (i, idx) in enumerate(indices)
            tag_strings[i] = string(tags(idx))
        end
    end
    
    # Extract numbers from pre-computed strings
    begin
        for tag_str in tag_strings
            for m in eachmatch(NUMBER_REGEX, tag_str)
                push!(tag_numbers, parse(Int, m.match))
            end
        end
    end
    
    begin
        if isempty(tag_numbers)
            return indices
        end
        
        max_number = maximum(tag_numbers)
        max_number_str = string(max_number)  # Convert once
    end
    
    # Use pre-computed tag strings for filtering - return vector always
    begin
        # Always return a vector to avoid tuple conversion overhead
        filtered_indices = eltype(indices)[]
        for (i, idx) in enumerate(indices)
            if !occursin(max_number_str, tag_strings[i])
                push!(filtered_indices, idx)
            end
        end
        return filtered_indices
    end
end