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

function global_adder(initial_state, carry_links, T_vec, gate_vec, qubit_site; cutoff::Float64=0.0, maxdim::Int=2^(div(length(gate_vec),2)))
    for i1 in 1:length(gate_vec)
        gate = gate_vec[i1]
        # println("gate index: ", i1, " ", inds(gate))
        # println(i1)
        if i1 == 1
            # psi_tensor = contract(initial_state[i1], initial_state[i1+1], gate)
            psi_tensor = gate * (initial_state[i1] * initial_state[i1+1])
            # println(inds(psi_tensor))
            # println(Base.summarysize(psi_tensor))
            U, S, V = svd(psi_tensor, [prime(qubit_site[i1]), carry_links[i1]], lefttags = "Link,l=$(i1)", cutoff=cutoff, maxdim=maxdim)
            # println(Base.summarysize(U))
            # println(Base.summarysize(S))
            # println(Base.summarysize(V))
            initial_state[i1] = U
            initial_state[i1+1] = S * V
            noprime!(initial_state)
        elseif i1 == 2
            psi_tensor = gate * (initial_state[i1] * initial_state[i1+1])
            # println(Base.summarysize(psi_tensor))
            U, S, V = svd(psi_tensor, [prime(qubit_site[i1]), carry_links[i1], inds(initial_state[i1])[1], inds(initial_state[i1])[2]], lefttags = "Link,l=$(i1)", cutoff=cutoff, maxdim=maxdim)
            # println(Base.summarysize(U))
            # println(Base.summarysize(S))
            # println(Base.summarysize(V))
            initial_state[i1] = U
            initial_state[i1+1] = S * V
            noprime!(initial_state)
        else
            # psi_tensor = contract(initial_state[i1-1], initial_state[i1], initial_state[i1+1], gate)
            psi_tensor = gate * (initial_state[i1-1] * initial_state[i1] * initial_state[i1+1])
            # println(Base.summarysize(psi_tensor))
            left_inds_1 = i1 == 3 ? [qubit_site[i1-1], carry_links[i1-1], inds(initial_state[i1-1])[3]] : [inds(initial_state[i1-1])[1], qubit_site[i1-1]]
            # println(Base.summarysize(left_inds_1))
            U1, S1, V1 = svd(psi_tensor, left_inds_1, lefttags = "Link,l=$(i1-1)", cutoff=cutoff, maxdim=maxdim)
            # println(Base.summarysize(U1))
            # println(Base.summarysize(S1))
            # println(Base.summarysize(V1))
            U2, S2, V2 = svd(S1 * V1, [prime(qubit_site[i1]), carry_links[i1+1], inds(U1)[end]], lefttags = "Link,l=$(i1)", cutoff=cutoff, maxdim=maxdim)
            # println(Base.summarysize(U2))
            # println(Base.summarysize(S2))
            # println(Base.summarysize(V2))
            initial_state[i1-1] = U1
            initial_state[i1] = U2
            initial_state[i1+1] = S2 * V2
            noprime!(initial_state)
        end
    end
    
    # lsb_tensor = ITensor(carry_links[2])
    # lsb_tensor[carry_links[2]=>1] = 1.0
    # initial_state[2] = initial_state[2] * lsb_tensor;

    lsb = onehot(carry_links[2]=>1)
    initial_state[2] = initial_state[2] * lsb;
    
    # discard the carry out of the msb tensor
    msb_tensor = ITensor(carry_links[1])
    msb_tensor[carry_links[1]=>1] = 1.0
    msb_tensor[carry_links[1]=>2] = 1.0
    initial_state[1] = initial_state[1] * msb_tensor;
    
    psi = contract(initial_state[L-1], initial_state[L], T_vec[L])
    # println(Base.summarysize(psi))
    # println(setdiff(inds(psi), filterinds(inds(psi), tags="Qubit,Site,n=$L")))
    # println([inds(initial_state[L-1])[1], qubit_site[L-1]])
    U, S, V = svd(psi, [inds(initial_state[L-1])[1], qubit_site[L-1]], lefttags = "Link,l=$(L-1)", cutoff=cutoff, maxdim=maxdim)
    # println(Base.summarysize(U))
    # println(Base.summarysize(S))
    # println(Base.summarysize(V))
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

function sort_indices(tensor::ITensor, _tags::String)
    indices = filterinds(inds(tensor), _tags)
    sorted_inds = sort(collect(indices), by = x -> begin
        tag_str = string(tags(x))
        m = match(r"c=(\d+)", tag_str)
        parse(Int, m.captures[1])
    end)
    return sorted_inds
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

# set the boundary condition so that we are not carrying dangling indices around
function build_adder_mpo(qubit_site::Vector{Index{Int64}},L::Int,carry_links::Vector{Index{Int64}},gate_vec::Vector{ITensor},_eps::Float64,_maxdim::Int)
    lsb_tensor = onehot(carry_links[2]=>1)

    # discard the carry out of the msb tensor
    msb_tensor = ITensor(carry_links[1])
    msb_tensor[carry_links[1]=>1] = 1.0
    msb_tensor[carry_links[1]=>2] = 1.0

    gate_vec[1] = gate_vec[1] * msb_tensor
    gate_vec[2] = gate_vec[2] * lsb_tensor
    mpo_vec = [ITensor() for _ in 1:L];

    tmp_tensor = gate_vec[1] * prime(gate_vec[2], "Qubit,Site,n=2") * prime(gate_vec[3], "Qubit,Site,n=3")
    tmp_tensor = lower_prime_level(tmp_tensor, "Qubit,Site")

    U, S, V = svd(tmp_tensor, [qubit_site[1], prime(qubit_site[1])], lefttags = "Link,l=1", cutoff=_eps, maxdim=_maxdim);

    mpo_vec[1] = U
    i1 = 4
    while i1 <= length(gate_vec)
        println("i1 = ", i1)
        tmp_tensor = lower_prime_level(S * V * prime(gate_vec[i1], "Qubit,Site,n=$(i1)"), "Qubit,Site")
        U, S, V = svd(tmp_tensor, [qubit_site[i1-2], prime(qubit_site[i1-2]), filterinds(tmp_tensor, tags="Link,l=$(i1-3)")...], lefttags = "Link,l=$(i1-2)", cutoff=_eps, maxdim=_maxdim);
        # println(filter(x -> x >= 1e-10, vec(array(S))))
        mpo_vec[i1-2] = U
        i1 += 1
    end

    U, S, V = svd(S * V, [qubit_site[end-2], prime(qubit_site[end-2]), filterinds(S * V, tags="Link,l=$(L-3)")...], lefttags = "Link,l=$(L-2)", cutoff=_eps, maxdim=_maxdim);
    mpo_vec[L-2] = U
    U, S, V = svd(lower_prime_level(S * V * prime(T_vec[L], "Qubit,Site,n=$(L)"), "Qubit,Site"), [qubit_site[L-1], prime(qubit_site[L-1])], lefttags = "Link,l=$(L-1)", cutoff=_eps, maxdim=_maxdim);
    mpo_vec[L-1] = U
    mpo_vec[L] = S * V

    return MPO(mpo_vec)

end


###### generalizing to arbitrary leading bits on folded geometry #########

function conn_pairs(L::Int)
    # Generate list of right translations of 1:L more efficiently
    # Using array comprehension and avoiding modulo operations where possible

    translations = [circshift(1:L, shift) for shift in 0:L-1]
    conn_pairs_dict = Dict()
    for (i1, translation) in enumerate(translations)
        # println(fold(translation, 0))
        ram_phy, phy_ram = fold(translation, 0)
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
    return conn_pairs_dict
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

function initialize_gate_vec(pairings_fixed_i1, shift_bits, qubit_site, L)

    carry_links = [Index(2, "Carry,c=$(ram_pos-1)") for ram_pos in 1:L+1]

    gate_vec = ITensor[]

    for _pair in pairings_fixed_i1
        if _pair[2] - _pair[1] == 1
            # forward one
            T_tensor = create_addition_tensor_with_carry(shift_bits[_pair[1]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[_pair[2]])
            push!(gate_vec, T_tensor)
        elseif _pair[2] - _pair[1] == 2
            # next nearest neighbor case, need to create two qubit gates
            mid_pos = (_pair[1] + _pair[2]) ÷ 2
            T_tensor = create_addition_tensor_with_carry(shift_bits[_pair[1]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[mid_pos])
            id_tensor = create_identity_tensor_4d(qubit_site[mid_pos], prime(qubit_site[mid_pos]), carry_links[mid_pos], carry_links[_pair[2]])
            gate = T_tensor * id_tensor
            push!(gate_vec, gate)
        elseif _pair[2] - _pair[1] == -2
            # next nearest neighbor case, need to create two qubit gates
            mid_pos = (_pair[1] + _pair[2]) ÷ 2
            T_tensor = create_addition_tensor_with_carry(shift_bits[_pair[1]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[mid_pos])
            id_tensor = create_identity_tensor_4d(qubit_site[mid_pos], prime(qubit_site[mid_pos]), carry_links[mid_pos], carry_links[_pair[2]])
            gate = T_tensor * id_tensor
            push!(gate_vec, gate)
        elseif _pair[2] - _pair[1] == -1
            # backward one
            T_tensor = create_addition_tensor_with_carry(shift_bits[_pair[1]], qubit_site[_pair[1]], prime(qubit_site[_pair[1]]), carry_links[_pair[1]], carry_links[_pair[2]])
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

    msb_T_tensor = create_addition_tensor_with_carry(shift_bits[msb_pos], qubit_site[msb_pos], prime(qubit_site[msb_pos]), msb_carry_in..., msb_link)
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

function create_mpo(gate_vec_collapsed)
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
function adder_mpo_vec(L, ancilla, folded, numerator, denominator)
    adder_mpo_vec = MPO[]
    for i1 in 1:L
        println("constructed the $(i1)-th mpo")
        shift_bits, _ = fraction_to_binary_shift(numerator, denominator, L)
        qubit_site, ram_phy, phy_ram, phy_list = _initialize_basis(L, ancilla, folded)
        pairing_fixed_i1 = conn_pairs(L)[i1]

        # initialize the gate vector
        gate_vec = initialize_gate_vec(pairing_fixed_i1, shift_bits, qubit_site, L)

        # sort the gate vector according to the qubit site number
        gate_sites = [qubit_site_number(gate) for gate in gate_vec]
        gate_vec_sorted = gate_vec[sortperm(gate_sites)];
        gate_vec_collapsed = collapse_gate_vec(gate_vec_sorted);

        # create the mpo
        adder_mpo = create_mpo(gate_vec_collapsed);
        push!(adder_mpo_vec, adder_mpo)
    end
    return adder_mpo_vec
end
