"""
Create a single addition tensor for position i with carry bonds
Indices: s (input), s' (output), c_in (carry from right), c_out (carry to left)
"""
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

"""
Create extended MPO for binary addition with carry propagation
"""
function create_binary_addition_mpo(shift_bits::Vector{Int}, qubit_site::Vector{Index{Int64}}, phy_ram::Vector{Int}, phy_list::Vector{Int}, ram_phy::Vector{Int})
    L = length(shift_bits)
    
    # Create all unique carry indices first
    all_carry_labels = unique(vcat(phy_list, phy_list .+ 1))
    carry_indices = Dict(label => Index(2, "carry_phy_$(label)") for label in all_carry_labels)

    # Now use the same index objects
    dict = Dict()
    for i in 1:L
        dict[phy_list[i]] = Dict(
            "in" => carry_indices[phy_list[i] + 1],
            "out" => carry_indices[phy_list[i]]
        )
    end
    # println(dict)
    # # Create carry bonds using PHYSICAL position indexing
    # carry_links = Dict{Int, Index}()
    # for i in 1:L+1
    #     carry_links[i] = Index(2, "carry_phy_$i")
    # end
    
    # Initialize tensor array with correct size
    tensors = Vector{ITensor}(undef, L)
    
    for i in 1:L
        phy_pos = phy_list[i]  # Physical position
        ram_pos = phy_ram[phy_pos]  # RAM position where tensor goes
        
        s = qubit_site[ram_pos]
        s_prime = prime(s)
        # println(s)
        # Use PHYSICAL position indexing for carry bonds
        c_in = dict[ram_phy[ram_pos]]["in"]    # From higher physical position
        # println("in c_in: ", c_in)
        c_out = dict[ram_phy[ram_pos]]["out"]       # To lower physical position
        # println("in c_out: ", c_out)
        T = create_addition_tensor_with_carry(shift_bits[phy_pos], s, s_prime, c_in, c_out)
        tensors[ram_pos] = T  # Place at correct RAM position
    end

    # println(tensors)
    
    # Now boundary tensor matches the LSB tensor
    boundary_tensor = ITensor(dict[L]["in"])  # Physical position L+1
    boundary_tensor[dict[L]["in"] => 1] = 1.0
    lsb_ram_pos = phy_ram[phy_list[L]]  # RAM position of physical LSB
    tensors[lsb_ram_pos] = tensors[lsb_ram_pos] * boundary_tensor

    # MSB (physical position 1) carry_out is traced out
    final_carry_tensor = ITensor(dict[1]["out"])  # The carry bond for MSB output
    final_carry_tensor[dict[1]["out"] => 1] = 1.0  # Trace out both carry states
    final_carry_tensor[dict[1]["out"] => 2] = 1.0
    msb_ram_pos = phy_ram[phy_list[1]]  # RAM position of physical MSB
    tensors[msb_ram_pos] = tensors[msb_ram_pos] * final_carry_tensor
    # println(tensors)
    return MPO(tensors)
end

# function create_binary_addition_mpo_folded(shift_bits::Vector{Int}, qubit_site::Vector{Index{Int64}}, 
#     phy_ram::Vector{Int}, phy_list::Vector{Int})
#     L = length(shift_bits)

#     # Create carry bond indices - we'll connect them based on physical bit order
#     carry_links = Dict{Int, Index}()
#     for i in 1:L+1
#         carry_links[i] = Index(2, "carry_phy_$i")
#     end

#     tensors = Vector{ITensor}(undef, L)

#     # Iterate over physical positions (like add1 does)
#     for phy_pos in 1:L
#         # Find RAM position for this physical position (following add1 pattern)
#         ram_pos = phy_ram[phy_list[phy_pos]]

#         s = qubit_site[ram_pos]
#         s_prime = prime(s)

#         # Carry bonds follow physical bit order:
#         # phy_pos receives carry from phy_pos+1 (if exists)
#         # phy_pos sends carry to phy_pos-1 (if exists)
#         c_in = carry_links[phy_pos + 1]    # From higher physical position (lower significance)
#         c_out = carry_links[phy_pos]       # To lower physical position (higher significance)

#         # Use shift bit for this physical position
#         T = create_addition_tensor_with_carry(shift_bits[phy_pos], s, s_prime, c_in, c_out)
#         tensors[ram_pos] = T  # Place tensor at correct RAM position
#     end

#     # Boundary conditions based on physical bit positions
#     # LSB (physical position L) gets carry_in = 0
#     boundary_tensor = ITensor(carry_links[L+1])
#     boundary_tensor[carry_links[L+1] => 1] = 1.0
#     lsb_ram_pos = phy_ram[phy_list[L]]
#     tensors[lsb_ram_pos] = tensors[lsb_ram_pos] * boundary_tensor

#     # MSB (physical position 1) carry_out is traced out
#     final_carry_tensor = ITensor(carry_links[1])
#     final_carry_tensor[carry_links[1] => 1] = 1.0
#     final_carry_tensor[carry_links[1] => 2] = 1.0
#     msb_ram_pos = phy_ram[phy_list[1]]
#     tensors[msb_ram_pos] = tensors[msb_ram_pos] * final_carry_tensor

#     return MPO(tensors)
# end

"""
Convert a fraction like 1/3 or 1/6 to binary representation for given L
"""
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

"""
Optimized version of adder_MPO using binary addition with carry bonds
"""
function adder_MPO_binary_carry(i1::Int, xj::Set, qubit_site::Vector{Index{Int64}}, L::Int, phy_ram::Vector{Int}, phy_list::Vector{Int})
    if xj == Set([1 // 3, 2 // 3])
        i2 = phy_list[mod(i1, L) + 1]    # Site for conditional logic
        
        # Convert fractions to binary shifts
        shift_1_6_bits, shift_1_6_amount = fraction_to_binary_shift(1, 6, L)
        shift_1_3_bits, shift_1_3_amount = fraction_to_binary_shift(1, 3, L)
        
        println("DEBUG: Adding $(shift_1_6_amount)/$(2^L) and $(shift_1_3_amount)/$(2^L)")
        println("DEBUG: Binary shifts: 1/6 = $shift_1_6_bits, 1/3 = $shift_1_3_bits")
        
        # Create binary addition MPOs
        add_mpo_1_6 = create_binary_addition_mpo(shift_1_6_bits, qubit_site)
        add_mpo_1_3 = create_binary_addition_mpo(shift_1_3_bits, qubit_site)
        
        # Create conditional projections
        P0 = P_MPO([phy_ram[i2]], [0], qubit_site)  # Use 1/6 shift when site i2 = |0⟩
        P1 = P_MPO([phy_ram[i2]], [1], qubit_site)  # Use 1/3 shift when site i2 = |1⟩
        
        # Apply conditionally
        add_condition = apply(add_mpo_1_6, P0) + apply(add_mpo_1_3, P1)
        
        # Apply same fix_spurs as original (unchanged)
        iLm2 = phy_list[mod(i1 + L - 4, L) + 1]  # L-2
        iLm1 = phy_list[mod(i1 + L - 3, L) + 1]  # L-1
        iL = phy_list[mod(i1 + L - 2, L) + 1]    # L
        P2 = (P_MPO([phy_ram[i1], phy_ram[iLm2], phy_ram[iL]], [1, 0, 1], qubit_site) +
              P_MPO([phy_ram[i1], phy_ram[iLm2], phy_ram[iL]], [0, 1, 0], qubit_site))
        XI = XI_MPO([phy_ram[iLm1]], qubit_site)
        
        fix_spurs = apply(XI, P2) + I_MPO([phy_ram[iLm1]], qubit_site)
        return apply(fix_spurs, add_condition)
    else
        return nothing
    end
end

"""
Create a computational basis state |n⟩ as an MPS for given L
"""
function create_computational_basis_mps(n::Int, L::Int, qubit_site::Vector{Index{Int64}})
    @assert 0 <= n < 2^L "n must be in range [0, 2^L-1]"
    
    # Convert n to binary representation
    binary_n = zeros(Int, L)
    temp = n
    for i in L:-1:1
        binary_n[i] = temp % 2
        temp ÷= 2
    end
    
    # Create product state MPS
    tensors = ITensor[]
    for i in 1:L
        tensor = ITensor(qubit_site[i])
        if binary_n[i] == 0
            tensor[qubit_site[i] => 1] = 1.0  # |0⟩ state
        else
            tensor[qubit_site[i] => 2] = 1.0  # |1⟩ state
        end
        push!(tensors, tensor)
    end
    
    return MPS(tensors)
end

"""
Extract the computational basis state number from an MPS
Assumes the MPS is (approximately) a product state
"""
function mps_to_computational_basis(mps::MPS)
    L = length(mps)
    result = 0
    
    for i in 1:L
        # Get the dominant coefficient for this site
        site_tensor = mps[i]
        
        # Contract with |0⟩ and |1⟩ to get amplitudes
        s = inds(site_tensor)[1]  # Get the physical index
        amp_0 = abs(site_tensor[s => 1])
        amp_1 = abs(site_tensor[s => 2])
        
        # Determine which state is dominant
        if amp_1 > amp_0
            result += 2^(L-i)  # Add 2^(position) for bit = 1
        end
    end
    
    return result
end

"""
Extract the computational basis state as a binary string from an MPS
Returns a string like "0111" representing the state |0111⟩
"""
function mps_to_binary_string(mps::MPS)
    L = length(mps)
    binary_string = ""
    
    for i in 1:L
        # Get the dominant coefficient for this site
        site_tensor = mps[i]
        
        # Contract with |0⟩ and |1⟩ to get amplitudes
        s = inds(site_tensor)[1]  # Get the physical index
        amp_0 = abs(site_tensor[s => 1])
        amp_1 = abs(site_tensor[s => 2])
        
        # Determine which state is dominant and append to string
        if amp_1 > amp_0
            binary_string *= "1"
        else
            binary_string *= "0"
        end
    end
    
    return binary_string
end

"""
Extract both decimal and binary representations of MPS state
"""
function mps_to_basis_state(mps::MPS)
    decimal = mps_to_computational_basis(mps)
    binary = mps_to_binary_string(mps)
    return (decimal=decimal, binary=binary)
end
