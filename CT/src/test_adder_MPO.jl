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

"""
Insert an identity tensor between tensors A and B along their shared bond.
Creates a 3-legged identity tensor that splits the original bond.

Args:
    A, B: ITensors that share an index
    shared_index: The Index they share
    new_leg_dim: Dimension of the new leg created by the junction

Returns:
    A_new, I_junction, B_new: Modified tensors with the identity inserted
"""
function insert_identity_junction(A::ITensor, B::ITensor, shared_index::Index; new_leg_dim::Int=1)
    # Get the dimension of the shared index
    bond_dim = dim(shared_index)
    
    # Create new indices to replace the shared one
    i_left = Index(bond_dim, "left_$(tags(shared_index))")
    i_right = Index(bond_dim, "right_$(tags(shared_index))")
    i_new = Index(new_leg_dim, "junction_leg")
    
    # Create the 3-legged identity tensor
    I_junction = ITensor(i_left, i_new, i_right)
    
    # Fill the identity: I[i', k, i''] = δ[i', i''] for all k
    for i in 1:bond_dim
        for k in 1:new_leg_dim
            I_junction[i_left => i, i_new => k, i_right => i] = 1.0
        end
    end
    
    # Replace the shared index in A and B
    A_new = replaceinds(A, shared_index => i_left)
    B_new = replaceinds(B, shared_index => i_right)
    
    return A_new, I_junction, B_new
end

"""
Create folded binary addition MPO using identity junctions for non-adjacent carry bonds
Handles the L=4 folded case where carry bonds need to cross over intermediate qubits
"""
function create_folded_binary_addition_mpo(shift_bits::Vector{Int}, qubit_site::Vector{Index{Int64}}, 
                                         phy_ram::Vector{Int}, phy_list::Vector{Int}, ram_phy::Vector{Int})
    L = length(shift_bits)
    @assert L == 4 "Currently only supports L=4 folded case"
    
    # For L=4 folded: RAM positions [1,2,3,4] have physical qubits [1,4,2,3]
    # Carry flow (physical): 4->3->2->1 
    # Carry flow (RAM): 2->4->3->1
    # Problem bonds: 4->1 (skips 3), need to pass through RAM 3
    # Direct bond: 2->4 (adjacent), 3->1 (after junction insertion)
    
    println("Creating folded binary addition MPO for L=$L")
    println("RAM to Physical mapping: $ram_phy")
    println("Physical to RAM mapping: $phy_ram")
    
    # Create carry indices for the folded structure
    carry_indices = Dict{String, Index}()
    
    # Direct adjacent bonds (dimension 2)
    carry_indices["2_to_4"] = Index(2, "carry_2_to_4")      # RAM 2->4 (phy 4->3)
    carry_indices["3_to_1"] = Index(2, "carry_3_to_1")      # RAM 3->1 (phy 2->1) after junction
    
    # Bond that needs junction (will become dimension 4 = 2 × 2)
    carry_indices["4_to_1"] = Index(4, "carry_4_to_1_through_3")  # RAM 4->1 through RAM 3
    
    # Initialize tensor array
    tensors = Vector{ITensor}(undef, L)
    
    # Create tensors for each RAM position
    for ram_pos in 1:L
        phy_pos = ram_phy[ram_pos]
        s = qubit_site[ram_pos]
        s_prime = prime(s)
        
        println("Creating tensor for RAM $ram_pos (physical $phy_pos)")
        
        if ram_pos == 1  # Physical qubit 1 (MSB)
            # Only receives carry from RAM 3 (via junction)
            c_in = carry_indices["3_to_1"]
            T = create_addition_tensor_with_carry(shift_bits[phy_pos], s, s_prime, c_in, nothing)
            # Trace out carry output (MSB doesn't send carry)
            c_out_dummy = Index(2, "dummy_out_1")
            T_temp = create_addition_tensor_with_carry(shift_bits[phy_pos], s, s_prime, c_in, c_out_dummy)
            boundary = ITensor(c_out_dummy)
            boundary[c_out_dummy => 1] = 1.0
            boundary[c_out_dummy => 2] = 1.0
            tensors[ram_pos] = T_temp * boundary
            
        elseif ram_pos == 2  # Physical qubit 4 (LSB)
            # Only sends carry to RAM 4
            c_out = carry_indices["2_to_4"]
            T = create_addition_tensor_with_carry(shift_bits[phy_pos], s, s_prime, nothing, c_out)
            # No carry input for LSB
            c_in_dummy = Index(2, "dummy_in_2")
            T_temp = create_addition_tensor_with_carry(shift_bits[phy_pos], s, s_prime, c_in_dummy, c_out)
            boundary = ITensor(c_in_dummy)
            boundary[c_in_dummy => 1] = 1.0  # No carry in
            tensors[ram_pos] = T_temp * boundary
            
        elseif ram_pos == 3  # Physical qubit 2
            # This position handles junction: receives direct carry from 4->1 bond and sends to RAM 1
            # The 4D carry bond passes through here with identity operation on qubit
            c_junction = carry_indices["4_to_1"]  # 4D index
            c_out = carry_indices["3_to_1"]       # 2D output to RAM 1
            
            # Create special tensor that does addition AND passes junction carry
            T = create_junction_addition_tensor(shift_bits[phy_pos], s, s_prime, c_junction, c_out)
            tensors[ram_pos] = T
            
        elseif ram_pos == 4  # Physical qubit 3
            # Receives carry from RAM 2 and sends to RAM 1 (via junction through RAM 3)
            c_in = carry_indices["2_to_4"]       # From RAM 2
            c_junction = carry_indices["4_to_1"]  # 4D output to RAM 1 via junction
            
            # Create tensor that sends to junction
            T = create_junction_source_tensor(shift_bits[phy_pos], s, s_prime, c_in, c_junction)
            tensors[ram_pos] = T
        end
    end
    
    return MPO(tensors)
end

"""
Create addition tensor that acts as junction destination (RAM 3 in L=4 folded case)
This tensor performs addition AND acts as an identity junction for the 4D carry bond
"""
function create_junction_addition_tensor(shift_bit::Int, s::Index, s_prime::Index, 
                                       c_junction::Index, c_out::Index)
    # c_junction is 4D: encodes (own_carry_in, junction_carry) × 2 each
    # c_out is 2D: standard carry output to next position
    
    T = ITensor(s, s_prime, c_junction, c_out)
    
    for s_val in 1:2, junction_val in 1:4
        input_bit = s_val - 1
        
        # Decode 4D junction input
        # junction_val ∈ {1,2,3,4} maps to (own_carry, through_carry) ∈ {(0,0), (1,0), (0,1), (1,1)}
        junction_idx = junction_val - 1  # Convert to 0-indexing
        own_carry_in = junction_idx & 1
        through_carry = (junction_idx >> 1) & 1
        
        # Perform addition using only own carry
        sum = input_bit + shift_bit + own_carry_in
        output_bit = sum % 2
        own_carry_out = sum ÷ 2
        
        # Output carry is just the own carry (through_carry goes to junction leg, not output)
        c_out_val = own_carry_out + 1  # Convert back to 1-indexing
        
        T[s => s_val, s_prime => output_bit + 1, c_junction => junction_val, c_out => c_out_val] = 1.0
    end
    
    return T
end

"""
Create addition tensor that acts as junction source (RAM 4 in L=4 folded case)
This tensor performs addition and outputs to a 4D junction bond
"""
function create_junction_source_tensor(shift_bit::Int, s::Index, s_prime::Index, 
                                     c_in::Index, c_junction::Index)
    # c_in is 2D: standard carry input
    # c_junction is 4D: outputs (own_carry_out, junction_carry)
    
    T = ITensor(s, s_prime, c_in, c_junction)
    
    for s_val in 1:2, c_in_val in 1:2
        input_bit = s_val - 1
        carry_in_bit = c_in_val - 1
        
        # Perform addition
        sum = input_bit + shift_bit + carry_in_bit
        output_bit = sum % 2
        carry_out_bit = sum ÷ 2
        
        # For the junction, we send our carry as the "through_carry" component
        # The own_carry component at RAM 3 will come from RAM 3's own addition
        # So we encode as (0, carry_out_bit) in the 4D space
        junction_val = (carry_out_bit << 1) + 1  # +1 for 1-indexing
        
        T[s => s_val, s_prime => output_bit + 1, c_in => c_in_val, c_junction => junction_val] = 1.0
    end
    
    return T
end

"""
Test the folded binary addition MPO with a simple example
"""
function test_folded_adder()
    println("Testing folded adder implementation...")
    
    L = 4
    # Set up folded mapping
    ram_phy = [1, 4, 2, 3]  # RAM 1->phy 1, RAM 2->phy 4, etc.
    phy_ram = [1, 3, 4, 2]  # phy 1->RAM 1, phy 2->RAM 3, etc.
    phy_list = collect(1:L)
    
    # Create qubit sites
    qubit_site = siteinds("Qubit", L)
    
    # Test with simple binary shift: add 1 (binary 0001)
    shift_bits = [0, 0, 0, 1]  # Add 1 in binary
    
    println("Testing addition of 1 (binary $(shift_bits)) in folded configuration")
    println("RAM to Physical mapping: $ram_phy")
    
    # Create the folded adder MPO
    folded_mpo = create_folded_binary_addition_mpo(shift_bits, qubit_site, phy_ram, phy_list, ram_phy)
    
    # Test on computational basis state |0000⟩ -> should give |0001⟩
    test_state = create_computational_basis_mps(0, L, qubit_site)  # |0000⟩
    println("Input state: |0000⟩ (decimal 0)")
    
    # Apply the adder
    result_mps = apply(folded_mpo, test_state)
    result = mps_to_basis_state(result_mps)
    
    println("Output state: |$(result.binary)⟩ (decimal $(result.decimal))")
    println("Expected: |0001⟩ (decimal 1)")
    println("Test $(result.decimal == 1 ? "PASSED" : "FAILED")")
    
    return result.decimal == 1
end
