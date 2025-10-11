using ITensors

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