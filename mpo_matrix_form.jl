using ITensors
include("CT/src/test_adder_MPO.jl")
using .CT: _initialize_basis, _initialize_vector, P_MPO, XI_MPO, I_MPO, adder_MPO
using Random

using SparseArrays

function create_msb_ordered_tensor(dense_mat, L)
    """Create tensor with MSB-first bit ordering instead of default reshape"""
    tensor_shape = ntuple(i -> 2, 2*L)
    A_tensor = zeros(Int, tensor_shape)
    
    # Fill tensor manually with MSB-first ordering
    for i in 0:2^L-1
        for j in 0:2^L-1
            if dense_mat[i+1, j+1] != 0
                # Convert i,j to MSB-first bit indices
                i_str = string(i, base=2, pad=L)
                j_str = string(j, base=2, pad=L)
                i_bits = [parse(Int, c) + 1 for c in i_str]  # MSB first, 1-indexed
                j_bits = [parse(Int, c) + 1 for c in j_str]  # MSB first, 1-indexed
                
                A_tensor[i_bits..., j_bits...] = dense_mat[i+1, j+1]
            end
        end
    end
    
    return A_tensor
end

function mpo_adder_1_3(L::Int, qubit_site::Vector{Index{Int64}}, ram_phy::Vector{Int}, phy_ram::Vector{Int}, phy_list::Vector{Int}, folded::Bool)
    """
    This function creates the MPO of the adder_MPO from permutation projection matrix of size 2^L by 2^L.
    L: number of qubits
    qubit_site: site indices
    ram_phy: physical indices for ram
    phy_ram: physical indices for ram
    phy_list: physical indices for list
    folded: whether to fold the adder
    """
    coordinates = []
    shift_str = "01"^L
    shift_num = parse(Int, shift_str, base=2)
    for i in 0:2^L-1
        # println("phys order ", i, "->", (i+5) % (2^L))
        # println("phys order binary ", string(i, base=2, pad=L), "->", string((i+5) % (2^L), base=2, pad=L))
        # println("ram order binary ", string(string(i, base=2, pad=L)[ram_phy]), "->", string(string((i+5) % (2^L), base=2, pad=L)[ram_phy]))
        # println("ram order ", parse(Int, string(string(i, base=2, pad=L)[ram_phy]), base=2), "->", parse(Int, string(string((i+5) % (2^L), base=2, pad=L)[ram_phy]), base=2))
        coordinate = (parse(Int, string(string((i+shift_num) % (2^L), base=2, pad=L)[ram_phy]), base=2)+1, parse(Int, string(string(i, base=2, pad=L)[ram_phy]), base=2)+1)
        push!(coordinates, coordinate)
    end
    # println(coordinates)
    matrix_size = (2^L, 2^L)  # or whatever size you need

    # Method 2: Direct dense matrix construction
    dense_mat = zeros(Int, matrix_size)
    for (i, j) in coordinates
        dense_mat[i, j] = 1
    end
    # println(dense_mat)

    # Reshape 16x16 matrix into 4-qubit tensor with 8 indices
    # Each qubit has dimension 2, so we have (2,2,2,2) ⊗ (2,2,2,2)
    A_tensor = create_msb_ordered_tensor(dense_mat, L)
    # println(A_tensor[1,1,2,1,2,2,2,2])
    # Set small values to zero
    A_tensor[abs.(A_tensor) .< 1e-10] .= 0

    # Create ITensor with both input and output indices
    # Input indices (ket): sites
    # Output indices (bra): sites'
    A_itensor = ITensor(A_tensor, qubit_site'..., qubit_site...)

    # Convert to MPO (Matrix Product Operator)
    cutoff = 1e-12
    maxdim = 100

    # Create MPO from the operator ITensor
    mpo = MPO(A_itensor, qubit_site; cutoff=cutoff, maxdim=maxdim)
    # orthogonalize!(mpo, length(mpo))

    # get bond dimensions
        # Assuming you have an MPO called `mpo`
    bond_dims = []
    for i in 1:length(mpo)-1
        # Get the common index between sites i and i+1
        common_idx = commonind(mpo[i], mpo[i+1])
        push!(bond_dims, dim(common_idx))
    end
    return mpo, bond_dims, dense_mat, A_tensor
end

function identify_operator(matrix::Array{Float64})
    """
    Can be used to identify the operator of a 2x2 matrix.
    matrix: 2x2 matrix
    return: string of the operator, "I", "X", "S^-", "S^+", "0"
    """
    if size(matrix) != (2, 2)
        error("Input matrix must be 2x2")
    end

    coords = findall(x -> abs(x) > 1e-10, matrix)
    # println(coords)

    if coords == CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(2, 2)]
        return "I"
    elseif coords == CartesianIndex{2}[CartesianIndex(2, 1), CartesianIndex(1, 2)]
        return "X"
    elseif coords == CartesianIndex{2}[CartesianIndex(2, 1)]
        return "S^-"
    elseif coords == CartesianIndex{2}[CartesianIndex(1, 2)]
        return "S^+"
    elseif coords == []
        return "0"
    else
        error("Invalid operator")
    end
end

function mpo_matrix_form(mpo::MPO, n::Int)
    mpo_array = Array(mpo[n], inds(mpo[n])...)
    link_indices = filterinds(inds(mpo[n]), "Link")
    # println(link_indices)
    link_dims = dim.(link_indices)
    # println("Link dimensions at site $n: ", link_dims)
    # Initialize a matrix with undefined entries
    result_matrix = Array{Any}(undef, link_dims...)
    if n == 1
        # rank 1 tensor
        for i in 1:link_dims[1]
            # println(mpo_array[:,:,i], typeof(mpo_array[:,:,i]))
            result_matrix[i] = identify_operator(mpo_array[:,:,i])
        end
    elseif n == L
        for i in 1:link_dims[1]
            result_matrix[i] = identify_operator(mpo_array[i,:,:])
        end
    else
        for i in 1:link_dims[1], j in 1:link_dims[2]
            result_matrix[i,j] = identify_operator(mpo_array[i,:,:,j])
        end
    end
    
    return result_matrix
end

# This function is still buggy, need to fix it
# function symmetric_gauge!(M::MPO)
#     N = length(M)
#     # orthogonalize!(M, 1)
#     for j in 1:N-1
#         left_inds = filterinds(M[j]; tags="Site") ∪ [linkind(M, j-1)]
#         # println(left_inds)
#         # println(linkind(M, j-1))
#         U, S, V = svd(M[j], left_inds)
#         sqrt_S = sqrt.(S)
#         M[j] = U * sqrt_S
#         M[j+1] = sqrt_S * V * M[j+1]
#     end
# end