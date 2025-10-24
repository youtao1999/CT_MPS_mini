using ITensors
using LinearAlgebra
using Printf

# Generate rank-deficient random matrices
function generate_rank_deficient_matrix(d1::Int, d2::Int, rank::Int; 
    sv_range::Tuple{Float64,Float64}=(0.1, 1.0),
    method::Symbol=:svd)
"""
Generate a random rank-deficient matrix.

Parameters:
- d1, d2: dimensions of the matrix (d1 × d2)
- rank: desired rank (must be ≤ min(d1, d2))
- sv_range: range for non-zero singular values (min, max)
- method: :svd (clean zeros) or :approximate (near-zero values)

Returns:
- A: rank-deficient matrix
- true_rank: actual rank
- sv_nonzero: non-zero singular values
"""
n = min(d1, d2)
@assert rank <= n "Rank must be ≤ min(d1, d2)"
@assert rank >= 0 "Rank must be non-negative"
@assert sv_range[1] < sv_range[2] "sv_range must be (min, max) with min < max"

if method == :svd
# Method 1: Exact rank deficiency using SVD construction
# Generate random non-zero singular values
sv_nonzero = sort(rand(rank) .* (sv_range[2] - sv_range[1]) .+ sv_range[1], rev=true)

# Create full singular value vector with zeros
singular_values = vcat(sv_nonzero, zeros(n - rank))

# Random orthogonal matrices
U, _ = qr(randn(d1, n))
V, _ = qr(randn(d2, n))

# Construct rank-deficient matrix
A = U * diagm(singular_values) * V'

elseif method == :approximate
# Method 2: Approximate rank deficiency (small but non-zero values)
sv_nonzero = sort(rand(rank) .* (sv_range[2] - sv_range[1]) .+ sv_range[1], rev=true)

# Create singular values with very small (but non-zero) trailing values
sv_small = rand(n - rank) .* 1e-12
singular_values = vcat(sv_nonzero, sv_small)

# Random orthogonal matrices
U, _ = qr(randn(d1, n))
V, _ = qr(randn(d2, n))

A = U * diagm(singular_values) * V'

else
error("Unknown method: $method. Use :svd or :approximate")
end

# Verify rank
true_rank = LinearAlgebra.rank(A, rtol=1e-10)

return A, true_rank, sv_nonzero
end

function test_rank_determination(dimension::Int, rank::Int, cutoff::Float64)
    for i in 1:10
        A, true_rank, sv_nonzero = generate_rank_deficient_matrix(dimension, dimension, rank);
        sv_total = vcat(sv_nonzero, zeros(dimension - true_rank));
        i = Index(dimension, "i")
        j = Index(dimension, "j")
        A_tensor = ITensor(A, i, j)
        U, S, V = svd(A_tensor, i; cutoff=cutoff);
        s_values = diag(S);
        println("svd rank: ", length(s_values))
        # @show true_rank, length(s_values)
        s_values_padded = vcat(s_values, zeros(dimension - length(s_values)));
        # @show norm(sv_total-s_values_padded)
        if length(s_values) > true_rank
            # println("counted extra rank")
            return false
        elseif length(s_values) == true_rank
            return true
        else
            error("unexpected rank determination")
        end
    end
end

num_tests = 1000
dimension = 10
cutoff = 0.0
accurate_rank_count = 0
for i in 1:num_tests
    global accurate_rank_count
    true_rank = rand(1:dimension-1)
    println("true_rank: ", true_rank)
    if test_rank_determination(dimension, true_rank, cutoff)
        accurate_rank_count += 1
    end
end
# println("accurate_rank_count: ", accurate_rank_count)
println("accuracy: ", accurate_rank_count/num_tests)
#     println(@sprintf("\nCutoff = %.0e:", cutoff))
#     println(@sprintf("  Bond dimension (rank): %d", bond_dim))
#     println("  Singular values kept: ", [@sprintf("%.2e", sv) for sv in s_values])
    
#     if cutoff == 0.0
#         println("  ⚠️  With cutoff=0.0, ALL numerical noise is kept!")
#         println("  ⚠️  Rank = $bond_dim instead of true rank = 1")
#     end
# end

# println("=== Test: True Rank-3 Matrix ===")

# sv_rank3 = [1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# A_rank3 = U_base * diagm(sv_rank3) * V_base'
# A_tensor3 = ITensor(A_rank3, i, j)

# println("Intended: Rank = 3")
# println("Matrix constructed with: ", [@sprintf("%.1e", sv) for sv in sv_rank3])

# for cutoff in [0.0, 1e-15, 1e-12, 1e-2]
#     U, S, V = svd(A_tensor3, i; cutoff=cutoff)
#     s_values = [S[n, n] for n in 1:dim(inds(S)[1])]
#     bond_dim = length(s_values)
    
#     println(@sprintf("\nCutoff = %.0e:", cutoff))
#     println(@sprintf("  Bond dimension: %d", bond_dim))
#     println("  SVs: ", [@sprintf("%.2e", sv) for sv in s_values])
# end

# println("=== Key Takeaways ===")
# println("1. cutoff=0.0: Keeps EVERYTHING, including ~1e-16 numerical noise")
# println("   → Bond dimension = full matrix dimension (no truncation)")
# println("   → Cannot determine true rank of matrix")
# println()
# println("2. cutoff=1e-15: Discards numerical noise")
# println("   → Bond dimension ≈ true rank")
# println("   → Good default for Float64 calculations")
# println()
# println("3. SVD algorithm:")
# println("   • Computes ALL singular values first")
# println("   • Then DISCARDS values < cutoff")
# println("   • With cutoff=0.0, nothing is discarded")
# println("   • Numerical noise at ~1e-16 is treated as 'real'")
# println("=" ^ 60)

# # Additional test: Show the decision process
# println("=== Decision Process Visualization ===")
# A_test = ITensor(A_rank1, i, j)
# U, S, V = svd(A_test, i; cutoff=0.0)
# s_all = [S[n, n] for n in 1:dim(inds(S)[1])]

# println("\nAll SVs computed by algorithm:")
# for (idx, sv) in enumerate(s_all)
#     println(@sprintf("  SV[%d] = %.6e", idx, sv))
# end

# println("\nWith different cutoffs, which SVs are kept?")
# for cutoff in [0.0, 1e-16, 1e-14, 1e-12]
#     kept_count = sum(s_all .> cutoff)
#     println(@sprintf("  cutoff = %.0e → keep %d/%d values", cutoff, kept_count, length(s_all)))
# end

