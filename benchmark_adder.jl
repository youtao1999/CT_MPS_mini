using Pkg
Pkg.activate("CT")
using CT
include("global_adder_passthrough.jl")

# benchmark adder_MPO
using ITensors
# initialize random state
using .CT: _initialize_basis, _initialize_vector, P_MPO, XI_MPO, I_MPO, adder_MPO, add1, power_mpo
using Random

L = 12
ancilla = 0
folded = true
seed_vec = 123457
xj = Set([1//3, 2//3])
i1 = 1
p_ctrl = 1.0
p_proj = 0.0
_maxdim = 50
_cutoff = 1e-15
seed = 123457
x0 = nothing
qubit_site, ram_phy, phy_ram, phy_list = _initialize_basis(L, ancilla, folded)
println(phy_list)
println(ram_phy)
println(phy_ram)
rng = MersenneTwister(seed_vec)
rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
# initial_state = _initialize_vector(L, ancilla, x0, folded, qubit_site, ram_phy, phy_ram, phy_list, rng_vec, _cutoff, _maxdim);
# println(initial_state)
shift_1_3_bits, shift_1_3_amount = fraction_to_binary_shift(1, 3, L)
# initial_state = productMPS(qubit_site, [1,2,1,2,2,1,2,1,2,1,2,1,1,1,2,2])
initial_state = _initialize_vector(L, ancilla, x0, folded, qubit_site, ram_phy, phy_ram, phy_list, rng_vec, _cutoff, _maxdim);

# initialize haining adder mpo
add1_mpo=MPO(add1(i1,L,phy_ram,phy_list),qubit_site)
add1_6,add1_3=power_mpo(add1_mpo,[div(2^L,6)+1,div(2^L,3)])

# initialize tao adder mpo
carry_links, T_vec, id_vec, gate_vec = initialize_links(L, qubit_site, shift_1_3_bits, ram_phy);
initial_state_1 = copy(initial_state);
initial_state_2 = copy(initial_state);

# Performance Benchmark Configuration
using LinearAlgebra

# Print system information
println("Julia version: ", VERSION)
println("Available CPU cores: ", Sys.CPU_THREADS)
println("Julia threads: ", Threads.nthreads())

# Set BLAS to single-threaded for reproducible benchmarks
BLAS.set_num_threads(1)
println("BLAS threads: ", BLAS.get_num_threads())

# Optional: Set MKL threads if using MKL
try
    using MKL
    MKL.set_num_threads(1)
    println("MKL threads: ", MKL.get_num_threads())
catch
    println("MKL not available")
end

# Configure BenchmarkTools for consistency
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50      # Reduced for faster iteration
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1         # Single evaluation per sample
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60      # Reduced time budget
BenchmarkTools.DEFAULT_PARAMETERS.gctrial = false   # Disable GC between trials
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false  # Disable GC between samples

println("Benchmark configuration complete!")


# Prepare for benchmarking
using BenchmarkTools

# Force garbage collection before benchmarks for clean memory state
GC.gc()
GC.gc()  # Run twice to ensure cleanup

# Display current benchmark settings
println("Current BenchmarkTools settings:")
println("  samples: ", BenchmarkTools.DEFAULT_PARAMETERS.samples)
println("  evals: ", BenchmarkTools.DEFAULT_PARAMETERS.evals) 
println("  seconds: ", BenchmarkTools.DEFAULT_PARAMETERS.seconds)
println("  gctrial: ", BenchmarkTools.DEFAULT_PARAMETERS.gctrial)
println("  gcsample: ", BenchmarkTools.DEFAULT_PARAMETERS.gcsample)

println("\nSystem ready for benchmarking!")


initial_state_1 = apply(add1_3,initial_state_1);
global_adder(initial_state_2, carry_links, T_vec, id_vec, gate_vec, qubit_site, shift_1_3_bits, ram_phy)

@time initial_state_1 = apply(add1_3,initial_state_1; cutoff=1e-10, maxdim=typemax(Int))

# println(Sys.maxrss())
@time global_adder(initial_state_2, carry_links, T_vec, id_vec, gate_vec, qubit_site, shift_1_3_bits, ram_phy; cutoff=1e-10, maxdim=typemax(Int))
# println(Sys.maxrss())
