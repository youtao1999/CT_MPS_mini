# precompile.jl
using CT, ArgParse
include("run_CT_MPS_1-3.jl")
# do one dummy call so all methods get JIT’ed
main_interactive(10, 0.3, 0.5, 0, typemax(Int), 1e-10, 43)

# To run the precompilation, use the following command:
# export JULIA_DEPOT_PATH=~/julia_depot
# export TMPDIR="/tmp" # (optional)
# using PackageCompiler; using Pkg; Pkg.activate("CT")
# create_sysimage(
#     [:CT, :ITensors, :ArgParse, :JSON, :MKL, :HDF5],
#     sysimage_path="ct_with_wrapper.so",
#     precompile_execution_file="precompile.jl",
#     project="CT"
#   )


# Then to start from the sysimage, use: 
# julia --sysimage ct_with_wrapper.so --project=.
# > using Pkg
# > Pkg.activate("CT")
# > main_interactive(8, 0.1, 0.2, 1, 1)
#
# Or call the script directly:
# julia --sysimage ct_with_wrapper.so \
#       --project=. \
#       -p 4 \
#       run_CT_MPS_C_m_T_multiproc.jl \
#       --params "0.5,0.0,8,1,1,0.5,0.0,8,2,2"