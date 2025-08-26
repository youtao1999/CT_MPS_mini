#!/usr/bin/env julia

using MPI

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    println("=== MPI Test Worker $rank ===")
    println("Worker $rank of $size on node $(gethostname())")
    println("Worker $rank: MPI is working correctly!")
    flush(stdout)
    
    # Simple synchronization test
    MPI.Barrier(comm)
    
    if rank == 0
        println("All $size workers completed successfully!")
    end
    
    MPI.Finalize()
end

main()
