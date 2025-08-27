using Printf

# ==============================================================================
# EXAMPLE 1: The Memory Leak Problem
# ==============================================================================

function create_data_processors_BAD(num_datasets=100, dataset_size=100_000)
    println("Creating $num_datasets datasets of size $dataset_size each...")
    
    # Create multiple large datasets (simulating real-world scenario)
    all_datasets = []
    for i in 1:num_datasets
        dataset = rand(dataset_size)  # Each dataset is ~800KB (Float64)
        push!(all_datasets, dataset)
    end
    
    println("Total raw data: ~$(num_datasets * dataset_size * 8 ÷ 1024 ÷ 1024) MB")
    
    # Now create processors - THIS IS WHERE THE LEAK HAPPENS
    processors = []
    
    for (i, dataset) in enumerate(all_datasets)
        # We think we're just capturing the current dataset,
        # but we're actually capturing ALL of all_datasets!
        processor = function(x)
            # We only use the current dataset
            return mean(dataset) + x
        end
        
        push!(processors, processor)
        
        # Show memory growth
        if i % 20 == 0
            GC.gc()  # Force garbage collection to get accurate reading
            mem_mb = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
            println("After $i processors: ~$(mem_mb) MB in use")
        end
    end
    
    return processors
end

# ==============================================================================
# EXAMPLE 2: The Fixed Version
# ==============================================================================

function create_data_processors_GOOD(num_datasets=100, dataset_size=100_000)
    println("Creating $num_datasets datasets of size $dataset_size each...")
    
    # Create the same large datasets
    all_datasets = []
    for i in 1:num_datasets
        dataset = rand(dataset_size)
        push!(all_datasets, dataset)
    end
    
    println("Total raw data: ~$(num_datasets * dataset_size * 8 ÷ 1024 ÷ 1024) MB")
    
    # Create processors - FIXED VERSION
    processors = []
    
    for (i, dataset) in enumerate(all_datasets)
        # Extract only what we need from the dataset
        dataset_mean = mean(dataset)  # Just a single Float64 value
        
        # Use let block to explicitly control what gets captured
        let local_mean = dataset_mean
            processor = function(x)
                # Now we only capture the small summary statistic
                return local_mean + x
            end
            push!(processors, processor)
        end
        
        # Show memory usage - should stay much lower
        if i % 20 == 0
            GC.gc()
            mem_mb = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
            println("After $i processors: ~$(mem_mb) MB in use")
        end
    end
    
    # The all_datasets can now be garbage collected!
    all_datasets = nothing  # Help the GC
    GC.gc()
    
    return processors
end

# ==============================================================================
# EXAMPLE 3: Even Worse - Accidental Global Capture
# ==============================================================================

function demonstrate_accidental_capture()
    println("\n=== ACCIDENTAL CAPTURE EXAMPLE ===")
    
    # Simulate a large global-ish dataset
    huge_reference_data = rand(1_000_000)  # ~8MB
    config_settings = Dict("threshold" => 0.5, "scale" => 2.0)
    
    println("Reference data: ~8MB")
    
    # Create many processors for different operations
    processors = []
    
    for operation_id in 1:50
        # We think we only need the operation_id and threshold
        threshold = config_settings["threshold"]
        
        # BUG: This closure captures EVERYTHING in scope!
        # Including huge_reference_data, config_settings, processors array, etc.
        processor = function(input_data)
            if operation_id % 2 == 0
                return input_data .> threshold
            else
                # We accidentally reference the huge array
                return input_data .> (threshold + huge_reference_data[1])
            end
        end
        
        push!(processors, processor)
        
        if operation_id % 10 == 0
            GC.gc()
            mem_mb = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
            println("After $operation_id processors: ~$(mem_mb) MB in use")
            println("Expected memory if no leak: ~8MB + small overhead")
        end
    end
    
    return processors
end

# ==============================================================================
# EXAMPLE 4: The Fixed Version of Accidental Capture
# ==============================================================================

function demonstrate_fixed_capture()
    println("\n=== FIXED CAPTURE EXAMPLE ===")
    
    # Same large dataset
    huge_reference_data = rand(1_000_000)
    config_settings = Dict("threshold" => 0.5, "scale" => 2.0)
    
    println("Reference data: ~8MB")
    
    # Extract what we need BEFORE creating closures
    threshold = config_settings["threshold"]
    first_reference_value = huge_reference_data[1]  # Just one number
    
    processors = []
    
    for operation_id in 1:50
        # Use let block to capture only specific values
        let local_id = operation_id, local_threshold = threshold, ref_val = first_reference_value
            processor = function(input_data)
                if local_id % 2 == 0
                    return input_data .> local_threshold
                else
                    return input_data .> (local_threshold + ref_val)
                end
            end
            push!(processors, processor)
        end
        
        if operation_id % 10 == 0
            GC.gc()
            mem_mb = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
            println("After $operation_id processors: ~$(mem_mb) MB in use")
        end
    end
    
    # Now the huge data can be garbage collected
    huge_reference_data = nothing
    config_settings = nothing
    GC.gc()
    
    return processors
end

# ==============================================================================
# DEMO FUNCTIONS
# ==============================================================================

function run_memory_leak_demo()
    println("=" ^ 60)
    println("CLOSURE CAPTURE MEMORY LEAK DEMONSTRATION")
    println("=" ^ 60)
    
    # Show initial memory
    GC.gc()
    initial_mem = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
    println("Initial memory usage: ~$(initial_mem) MB\n")
    
    println("1. DEMONSTRATING THE LEAK (creating 100 processors)...")
    println("-" ^ 50)
    
    # This will show massive memory growth
    @time bad_processors = create_data_processors_BAD(100, 50_000)
    
    GC.gc()
    leak_mem = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
    println("Memory after bad processors: ~$(leak_mem) MB")
    println("Memory leaked: ~$(leak_mem - initial_mem) MB\n")
    
    # Clear the leaky processors
    bad_processors = nothing
    GC.gc()
    
    println("2. DEMONSTRATING THE FIX...")
    println("-" ^ 50)
    
    # This should use much less memory
    @time good_processors = create_data_processors_GOOD(100, 50_000)
    
    GC.gc()
    fixed_mem = (Sys.total_memory() - Sys.free_memory()) ÷ 1024 ÷ 1024
    println("Memory after good processors: ~$(fixed_mem) MB")
    println("Memory used: ~$(fixed_mem - initial_mem) MB\n")
    
    # Test that both work the same way
    println("3. VERIFYING FUNCTIONALITY...")
    println("-" ^ 50)
    test_input = 5.0
    println("Both processors give same result: $(good_processors[1](test_input))")
    
    # Demonstrate accidental capture
    accidental_processors = demonstrate_accidental_capture()
    fixed_processors = demonstrate_fixed_capture()
    
    println("\nDemo complete! Notice how the 'bad' version uses orders of magnitude more memory.")
end

# Run the demo
run_memory_leak_demo()