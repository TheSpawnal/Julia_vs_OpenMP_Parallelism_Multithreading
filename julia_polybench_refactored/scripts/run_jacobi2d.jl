#!/usr/bin/env julia
# Jacobi-2D Stencil Benchmark Runner - CORRECTED VERSION
# Usage: julia -t N scripts/run_jacobi2d.jl --dataset MEDIUM

using LinearAlgebra
using Statistics
using Printf
using Dates

# Configure BLAS threads FIRST
function configure_blas_threads()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
    end
end

configure_blas_threads()

# Dataset sizes
const DATASET_SIZES = Dict(
    "MINI" => (n=30, tsteps=20),
    "SMALL" => (n=90, tsteps=40),
    "MEDIUM" => (n=250, tsteps=100),
    "LARGE" => (n=1300, tsteps=500),
    "EXTRALARGE" => (n=2800, tsteps=1000)
)

# FLOPs calculation: 5 FLOPs per point (4 adds + 1 mul by 0.2), (n-2)^2 points, tsteps iterations
flops_jacobi2d(n, tsteps) = Float64(tsteps) * Float64(n-2)^2 * 5

# Strategy classification
# threads, tiled, redblack are ALL threaded strategies
# const THREADED_STRATEGIES = Set(["threads", "tiled", "redblack"])

# function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)
#     if lowercase(strategy) in THREADED_STRATEGIES
#         return (speedup / max(nthreads, 1)) * 100.0
#     else
#         return speedup * 100.0
#     end
# end
const PARALLEL_STRATEGIES = Set(["threads", "tiled", "redblack"])

function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)::Float64
    if !(lowercase(strategy) in PARALLEL_STRATEGIES)
        return NaN
    end
    return (speedup / max(nthreads, 1)) * 100.0
end
#=============================================================================
 INITIALIZATION
=============================================================================#
function init_arrays!(A::Matrix{Float64}, B::Matrix{Float64})
    n = size(A, 1)
    @inbounds for j in 1:n
        for i in 1:n
            A[i, j] = Float64((i-1) * (n - (i-1)) + (j-1) * (n - (j-1))) / n
            B[i, j] = Float64((i-1) * (n - (i-1)) + (j-1) * (n - (j-1))) / n
        end
    end
    return nothing
end

function verify_result(A_ref::Matrix{Float64}, A_test::Matrix{Float64})
    return norm(A_ref - A_test) / max(norm(A_ref), 1e-10)
end

#=============================================================================
 KERNEL IMPLEMENTATIONS
=============================================================================#

# Strategy 1: Sequential baseline
function kernel_jacobi2d_seq!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int)
    n = size(A, 1)
    
    @inbounds for t in 1:tsteps
        # Update B from A
        for j in 2:(n-1)
            for i in 2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
            end
        end
        
        # Update A from B
        for j in 2:(n-1)
            for i in 2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
            end
        end
    end
    
    return nothing
end

# Strategy 2: Threaded (parallel over rows)
function kernel_jacobi2d_threads!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int)
    n = size(A, 1)
    
    @inbounds for t in 1:tsteps
        # Update B from A (parallel over columns for column-major)
        Threads.@threads :static for j in 2:(n-1)
            @simd for i in 2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
            end
        end
        
        # Update A from B
        Threads.@threads :static for j in 2:(n-1)
            @simd for i in 2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
            end
        end
    end
    
    return nothing
end

# Strategy 3: Tiled (threaded with cache blocking)
function kernel_jacobi2d_tiled!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int;
                                 tile_size::Int=32)
    n = size(A, 1)
    ts = tile_size
    
    @inbounds for t in 1:tsteps
        # Update B from A (tiled and threaded)
        Threads.@threads :static for jj in 2:ts:(n-1)
            j_end = min(jj + ts - 1, n - 1)
            for ii in 2:ts:(n-1)
                i_end = min(ii + ts - 1, n - 1)
                for j in jj:j_end
                    @simd for i in ii:i_end
                        B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
                    end
                end
            end
        end
        
        # Update A from B (tiled and threaded)
        Threads.@threads :static for jj in 2:ts:(n-1)
            j_end = min(jj + ts - 1, n - 1)
            for ii in 2:ts:(n-1)
                i_end = min(ii + ts - 1, n - 1)
                for j in jj:j_end
                    @simd for i in ii:i_end
                        A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
                    end
                end
            end
        end
    end
    
    return nothing
end

# Strategy 4: Red-Black Gauss-Seidel (threaded)
# Note: This changes the algorithm slightly but allows more parallelism
function kernel_jacobi2d_redblack!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int)
    n = size(A, 1)
    
    @inbounds for t in 1:tsteps
        # Red phase: update B from A where (i+j) is even
        Threads.@threads :static for j in 2:(n-1)
            start_i = 2 + ((j + 1) % 2)  # Start at 2 if j is odd, 3 if j is even
            for i in start_i:2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
            end
        end
        
        # Black phase: update B from A where (i+j) is odd
        Threads.@threads :static for j in 2:(n-1)
            start_i = 2 + (j % 2)  # Start at 3 if j is odd, 2 if j is even
            for i in start_i:2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
            end
        end
        
        # Red phase: update A from B
        Threads.@threads :static for j in 2:(n-1)
            start_i = 2 + ((j + 1) % 2)
            for i in start_i:2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
            end
        end
        
        # Black phase: update A from B
        Threads.@threads :static for j in 2:(n-1)
            start_i = 2 + (j % 2)
            for i in start_i:2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
            end
        end
    end
    
    return nothing
end

#=============================================================================
 BENCHMARK RUNNER
=============================================================================#

struct BenchmarkResult
    strategy::String
    times_ms::Vector{Float64}
    verified::Bool
    error::Float64
end

function run_benchmark(kernel!, A_orig::Matrix{Float64}, B_orig::Matrix{Float64}, tsteps::Int;
                       warmup::Int=3, iterations::Int=10)
    times = Float64[]
    
    # Warmup
    for _ in 1:warmup
        A = copy(A_orig)
        B = copy(B_orig)
        kernel!(A, B, tsteps)
    end
    GC.gc()
    
    # Timed runs
    for _ in 1:iterations
        A = copy(A_orig)
        B = copy(B_orig)
        t = @elapsed kernel!(A, B, tsteps)
        push!(times, t * 1000)
    end
    
    return times
end

function main()
    # Parse arguments
    dataset = "MEDIUM"
    strategies_arg = "all"
    iterations = 10
    warmup = 5
    do_verify = true
    output_csv = false
    
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--dataset" && i < length(ARGS)
            dataset = uppercase(ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--strategies" && i < length(ARGS)
            strategies_arg = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--iterations" && i < length(ARGS)
            iterations = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--output" && i < length(ARGS)
            output_csv = ARGS[i+1] == "csv"
            i += 2
        else
            i += 1
        end
    end
    
    if !haskey(DATASET_SIZES, dataset)
        println("Unknown dataset: $dataset")
        return
    end
    
    params = DATASET_SIZES[dataset]
    n, tsteps = params.n, params.tsteps
    flops = flops_jacobi2d(n, tsteps)
    memory_mb = 2 * n * n * 8 / 1024^2
    
    # Arithmetic intensity (memory-bound indicator)
    bytes_per_point = 5 * 8 + 1 * 8  # 5 reads + 1 write per point
    ai = 5.0 / (bytes_per_point / 8)  # FLOPs per double loaded
    
    # Available strategies
    all_strategies = Dict(
        "sequential" => kernel_jacobi2d_seq!,
        "threads" => kernel_jacobi2d_threads!,
        "tiled" => kernel_jacobi2d_tiled!,
        "redblack" => kernel_jacobi2d_redblack!,
    )
    
    if strategies_arg == "all"
        strategies = ["sequential", "threads", "tiled", "redblack"]
    else
        strategies = [strip(String(s)) for s in split(strategies_arg, ",")]
    end
    
    # Header
    println("="^70)
    println("JACOBI-2D STENCIL BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset (n=$n, tsteps=$tsteps)")
    @printf("Memory: %.2f MB\n", memory_mb)
    @printf("NOTE: Memory-bound (arithmetic intensity ~%.2f FLOPs/byte)\n", ai)
    println()
    
    # Allocate arrays
    A_orig = Matrix{Float64}(undef, n, n)
    B_orig = Matrix{Float64}(undef, n, n)
    init_arrays!(A_orig, B_orig)
    
    # Reference result
    A_ref = copy(A_orig)
    B_ref = copy(B_orig)
    kernel_jacobi2d_seq!(A_ref, B_ref, tsteps)
    
    # Run benchmarks
    results = Dict{String, BenchmarkResult}()
    
    for strat in strategies
        if !haskey(all_strategies, strat)
            println("Unknown strategy: $strat")
            continue
        end
        
        kernel! = all_strategies[strat]
        times = run_benchmark(kernel!, A_orig, B_orig, tsteps;
                              warmup=warmup, iterations=iterations)
        
        # Verify
        A_test = copy(A_orig)
        B_test = copy(B_orig)
        kernel!(A_test, B_test, tsteps)
        
        # Red-black changes convergence slightly, use looser tolerance
        tolerance = strat == "redblack" ? 1e-4 : 1e-6
        err = do_verify ? verify_result(A_ref, A_test) : 0.0
        passed = err < tolerance
        
        results[strat] = BenchmarkResult(strat, times, passed, err)
        
        if do_verify && !passed
            @printf("  Verification FAILED: %s (err=%.2e)\n", strat, err)
        end
    end
    
    # Find sequential baseline
    seq_time = haskey(results, "sequential") ? minimum(results["sequential"].times_ms) : nothing
    
    # Print results
    println()
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    for strat in ["sequential", "threads", "tiled", "redblack"]
        if !haskey(results, strat)
            continue
        end
        
        r = results[strat]
        min_t = minimum(r.times_ms)
        med_t = median(r.times_ms)
        mean_t = mean(r.times_ms)
        gflops = flops / (min_t / 1000) / 1e9
        
        speedup = seq_time === nothing ? 1.0 : seq_time / min_t
        efficiency = compute_efficiency(strat, speedup, Threads.nthreads())
        
        # @printf("%-16s | %10.3f | %10.3f | %10.3f | %8.2f | %8.2fx | %6.1f\n",
        #         strat, min_t, med_t, mean_t, gflops, speedup, efficiency)
        eff_str = isnan(efficiency) ? "  N/A" : @sprintf("%6.1f", efficiency)
        @printf("%-16s | %10.3f | %10.3f | %10.3f | %8.2f | %8.2fx | %s\n",
            strat, min_t, med_t, mean_t, gflops, speedup, eff_str)
    end
    println("-"^90)
    
    # Memory bandwidth estimate
    if haskey(results, "threads")
        min_t = minimum(results["threads"].times_ms) / 1000  # seconds
        bytes_transferred = Float64(tsteps) * Float64(n-2)^2 * bytes_per_point
        bandwidth_gbs = bytes_transferred / min_t / 1e9
        println()
        println("Memory Bandwidth:")
        @printf("  Estimated: %.2f GB/s\n", bandwidth_gbs)
    end
    
    # CSV output
    if output_csv
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        mkpath("results")
        filepath = "results/jacobi2d_$(dataset)_$(timestamp).csv"
        
        open(filepath, "w") do io
            println(io, "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified")
            
            for strat in ["sequential", "threads", "tiled", "redblack"]
                if !haskey(results, strat)
                    continue
                end
                
                r = results[strat]
                min_t = minimum(r.times_ms)
                med_t = median(r.times_ms)
                mean_t = mean(r.times_ms)
                std_t = length(r.times_ms) > 1 ? std(r.times_ms) : 0.0
                gflops = flops / (min_t / 1000) / 1e9
                speedup = seq_time === nothing ? 1.0 : seq_time / min_t
                efficiency = compute_efficiency(strat, speedup, Threads.nthreads())
                
                # @printf(io, "jacobi2d,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
                #         dataset, strat, Threads.nthreads(),
                #         min_t, med_t, mean_t, std_t, gflops, speedup, efficiency,
                #         r.verified ? "PASS" : "FAIL")
                eff_str = isnan(efficiency) ? "" : @sprintf("%.1f", efficiency)
                @printf(io, "correlation,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%s,%s\n",
                        dataset, strat, Threads.nthreads(),
                        min_t, med_t, mean_t, std_t, gflops, speedup, eff_str,
                        r.verified ? "PASS" : "FAIL")
            end
        end
        println("Results exported to: $filepath")
    end
end

main()