#!/usr/bin/env julia
# Cholesky Decomposition Benchmark Runner - CORRECTED VERSION
# Usage: julia -t N scripts/run_cholesky.jl --dataset MEDIUM

using LinearAlgebra
using Statistics
using Printf
using Dates

# Configure BLAS threads FIRST (before any BLAS operations)
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
    "MINI" => 40,
    "SMALL" => 120,
    "MEDIUM" => 400,
    "LARGE" => 2000,
    "EXTRALARGE" => 4000
)

# FLOPs calculation
flops_cholesky(n) = Float64(n)^3 / 3

# # Strategy classification for efficiency calculation
# # Key insight: efficiency = speedup * 100 for single-threaded strategies
# #              efficiency = (speedup / threads) * 100 for multi-threaded strategies
# const THREADED_STRATEGIES = Set(["threads", "threaded", "parallel"])

# function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)
#     if lowercase(strategy) in THREADED_STRATEGIES
#         # Multi-threaded: efficiency = (speedup / threads) * 100
#         return (speedup / max(nthreads, 1)) * 100.0
#     else
#         # Single-threaded (sequential, simd, blas, tiled): efficiency = speedup * 100
#         return speedup * 100.0
#     end
# end
const PARALLEL_STRATEGIES = Set(["threads", "threaded", "parallel"])

function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)::Float64
    if !(lowercase(strategy) in PARALLEL_STRATEGIES)
        return NaN
    end
    return (speedup / max(nthreads, 1)) * 100.0
end
#=============================================================================
 MATRIX INITIALIZATION - CRITICAL FOR CHOLESKY
 Must create a symmetric positive-definite (SPD) matrix
=============================================================================#
function init_array!(A::Matrix{Float64})
    n = size(A, 1)
    
    # Method: A = L * L^T where L is lower triangular with positive diagonal
    # This guarantees SPD
    
    # First, create a lower triangular matrix L
    L = zeros(n, n)
    @inbounds for j in 1:n
        # Diagonal elements: positive and dominant
        L[j, j] = Float64(n - j + 1) + 1.0
        # Below diagonal
        for i in (j+1):n
            L[i, j] = 1.0 / (i + j)
        end
    end
    
    # A = L * L^T (guaranteed SPD)
    # Column-major optimized
    fill!(A, 0.0)
    @inbounds for j in 1:n
        for k in 1:j  # L is lower triangular, so L[j,k] = 0 for k > j
            l_jk = L[j, k]
            for i in j:n  # L[i,k] = 0 for i < k, and we want lower triangle of A
                A[i, j] += L[i, k] * l_jk
            end
        end
    end
    
    # Symmetrize (copy lower to upper)
    @inbounds for j in 1:n
        for i in 1:(j-1)
            A[i, j] = A[j, i]
        end
    end
    
    return nothing
end

# Verify Cholesky result: ||A - L*L^T|| / ||A||
function verify_result(A_orig::Matrix{Float64}, L::Matrix{Float64})
    n = size(A_orig, 1)
    
    # Compute L * L^T
    LLT = zeros(n, n)
    @inbounds for j in 1:n
        for k in 1:j
            l_jk = L[j, k]
            for i in j:n
                LLT[i, j] += L[i, k] * l_jk
            end
        end
    end
    
    # Symmetrize
    @inbounds for j in 1:n
        for i in 1:(j-1)
            LLT[i, j] = LLT[j, i]
        end
    end
    
    # Relative error
    diff_norm = norm(A_orig - LLT)
    orig_norm = norm(A_orig)
    
    return diff_norm / max(orig_norm, 1e-10)
end

#=============================================================================
 KERNEL IMPLEMENTATIONS
=============================================================================#

# Strategy 1: Sequential baseline (Cholesky-Banachiewicz)
function kernel_cholesky_seq!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2))
        sum_sq = 0.0
        for k in 1:(j-1)
            sum_sq += A[j, k] * A[j, k]
        end
        diag_val = A[j, j] - sum_sq
        if diag_val <= 0.0
            error("Matrix not positive definite at column $j (diag=$diag_val)")
        end
        A[j, j] = sqrt(diag_val)
        
        # Column elements below diagonal
        diag_inv = 1.0 / A[j, j]
        for i in (j+1):n
            sum_prod = 0.0
            for k in 1:(j-1)
                sum_prod += A[i, k] * A[j, k]
            end
            A[i, j] = (A[i, j] - sum_prod) * diag_inv
        end
    end
    
    # Zero upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

# Strategy 2: SIMD-optimized dot products
function kernel_cholesky_simd!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element with SIMD reduction
        sum_sq = 0.0
        @simd for k in 1:(j-1)
            sum_sq += A[j, k] * A[j, k]
        end
        diag_val = A[j, j] - sum_sq
        if diag_val <= 0.0
            error("Matrix not positive definite at column $j")
        end
        A[j, j] = sqrt(diag_val)
        
        # Column elements below diagonal
        diag_inv = 1.0 / A[j, j]
        for i in (j+1):n
            sum_prod = 0.0
            @simd for k in 1:(j-1)
                sum_prod += A[i, k] * A[j, k]
            end
            A[i, j] = (A[i, j] - sum_prod) * diag_inv
        end
    end
    
    # Zero upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

# Strategy 3: Right-looking blocked with parallel trailing update
# This is the ONLY correct way to parallelize Cholesky
function kernel_cholesky_threads!(A::Matrix{Float64}; tile_size::Int=64)
    n = size(A, 1)
    ts = tile_size
    
    @inbounds for kk in 1:ts:n
        k_end = min(kk + ts - 1, n)
        
        # 1. Factor diagonal block (MUST be sequential due to dependencies)
        for j in kk:k_end
            sum_sq = 0.0
            for k in 1:(j-1)
                sum_sq += A[j, k] * A[j, k]
            end
            diag_val = A[j, j] - sum_sq
            if diag_val <= 0.0
                error("Matrix not positive definite at column $j in threads kernel")
            end
            A[j, j] = sqrt(diag_val)
            
            diag_inv = 1.0 / A[j, j]
            for i in (j+1):k_end
                sum_prod = 0.0
                for k in 1:(j-1)
                    sum_prod += A[i, k] * A[j, k]
                end
                A[i, j] = (A[i, j] - sum_prod) * diag_inv
            end
        end
        
        # 2. Panel solve: compute L21 (rows k_end+1:n, cols kk:k_end)
        if k_end < n
            for j in kk:k_end
                diag_inv = 1.0 / A[j, j]
                for i in (k_end+1):n
                    sum_prod = 0.0
                    for k in kk:(j-1)
                        sum_prod += A[i, k] * A[j, k]
                    end
                    A[i, j] = (A[i, j] - sum_prod) * diag_inv
                end
            end
        end
        
        # 3. PARALLEL trailing matrix update: A22 -= L21 * L21^T
        # This is where parallelism lives - columns are independent
        if k_end < n
            Threads.@threads :static for j in (k_end+1):n
                # Update column j of trailing matrix (rows j:n)
                for i in j:n
                    sum_update = 0.0
                    @simd for k in kk:k_end
                        sum_update += A[i, k] * A[j, k]
                    end
                    A[i, j] -= sum_update
                end
            end
        end
    end
    
    # Zero upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

# Strategy 4: BLAS-accelerated (LAPACK)
function kernel_cholesky_blas!(A::Matrix{Float64})
    n = size(A, 1)
    
    try
        # Use LAPACK cholesky (lower triangular)
        cholesky!(Hermitian(A, :L))
        
        # Zero upper triangular
        @inbounds for j in 2:n
            for i in 1:(j-1)
                A[i, j] = 0.0
            end
        end
    catch e
        error("BLAS Cholesky failed: $e")
    end
    
    return nothing
end

# Strategy 5: Tiled/Blocked (single-threaded, cache-optimized)
# FIXED VERSION - processes column by column within tiles
function kernel_cholesky_tiled!(A::Matrix{Float64}; tile_size::Int=64)
    n = size(A, 1)
    ts = tile_size
    
    @inbounds for kk in 1:ts:n
        k_end = min(kk + ts - 1, n)
        
        # Process columns kk to k_end
        for j in kk:k_end
            # Compute diagonal element
            sum_sq = 0.0
            for k in 1:(j-1)
                sum_sq += A[j, k] * A[j, k]
            end
            diag_val = A[j, j] - sum_sq
            if diag_val <= 0.0
                error("Matrix not positive definite at column $j in tiled kernel (diag=$diag_val)")
            end
            A[j, j] = sqrt(diag_val)
            
            # Compute column below diagonal
            diag_inv = 1.0 / A[j, j]
            for i in (j+1):n
                sum_prod = 0.0
                for k in 1:(j-1)
                    sum_prod += A[i, k] * A[j, k]
                end
                A[i, j] = (A[i, j] - sum_prod) * diag_inv
            end
        end
        
        # Update trailing matrix A22 = A22 - L21 * L21^T
        # where L21 is A[(k_end+1):n, kk:k_end]
        if k_end < n
            # Process trailing matrix in tiles for cache efficiency
            for jj in (k_end+1):ts:n
                j_end = min(jj + ts - 1, n)
                
                for j in jj:j_end
                    for i in j:n
                        sum_update = 0.0
                        @simd for k in kk:k_end
                            sum_update += A[i, k] * A[j, k]
                        end
                        A[i, j] -= sum_update
                    end
                end
            end
        end
    end
    
    # Zero upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
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

function run_benchmark(kernel!, A_orig::Matrix{Float64}; 
                       warmup::Int=3, iterations::Int=10, verify::Bool=true)
    times = Float64[]
    
    # Warmup
    for _ in 1:warmup
        A = copy(A_orig)
        try
            kernel!(A)
        catch e
            return BenchmarkResult("", Float64[], false, Inf)
        end
    end
    GC.gc()
    
    # Timed runs
    for _ in 1:iterations
        A = copy(A_orig)
        t = @elapsed kernel!(A)
        push!(times, t * 1000)  # Convert to ms
    end
    
    # Verify last result
    A_test = copy(A_orig)
    kernel!(A_test)
    err = verify ? verify_result(A_orig, A_test) : 0.0
    passed = err < 1e-6
    
    return BenchmarkResult("", times, passed, err)
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
        elseif ARGS[i] == "--warmup" && i < length(ARGS)
            warmup = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--no-verify"
            do_verify = false
            i += 1
        elseif ARGS[i] == "--output" && i < length(ARGS)
            output_csv = ARGS[i+1] == "csv"
            i += 2
        else
            i += 1
        end
    end
    
    if !haskey(DATASET_SIZES, dataset)
        println("Unknown dataset: $dataset")
        println("Available: ", join(keys(DATASET_SIZES), ", "))
        return
    end
    
    n = DATASET_SIZES[dataset]
    flops = flops_cholesky(n)
    memory_mb = n * n * 8 / 1024^2
    
    # Available strategies
    all_strategies = Dict(
        "sequential" => kernel_cholesky_seq!,
        "simd" => kernel_cholesky_simd!,
        "threads" => kernel_cholesky_threads!,
        "blas" => kernel_cholesky_blas!,
        "tiled" => kernel_cholesky_tiled!,
    )
    
    if strategies_arg == "all"
        strategies = ["sequential", "simd", "threads", "blas", "tiled"]
    else
        strategies = [strip(String(s)) for s in split(strategies_arg, ",")]
    end
    
    # Header
    println("="^70)
    println("CHOLESKY DECOMPOSITION BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset (n=$n)")
    @printf("Memory: %.2f MB\n", memory_mb)
    println()
    
    # Initialize matrix
    A_orig = Matrix{Float64}(undef, n, n)
    init_array!(A_orig)
    
    # Verify matrix is SPD by checking eigenvalues (debug)
    # min_eig = minimum(eigvals(Symmetric(A_orig)))
    # println("Min eigenvalue: $min_eig (should be > 0)")
    
    # Run benchmarks
    results = Dict{String, BenchmarkResult}()
    
    for strat in strategies
        if !haskey(all_strategies, strat)
            println("Unknown strategy: $strat")
            continue
        end
        
        kernel! = all_strategies[strat]
        res = run_benchmark(kernel!, A_orig; warmup=warmup, iterations=iterations, verify=do_verify)
        
        if isempty(res.times_ms)
            println("  Strategy $strat FAILED during execution")
            continue
        end
        
        results[strat] = BenchmarkResult(strat, res.times_ms, res.verified, res.error)
        
        if do_verify && !res.verified
            @printf("  Verification FAILED: %s (err=%.2e)\n", strat, res.error)
        end
    end
    
    if isempty(results)
        println("No successful benchmarks!")
        return
    end
    
    # Find sequential baseline
    seq_time = haskey(results, "sequential") ? minimum(results["sequential"].times_ms) : nothing
    
    # Print results table
    println()
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    # Output in consistent order
    for strat in ["sequential", "simd", "threads", "blas", "tiled"]
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
    
    # CSV output
    if output_csv
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        mkpath("results")
        filepath = "results/cholesky_$(dataset)_$(timestamp).csv"
        
        open(filepath, "w") do io
            println(io, "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified")
            
            for strat in ["sequential", "simd", "threads", "blas", "tiled"]
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
                
                # @printf(io, "cholesky,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
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
        println("\nResults exported to: $filepath")
    end
end

main()