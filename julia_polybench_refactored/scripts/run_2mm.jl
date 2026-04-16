#!/usr/bin/env julia
#=
2MM Benchmark Runner - REDESIGNED
Computation: D = alpha * A * B * C + beta * D
    tmp = alpha * A * B
    D = tmp * C + beta * D

Follows the working pattern from correlation, jacobi2d, cholesky benchmarks.

Usage:
  julia -t 8 run_2mm.jl --dataset MEDIUM
  julia -t 16 run_2mm.jl --dataset LARGE --output csv
  julia -t 8 run_2mm.jl --strategies sequential,threads_static,blas

DAS-5 SLURM:
  srun -N 1 -c 16 --time=01:00:00 julia -t 16 run_2mm.jl --dataset LARGE --output csv
=#

using LinearAlgebra
using Statistics
using Printf
using Dates

#=============================================================================
 BLAS Configuration - Must happen before any BLAS calls
=============================================================================#
function configure_blas_threads()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    else
        BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
    end
end

configure_blas_threads()

#=============================================================================
 Dataset Definitions
=============================================================================#
const DATASET_SIZES = Dict(
    "MINI"       => (ni=16,  nj=18,  nk=22,  nl=24),
    "SMALL"      => (ni=40,  nj=50,  nk=70,  nl=80),
    "MEDIUM"     => (ni=180, nj=190, nk=210, nl=220),
    "LARGE"      => (ni=800, nj=900, nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

#=============================================================================
 FLOPs and Memory Calculations
=============================================================================#
function flops_2mm(ni, nj, nk, nl)
    # tmp = alpha * A * B: 2 * ni * nj * nk (multiply-add)
    # D = tmp * C + beta * D: 2 * ni * nl * nj + ni * nl (multiply-add + scale)
    return 2 * ni * nj * nk + 2 * ni * nl * nj + ni * nl
end

function memory_2mm(ni, nj, nk, nl)
    # A(ni,nk) + B(nk,nj) + tmp(ni,nj) + C(nj,nl) + D(ni,nl)
    return (ni*nk + nk*nj + ni*nj + nj*nl + ni*nl) * sizeof(Float64)
end

#=============================================================================
 Strategy Classification for Efficiency Calculation
=============================================================================#
# const THREADED_STRATEGIES = Set(["threads_static", "threads_dynamic", "tiled", "tasks"])

# function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)
#     if lowercase(strategy) in THREADED_STRATEGIES
#         return (speedup / max(nthreads, 1)) * 100.0
#     else
#         return speedup * 100.0
#     end
# end
const PARALLEL_STRATEGIES = Set(["threads_static", "threads_dynamic", "tiled", "tasks"])

function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)::Float64
    if !(lowercase(strategy) in PARALLEL_STRATEGIES)
        return NaN
    end
    return (speedup / max(nthreads, 1)) * 100.0
end
#=============================================================================
 Data Initialization (PolyBench standard)
=============================================================================#
function init_2mm!(alpha::Ref{Float64}, beta::Ref{Float64},
                   A::Matrix{Float64}, B::Matrix{Float64},
                   tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(D)
    
    alpha[] = 1.5
    beta[] = 1.2
    
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = Float64((i-1) * (j-1) % ni) / ni
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = Float64((i-1) * ((j-1)+1) % nj) / nj
    end
    
    @inbounds for j in 1:nl, i in 1:nj
        C[i, j] = Float64(((i-1)+3) * (j-1) % nl) / nl
    end
    
    @inbounds for j in 1:nl, i in 1:ni
        D[i, j] = Float64((i-1) * ((j-1)+2) % nk) / nk
    end
    
    fill!(tmp, 0.0)
    return nothing
end

#=============================================================================
 Kernel Implementations
=============================================================================#

# Strategy 1: Sequential baseline with SIMD hints
function kernel_2mm_seq!(alpha::Float64, beta::Float64,
                         A::Matrix{Float64}, B::Matrix{Float64},
                         tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(D)
    
    # tmp = alpha * A * B
    @inbounds for j in 1:nj
        for i in 1:ni
            sum_val = 0.0
            @simd for k in 1:nk
                sum_val += alpha * A[i, k] * B[k, j]
            end
            tmp[i, j] = sum_val
        end
    end
    
    # D = tmp * C + beta * D
    @inbounds for j in 1:nl
        for i in 1:ni
            D[i, j] *= beta
            @simd for k in 1:nj
                D[i, j] += tmp[i, k] * C[k, j]
            end
        end
    end
    return nothing
end

# Strategy 2: Threads with static scheduling
function kernel_2mm_threads_static!(alpha::Float64, beta::Float64,
                                    A::Matrix{Float64}, B::Matrix{Float64},
                                    tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(D)
    
    # tmp = alpha * A * B (parallelize over rows)
    Threads.@threads :static for i in 1:ni
        @inbounds for j in 1:nj
            sum_val = 0.0
            @simd for k in 1:nk
                sum_val += alpha * A[i, k] * B[k, j]
            end
            tmp[i, j] = sum_val
        end
    end
    
    # D = tmp * C + beta * D (parallelize over rows)
    Threads.@threads :static for i in 1:ni
        @inbounds for j in 1:nl
            D[i, j] *= beta
            @simd for k in 1:nj
                D[i, j] += tmp[i, k] * C[k, j]
            end
        end
    end
    return nothing
end

# Strategy 3: Threads with dynamic scheduling
function kernel_2mm_threads_dynamic!(alpha::Float64, beta::Float64,
                                     A::Matrix{Float64}, B::Matrix{Float64},
                                     tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(D)
    
    Threads.@threads :dynamic for i in 1:ni
        @inbounds for j in 1:nj
            sum_val = 0.0
            @simd for k in 1:nk
                sum_val += alpha * A[i, k] * B[k, j]
            end
            tmp[i, j] = sum_val
        end
    end
    
    Threads.@threads :dynamic for i in 1:ni
        @inbounds for j in 1:nl
            D[i, j] *= beta
            @simd for k in 1:nj
                D[i, j] += tmp[i, k] * C[k, j]
            end
        end
    end
    return nothing
end

# Strategy 4: Tiled/Blocked for cache optimization
function kernel_2mm_tiled!(alpha::Float64, beta::Float64,
                           A::Matrix{Float64}, B::Matrix{Float64},
                           tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64};
                           tile_size::Int=32)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(D)
    ts = tile_size
    
    # tmp = alpha * A * B (tiled)
    Threads.@threads :static for ii in 1:ts:ni
        i_end = min(ii + ts - 1, ni)
        @inbounds for jj in 1:ts:nj
            j_end = min(jj + ts - 1, nj)
            for kk in 1:ts:nk
                k_end = min(kk + ts - 1, nk)
                for i in ii:i_end
                    for j in jj:j_end
                        sum_val = tmp[i, j]
                        @simd for k in kk:k_end
                            sum_val += alpha * A[i, k] * B[k, j]
                        end
                        tmp[i, j] = sum_val
                    end
                end
            end
        end
    end
    
    # D = tmp * C + beta * D (tiled)
    Threads.@threads :static for ii in 1:ts:ni
        i_end = min(ii + ts - 1, ni)
        @inbounds for i in ii:i_end
            for j in 1:nl
                D[i, j] *= beta
            end
        end
        @inbounds for jj in 1:ts:nl
            j_end = min(jj + ts - 1, nl)
            for kk in 1:ts:nj
                k_end = min(kk + ts - 1, nj)
                for i in ii:i_end
                    for j in jj:j_end
                        sum_val = D[i, j]
                        @simd for k in kk:k_end
                            sum_val += tmp[i, k] * C[k, j]
                        end
                        D[i, j] = sum_val
                    end
                end
            end
        end
    end
    return nothing
end

# Strategy 5: BLAS-based implementation
function kernel_2mm_blas!(alpha::Float64, beta::Float64,
                          A::Matrix{Float64}, B::Matrix{Float64},
                          tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    # tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0)
    # D = tmp * C + beta * D
    mul!(D, tmp, C, 1.0, beta)
    return nothing
end

# Strategy 6: Task-based parallelism
function kernel_2mm_tasks!(alpha::Float64, beta::Float64,
                           A::Matrix{Float64}, B::Matrix{Float64},
                           tmp::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(D)
    
    chunk_size = max(1, div(ni, 2 * Threads.nthreads()))
    
    # Phase 1: tmp = alpha * A * B
    @sync begin
        for i_start in 1:chunk_size:ni
            i_end = min(i_start + chunk_size - 1, ni)
            Threads.@spawn begin
                @inbounds for i in i_start:i_end
                    for j in 1:nj
                        sum_val = 0.0
                        @simd for k in 1:nk
                            sum_val += alpha * A[i, k] * B[k, j]
                        end
                        tmp[i, j] = sum_val
                    end
                end
            end
        end
    end
    
    # Phase 2: D = tmp * C + beta * D
    @sync begin
        for i_start in 1:chunk_size:ni
            i_end = min(i_start + chunk_size - 1, ni)
            Threads.@spawn begin
                @inbounds for i in i_start:i_end
                    for j in 1:nl
                        D[i, j] *= beta
                        @simd for k in 1:nj
                            D[i, j] += tmp[i, k] * C[k, j]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

#=============================================================================
 Strategy Registry
=============================================================================#
const ALL_STRATEGIES = Dict(
    "sequential"      => kernel_2mm_seq!,
    "threads_static"  => kernel_2mm_threads_static!,
    "threads_dynamic" => kernel_2mm_threads_dynamic!,
    "tiled"           => kernel_2mm_tiled!,
    "blas"            => kernel_2mm_blas!,
    "tasks"           => kernel_2mm_tasks!
)

const STRATEGY_ORDER = ["sequential", "threads_static", "threads_dynamic", "tiled", "blas", "tasks"]

#=============================================================================
 Local BenchmarkResult Struct (matches working pattern)
=============================================================================#
struct BenchmarkResult
    strategy::String
    times_ms::Vector{Float64}
    verified::Bool
    error::Float64
end

#=============================================================================
 Benchmark Runner
=============================================================================#
function run_benchmark(kernel!, alpha::Float64, beta::Float64,
                       A::Matrix{Float64}, B::Matrix{Float64},
                       tmp::Matrix{Float64}, C::Matrix{Float64},
                       D::Matrix{Float64}, D_orig::Matrix{Float64};
                       warmup::Int=5, iterations::Int=10)
    times = Float64[]
    
    # Warmup runs (not timed)
    for _ in 1:warmup
        fill!(tmp, 0.0)
        copyto!(D, D_orig)
        kernel!(alpha, beta, A, B, tmp, C, D)
    end
    GC.gc()
    
    # Timed runs
    for _ in 1:iterations
        fill!(tmp, 0.0)
        copyto!(D, D_orig)
        t = @elapsed kernel!(alpha, beta, A, B, tmp, C, D)
        push!(times, t * 1000)  # Convert to ms
    end
    
    return times
end

#=============================================================================
 Verification
=============================================================================#
function verify_result(D_ref::Matrix{Float64}, D_test::Matrix{Float64}, ni::Int, nj::Int, nk::Int, nl::Int)
    max_error = maximum(abs.(D_ref .- D_test))
    # Scale-aware tolerance for numerical stability
    # BLAS uses different accumulation order, expect O(sqrt(n*k)) * eps errors
    scale_factor = sqrt(Float64(ni) * Float64(nj) * Float64(nk) * Float64(nl))
    tolerance = max(1e-10, 1e-14 * scale_factor)
    return max_error, max_error < tolerance
end

#=============================================================================
 Main Function
=============================================================================#
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
            output_csv = lowercase(ARGS[i+1]) == "csv"
            i += 2
        elseif ARGS[i] == "--help" || ARGS[i] == "-h"
            println("2MM Benchmark Runner")
            println("Usage: julia -t N run_2mm.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      MINI/SMALL/MEDIUM/LARGE/EXTRALARGE (default: MEDIUM)")
            println("  --strategies LIST   Comma-separated or 'all' (default: all)")
            println("  --iterations N      Timed iterations (default: 10)")
            println("  --warmup N          Warmup iterations (default: 5)")
            println("  --no-verify         Skip verification")
            println("  --output csv        Export results to CSV file")
            println()
            println("Strategies: ", join(STRATEGY_ORDER, ", "))
            return
        else
            i += 1
        end
    end
    
    # Validate dataset
    if !haskey(DATASET_SIZES, dataset)
        println("Unknown dataset: $dataset")
        println("Available: ", join(keys(DATASET_SIZES), ", "))
        return
    end
    
    params = DATASET_SIZES[dataset]
    ni, nj, nk, nl = params.ni, params.nj, params.nk, params.nl
    flops = flops_2mm(ni, nj, nk, nl)
    memory_mb = memory_2mm(ni, nj, nk, nl) / 1024^2
    
    # Determine strategies to run
    if strategies_arg == "all"
        strategies = STRATEGY_ORDER
    else
        strategies = [strip(String(s)) for s in split(strategies_arg, ",")]
    end
    
    # Print header
    println("="^70)
    println("2MM BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset (ni=$ni, nj=$nj, nk=$nk, nl=$nl)")
    @printf("Memory: %.2f MB\n", memory_mb)
    @printf("FLOPs: %d (%.2f GFLOPs)\n", flops, flops / 1e9)
    println()
    
    # Allocate matrices
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    tmp = Matrix{Float64}(undef, ni, nj)
    C = Matrix{Float64}(undef, nj, nl)
    D = Matrix{Float64}(undef, ni, nl)
    D_orig = Matrix{Float64}(undef, ni, nl)
    D_ref = Matrix{Float64}(undef, ni, nl)
    
    # Initialize
    alpha = Ref(0.0)
    beta = Ref(0.0)
    init_2mm!(alpha, beta, A, B, tmp, C, D)
    copyto!(D_orig, D)
    
    # Compute reference result
    fill!(tmp, 0.0)
    copyto!(D_ref, D_orig)
    kernel_2mm_seq!(alpha[], beta[], A, B, tmp, C, D_ref)
    
    # Run benchmarks and collect results
    results = Dict{String, BenchmarkResult}()
    
    for strat in strategies
        if !haskey(ALL_STRATEGIES, strat)
            println("Unknown strategy: $strat")
            continue
        end
        
        kernel! = ALL_STRATEGIES[strat]
        times = run_benchmark(kernel!, alpha[], beta[], A, B, tmp, C, D, D_orig;
                              warmup=warmup, iterations=iterations)
        
        # Verify
        fill!(tmp, 0.0)
        copyto!(D, D_orig)
        kernel!(alpha[], beta[], A, B, tmp, C, D)
        err, passed = do_verify ? verify_result(D_ref, D, ni, nj, nk, nl) : (0.0, true)
        
        results[strat] = BenchmarkResult(strat, times, passed, err)
        
        if do_verify && !passed
            @printf("  Verification WARNING: %s (max_error=%.2e)\n", strat, err)
        end
    end
    
    # Find sequential baseline for speedup calculation
    seq_time = haskey(results, "sequential") ? minimum(results["sequential"].times_ms) : nothing
    
    # Print results table
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    for strat in STRATEGY_ORDER
        if !haskey(results, strat)
            continue
        end
        
        r = results[strat]
        min_t = minimum(r.times_ms)
        med_t = median(r.times_ms)
        mean_t = Statistics.mean(r.times_ms)  # Fully qualified to avoid shadowing
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
    
    # CSV Export (proper file writing)
    if output_csv
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        mkpath("results")
        filepath = "results/2mm_$(dataset)_$(timestamp).csv"
        
        open(filepath, "w") do io
            println(io, "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified")
            
            for strat in STRATEGY_ORDER
                if !haskey(results, strat)
                    continue
                end
                
                r = results[strat]
                min_t = minimum(r.times_ms)
                med_t = median(r.times_ms)
                mean_t = Statistics.mean(r.times_ms)
                std_t = length(r.times_ms) > 1 ? std(r.times_ms) : 0.0
                gflops = flops / (min_t / 1000) / 1e9
                speedup = seq_time === nothing ? 1.0 : seq_time / min_t
                efficiency = compute_efficiency(strat, speedup, Threads.nthreads())
                
                # @printf(io, "2mm,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
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
