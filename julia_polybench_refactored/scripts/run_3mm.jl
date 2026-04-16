#!/usr/bin/env julia
#=
3MM Benchmark Runner - REDESIGNED
Computation: G = (A * B) * (C * D)
    E = A * B
    F = C * D
    G = E * F

Follows the working pattern from correlation, jacobi2d, cholesky benchmarks.

Usage:
  julia -t 8 run_3mm.jl --dataset MEDIUM
  julia -t 16 run_3mm.jl --dataset LARGE --output csv
  julia -t 8 run_3mm.jl --strategies sequential,threads_static,blas

DAS-5 SLURM:
  srun -N 1 -c 16 --time=01:00:00 julia -t 16 run_3mm.jl --dataset LARGE --output csv
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
    "MINI"       => (ni=16,  nj=18,  nk=20,  nl=22,  nm=24),
    "SMALL"      => (ni=40,  nj=50,  nk=60,  nl=70,  nm=80),
    "MEDIUM"     => (ni=180, nj=190, nk=200, nl=210, nm=220),
    "LARGE"      => (ni=800, nj=900, nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

#=============================================================================
 FLOPs and Memory Calculations
=============================================================================#
function flops_3mm(ni, nj, nk, nl, nm)
    # E = A * B: 2 * ni * nj * nk
    # F = C * D: 2 * nj * nl * nm
    # G = E * F: 2 * ni * nl * nj
    return 2 * ni * nj * nk + 2 * nj * nl * nm + 2 * ni * nl * nj
end

function memory_3mm(ni, nj, nk, nl, nm)
    # A(ni,nk) + B(nk,nj) + C(nj,nm) + D(nm,nl) + E(ni,nj) + F(nj,nl) + G(ni,nl)
    return (ni*nk + nk*nj + nj*nm + nm*nl + ni*nj + nj*nl + ni*nl) * sizeof(Float64)
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
function init_3mm!(A::Matrix{Float64}, B::Matrix{Float64},
                   C::Matrix{Float64}, D::Matrix{Float64},
                   E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = Float64((i-1) * (j-1) % ni) / (5 * ni)
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = Float64((i-1) * ((j-1)+1) % nj) / (5 * nj)
    end
    
    @inbounds for j in 1:nm, i in 1:nj
        C[i, j] = Float64(((i-1)+3) * (j-1) % nl) / (5 * nl)
    end
    
    @inbounds for j in 1:nl, i in 1:nm
        D[i, j] = Float64((i-1) * ((j-1)+2) % nk) / (5 * nk)
    end
    
    fill!(E, 0.0)
    fill!(F, 0.0)
    fill!(G, 0.0)
    return nothing
end

#=============================================================================
 Kernel Implementations
=============================================================================#

# Strategy 1: Sequential baseline with SIMD hints
function kernel_3mm_seq!(A::Matrix{Float64}, B::Matrix{Float64},
                         C::Matrix{Float64}, D::Matrix{Float64},
                         E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B
    @inbounds for j in 1:nj
        for i in 1:ni
            sum_val = 0.0
            @simd for k in 1:nk
                sum_val += A[i, k] * B[k, j]
            end
            E[i, j] = sum_val
        end
    end
    
    # F = C * D
    @inbounds for j in 1:nl
        for i in 1:nj
            sum_val = 0.0
            @simd for k in 1:nm
                sum_val += C[i, k] * D[k, j]
            end
            F[i, j] = sum_val
        end
    end
    
    # G = E * F
    @inbounds for j in 1:nl
        for i in 1:ni
            sum_val = 0.0
            @simd for k in 1:nj
                sum_val += E[i, k] * F[k, j]
            end
            G[i, j] = sum_val
        end
    end
    return nothing
end

# Strategy 2: Threads with static scheduling
function kernel_3mm_threads_static!(A::Matrix{Float64}, B::Matrix{Float64},
                                    C::Matrix{Float64}, D::Matrix{Float64},
                                    E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B
    Threads.@threads :static for i in 1:ni
        @inbounds for j in 1:nj
            sum_val = 0.0
            @simd for k in 1:nk
                sum_val += A[i, k] * B[k, j]
            end
            E[i, j] = sum_val
        end
    end
    
    # F = C * D
    Threads.@threads :static for i in 1:nj
        @inbounds for j in 1:nl
            sum_val = 0.0
            @simd for k in 1:nm
                sum_val += C[i, k] * D[k, j]
            end
            F[i, j] = sum_val
        end
    end
    
    # G = E * F
    Threads.@threads :static for i in 1:ni
        @inbounds for j in 1:nl
            sum_val = 0.0
            @simd for k in 1:nj
                sum_val += E[i, k] * F[k, j]
            end
            G[i, j] = sum_val
        end
    end
    return nothing
end

# Strategy 3: Threads with dynamic scheduling
function kernel_3mm_threads_dynamic!(A::Matrix{Float64}, B::Matrix{Float64},
                                     C::Matrix{Float64}, D::Matrix{Float64},
                                     E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B
    Threads.@threads :dynamic for i in 1:ni
        @inbounds for j in 1:nj
            sum_val = 0.0
            @simd for k in 1:nk
                sum_val += A[i, k] * B[k, j]
            end
            E[i, j] = sum_val
        end
    end
    
    # F = C * D
    Threads.@threads :dynamic for i in 1:nj
        @inbounds for j in 1:nl
            sum_val = 0.0
            @simd for k in 1:nm
                sum_val += C[i, k] * D[k, j]
            end
            F[i, j] = sum_val
        end
    end
    
    # G = E * F
    Threads.@threads :dynamic for i in 1:ni
        @inbounds for j in 1:nl
            sum_val = 0.0
            @simd for k in 1:nj
                sum_val += E[i, k] * F[k, j]
            end
            G[i, j] = sum_val
        end
    end
    return nothing
end

# Strategy 4: Tiled/Blocked for cache optimization
function kernel_3mm_tiled!(A::Matrix{Float64}, B::Matrix{Float64},
                           C::Matrix{Float64}, D::Matrix{Float64},
                           E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64};
                           tile_size::Int=32)
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    ts = tile_size
    
    # E = A * B (tiled)
    Threads.@threads :static for ii in 1:ts:ni
        i_end = min(ii + ts - 1, ni)
        @inbounds for jj in 1:ts:nj
            j_end = min(jj + ts - 1, nj)
            for kk in 1:ts:nk
                k_end = min(kk + ts - 1, nk)
                for i in ii:i_end
                    for j in jj:j_end
                        sum_val = E[i, j]
                        @simd for k in kk:k_end
                            sum_val += A[i, k] * B[k, j]
                        end
                        E[i, j] = sum_val
                    end
                end
            end
        end
    end
    
    # F = C * D (tiled)
    Threads.@threads :static for ii in 1:ts:nj
        i_end = min(ii + ts - 1, nj)
        @inbounds for jj in 1:ts:nl
            j_end = min(jj + ts - 1, nl)
            for kk in 1:ts:nm
                k_end = min(kk + ts - 1, nm)
                for i in ii:i_end
                    for j in jj:j_end
                        sum_val = F[i, j]
                        @simd for k in kk:k_end
                            sum_val += C[i, k] * D[k, j]
                        end
                        F[i, j] = sum_val
                    end
                end
            end
        end
    end
    
    # G = E * F (tiled)
    Threads.@threads :static for ii in 1:ts:ni
        i_end = min(ii + ts - 1, ni)
        @inbounds for jj in 1:ts:nl
            j_end = min(jj + ts - 1, nl)
            for kk in 1:ts:nj
                k_end = min(kk + ts - 1, nj)
                for i in ii:i_end
                    for j in jj:j_end
                        sum_val = G[i, j]
                        @simd for k in kk:k_end
                            sum_val += E[i, k] * F[k, j]
                        end
                        G[i, j] = sum_val
                    end
                end
            end
        end
    end
    return nothing
end

# Strategy 5: BLAS-based implementation
function kernel_3mm_blas!(A::Matrix{Float64}, B::Matrix{Float64},
                          C::Matrix{Float64}, D::Matrix{Float64},
                          E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    # E = A * B
    mul!(E, A, B)
    # F = C * D
    mul!(F, C, D)
    # G = E * F
    mul!(G, E, F)
    return nothing
end

# Strategy 6: Task-based parallelism
function kernel_3mm_tasks!(A::Matrix{Float64}, B::Matrix{Float64},
                           C::Matrix{Float64}, D::Matrix{Float64},
                           E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    chunk_size_e = max(1, div(ni, 2 * Threads.nthreads()))
    chunk_size_f = max(1, div(nj, 2 * Threads.nthreads()))
    
    # Phase 1: E = A * B and F = C * D can run concurrently
    @sync begin
        # E = A * B
        for i_start in 1:chunk_size_e:ni
            i_end = min(i_start + chunk_size_e - 1, ni)
            Threads.@spawn begin
                @inbounds for i in i_start:i_end
                    for j in 1:nj
                        sum_val = 0.0
                        @simd for k in 1:nk
                            sum_val += A[i, k] * B[k, j]
                        end
                        E[i, j] = sum_val
                    end
                end
            end
        end
        
        # F = C * D
        for i_start in 1:chunk_size_f:nj
            i_end = min(i_start + chunk_size_f - 1, nj)
            Threads.@spawn begin
                @inbounds for i in i_start:i_end
                    for j in 1:nl
                        sum_val = 0.0
                        @simd for k in 1:nm
                            sum_val += C[i, k] * D[k, j]
                        end
                        F[i, j] = sum_val
                    end
                end
            end
        end
    end
    
    # Phase 2: G = E * F (depends on E and F)
    @sync begin
        for i_start in 1:chunk_size_e:ni
            i_end = min(i_start + chunk_size_e - 1, ni)
            Threads.@spawn begin
                @inbounds for i in i_start:i_end
                    for j in 1:nl
                        sum_val = 0.0
                        @simd for k in 1:nj
                            sum_val += E[i, k] * F[k, j]
                        end
                        G[i, j] = sum_val
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
    "sequential"      => kernel_3mm_seq!,
    "threads_static"  => kernel_3mm_threads_static!,
    "threads_dynamic" => kernel_3mm_threads_dynamic!,
    "tiled"           => kernel_3mm_tiled!,
    "blas"            => kernel_3mm_blas!,
    "tasks"           => kernel_3mm_tasks!
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
function run_benchmark(kernel!,
                       A::Matrix{Float64}, B::Matrix{Float64},
                       C::Matrix{Float64}, D::Matrix{Float64},
                       E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64};
                       warmup::Int=5, iterations::Int=10)
    times = Float64[]
    
    # Warmup runs (not timed)
    for _ in 1:warmup
        fill!(E, 0.0)
        fill!(F, 0.0)
        fill!(G, 0.0)
        kernel!(A, B, C, D, E, F, G)
    end
    GC.gc()
    
    # Timed runs
    for _ in 1:iterations
        fill!(E, 0.0)
        fill!(F, 0.0)
        fill!(G, 0.0)
        t = @elapsed kernel!(A, B, C, D, E, F, G)
        push!(times, t * 1000)  # Convert to ms
    end
    
    return times
end

#=============================================================================
 Verification
=============================================================================#
function verify_result(G_ref::Matrix{Float64}, G_test::Matrix{Float64}, 
                       ni::Int, nj::Int, nk::Int, nl::Int, nm::Int)
    max_error = maximum(abs.(G_ref .- G_test))
    # Scale-aware tolerance: BLAS uses different accumulation order
    # Three matrix multiplications compound the error
    scale_factor = sqrt(Float64(ni) * Float64(nj) * Float64(nk) * Float64(nl) * Float64(nm))
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
            println("3MM Benchmark Runner")
            println("Usage: julia -t N run_3mm.jl [OPTIONS]")
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
    ni, nj, nk, nl, nm = params.ni, params.nj, params.nk, params.nl, params.nm
    flops = flops_3mm(ni, nj, nk, nl, nm)
    memory_mb = memory_3mm(ni, nj, nk, nl, nm) / 1024^2
    
    # Determine strategies to run
    if strategies_arg == "all"
        strategies = STRATEGY_ORDER
    else
        strategies = [strip(String(s)) for s in split(strategies_arg, ",")]
    end
    
    # Print header
    println("="^70)
    println("3MM BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset (ni=$ni, nj=$nj, nk=$nk, nl=$nl, nm=$nm)")
    @printf("Memory: %.2f MB\n", memory_mb)
    @printf("FLOPs: %d (%.2f GFLOPs)\n", flops, flops / 1e9)
    println()
    
    # Allocate matrices
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E = Matrix{Float64}(undef, ni, nj)
    F = Matrix{Float64}(undef, nj, nl)
    G = Matrix{Float64}(undef, ni, nl)
    G_ref = Matrix{Float64}(undef, ni, nl)
    
    # Initialize
    init_3mm!(A, B, C, D, E, F, G)
    
    # Compute reference result
    E_ref = zeros(ni, nj)
    F_ref = zeros(nj, nl)
    fill!(G_ref, 0.0)
    kernel_3mm_seq!(A, B, C, D, E_ref, F_ref, G_ref)
    
    # Run benchmarks and collect results
    results = Dict{String, BenchmarkResult}()
    
    for strat in strategies
        if !haskey(ALL_STRATEGIES, strat)
            println("Unknown strategy: $strat")
            continue
        end
        
        kernel! = ALL_STRATEGIES[strat]
        times = run_benchmark(kernel!, A, B, C, D, E, F, G;
                              warmup=warmup, iterations=iterations)
        
        # Verify
        fill!(E, 0.0)
        fill!(F, 0.0)
        fill!(G, 0.0)
        kernel!(A, B, C, D, E, F, G)
        err, passed = do_verify ? verify_result(G_ref, G, ni, nj, nk, nl, nm) : (0.0, true)
        
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
        filepath = "results/3mm_$(dataset)_$(timestamp).csv"
        
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
                
                # @printf(io, "3mm,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
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
