#!/usr/bin/env julia
# Nussinov RNA Folding Benchmark Runner - CORRECTED VERSION
# Usage: julia -t N scripts/run_nussinov.jl --dataset MEDIUM

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
    "MINI" => 60,
    "SMALL" => 180,
    "MEDIUM" => 500,
    "LARGE" => 2500,
    "EXTRALARGE" => 5500
)

# FLOPs estimate: O(n^3) for the DP table
flops_nussinov(n) = Float64(n)^3 / 6

# Strategy classification
# wavefront, tiled are threaded; sequential, simd are not
# const THREADED_STRATEGIES = Set(["wavefront", "threads", "tiled"])

# function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)
#     if lowercase(strategy) in THREADED_STRATEGIES
#         return (speedup / max(nthreads, 1)) * 100.0
#     else
#         return speedup * 100.0
#     end
# end
const PARALLEL_STRATEGIES = Set(["wavefront", "threads", "tiled"])

function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)::Float64
    if !(lowercase(strategy) in PARALLEL_STRATEGIES)
        return NaN
    end
    return (speedup / max(nthreads, 1)) * 100.0
end
# Base pair matching (Watson-Crick pairs)
@inline function match_pair(b1::Char, b2::Char)::Int
    if (b1 == 'A' && b2 == 'U') || (b1 == 'U' && b2 == 'A') ||
       (b1 == 'G' && b2 == 'C') || (b1 == 'C' && b2 == 'G')
        return 1
    end
    return 0
end

# Initialize sequence
function init_sequence(n::Int)
    bases = ['A', 'C', 'G', 'U']
    return String([bases[((i-1) % 4) + 1] for i in 1:n])
end

#=============================================================================
 KERNEL IMPLEMENTATIONS
=============================================================================#

# Strategy 1: Sequential baseline
function kernel_nussinov_seq!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    # Process anti-diagonals (d = j - i, from 1 to n-1)
    @inbounds for d in 1:(n-1)
        for i in 1:(n-d)
            j = i + d
            
            # Option 1: Don't pair j
            best = S[i, j-1]
            
            # Option 2: Don't pair i
            best = max(best, S[i+1, j])
            
            # Option 3: Pair i and j (if they match)
            if match_pair(seq[i], seq[j]) == 1
                val = (i+1 <= j-1) ? S[i+1, j-1] + 1 : 1
                best = max(best, val)
            end
            
            # Option 4: Split at k
            for k in (i+1):(j-1)
                best = max(best, S[i, k] + S[k+1, j])
            end
            
            S[i, j] = best
        end
    end
    
    return S[1, n]
end

# Strategy 2: SIMD-optimized k-loop
function kernel_nussinov_simd!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    @inbounds for d in 1:(n-1)
        for i in 1:(n-d)
            j = i + d
            
            best = max(S[i, j-1], S[i+1, j])
            
            if match_pair(seq[i], seq[j]) == 1
                val = (i+1 <= j-1) ? S[i+1, j-1] + 1 : 1
                best = max(best, val)
            end
            
            # SIMD-friendly max reduction
            max_split = 0
            @simd for k in (i+1):(j-1)
                split_val = S[i, k] + S[k+1, j]
                max_split = max(max_split, split_val)
            end
            best = max(best, max_split)
            
            S[i, j] = best
        end
    end
    
    return S[1, n]
end

# Strategy 3: Wavefront parallel (optimized)
# Only parallelize when diagonal is long enough to overcome threading overhead
function kernel_nussinov_wavefront!(S::Matrix{Int}, seq::String)
    n = length(seq)
    min_parallel_size = 64  # Minimum diagonal length for parallelization
    
    @inbounds for d in 1:(n-1)
        diag_len = n - d
        
        if diag_len >= min_parallel_size
            # Parallel for large diagonals
            Threads.@threads :static for idx in 1:diag_len
                i = idx
                j = i + d
                
                best = max(S[i, j-1], S[i+1, j])
                
                if match_pair(seq[i], seq[j]) == 1
                    val = (i+1 <= j-1) ? S[i+1, j-1] + 1 : 1
                    best = max(best, val)
                end
                
                for k in (i+1):(j-1)
                    best = max(best, S[i, k] + S[k+1, j])
                end
                
                S[i, j] = best
            end
        else
            # Sequential for small diagonals (avoid thread overhead)
            for i in 1:diag_len
                j = i + d
                
                best = max(S[i, j-1], S[i+1, j])
                
                if match_pair(seq[i], seq[j]) == 1
                    val = (i+1 <= j-1) ? S[i+1, j-1] + 1 : 1
                    best = max(best, val)
                end
                
                for k in (i+1):(j-1)
                    best = max(best, S[i, k] + S[k+1, j])
                end
                
                S[i, j] = best
            end
        end
    end
    
    return S[1, n]
end

# Strategy 4: Tiled wavefront (threaded with chunking)
function kernel_nussinov_tiled!(S::Matrix{Int}, seq::String; chunk_size::Int=32)
    n = length(seq)
    min_parallel_size = 64
    
    @inbounds for d in 1:(n-1)
        diag_len = n - d
        
        if diag_len >= min_parallel_size
            num_chunks = cld(diag_len, chunk_size)
            
            Threads.@threads :static for chunk in 1:num_chunks
                i_start = (chunk - 1) * chunk_size + 1
                i_end = min(chunk * chunk_size, diag_len)
                
                for i in i_start:i_end
                    j = i + d
                    
                    best = max(S[i, j-1], S[i+1, j])
                    
                    if match_pair(seq[i], seq[j]) == 1
                        val = (i+1 <= j-1) ? S[i+1, j-1] + 1 : 1
                        best = max(best, val)
                    end
                    
                    for k in (i+1):(j-1)
                        best = max(best, S[i, k] + S[k+1, j])
                    end
                    
                    S[i, j] = best
                end
            end
        else
            for i in 1:diag_len
                j = i + d
                
                best = max(S[i, j-1], S[i+1, j])
                
                if match_pair(seq[i], seq[j]) == 1
                    val = (i+1 <= j-1) ? S[i+1, j-1] + 1 : 1
                    best = max(best, val)
                end
                
                for k in (i+1):(j-1)
                    best = max(best, S[i, k] + S[k+1, j])
                end
                
                S[i, j] = best
            end
        end
    end
    
    return S[1, n]
end

#=============================================================================
 BENCHMARK RUNNER
=============================================================================#

struct BenchmarkResult
    strategy::String
    times_ms::Vector{Float64}
    verified::Bool
    score::Int
end

function run_benchmark(kernel!, seq::String, n::Int;
                       warmup::Int=3, iterations::Int=10)
    times = Float64[]
    
    # Warmup
    for _ in 1:warmup
        S = zeros(Int, n, n)
        kernel!(S, seq)
    end
    GC.gc()
    
    # Timed runs
    score = 0
    for _ in 1:iterations
        S = zeros(Int, n, n)
        t = @elapsed begin
            score = kernel!(S, seq)
        end
        push!(times, t * 1000)
    end
    
    return times, score
end

function main()
    # Parse arguments
    dataset = "MEDIUM"
    strategies_arg = "all"
    iterations = 10
    warmup = 3
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
    
    n = DATASET_SIZES[dataset]
    flops = flops_nussinov(n)
    memory_mb = n * n * 4 / 1024^2  # Int = 4 bytes
    
    # Strategies
    all_strategies = Dict(
        "sequential" => kernel_nussinov_seq!,
        "simd" => kernel_nussinov_simd!,
        "wavefront" => kernel_nussinov_wavefront!,
        "tiled" => kernel_nussinov_tiled!,
    )
    
    if strategies_arg == "all"
        strategies = ["sequential", "simd", "wavefront", "tiled"]
    else
        strategies = [strip(String(s)) for s in split(strategies_arg, ",")]
    end
    
    # Header
    println("="^70)
    println("NUSSINOV RNA FOLDING BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset (n=$n)")
    @printf("Memory: %.2f MB\n", memory_mb)
    println("NOTE: Limited parallelism due to anti-diagonal dependencies")
    println()
    
    # Initialize
    seq = init_sequence(n)
    
    # Reference score
    S_ref = zeros(Int, n, n)
    ref_score = kernel_nussinov_seq!(S_ref, seq)
    println("Reference optimal score: $ref_score")
    println()
    
    # Run benchmarks
    results = Dict{String, BenchmarkResult}()
    
    for strat in strategies
        if !haskey(all_strategies, strat)
            continue
        end
        
        times, score = run_benchmark(all_strategies[strat], seq, n;
                                     warmup=warmup, iterations=iterations)
        verified = (score == ref_score)
        results[strat] = BenchmarkResult(strat, times, verified, score)
        
        if !verified
            println("  WARNING: $strat score=$score (expected $ref_score)")
        end
    end
    
    # Print results
    seq_time = haskey(results, "sequential") ? minimum(results["sequential"].times_ms) : nothing
    
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    for strat in ["sequential", "simd", "wavefront", "tiled"]
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
    
    # Performance note
    if haskey(results, "wavefront") && haskey(results, "sequential")
        wav_time = minimum(results["wavefront"].times_ms)
        seq_t = minimum(results["sequential"].times_ms)
        if wav_time > seq_t
            println()
            println("NOTE: Threaded strategies slower due to:")
            println("  - Anti-diagonal parallelism limited to n-d cells")
            println("  - Synchronization barrier after each diagonal")
            println("  - Threading overhead exceeds computation for small diagonals")
            println("  - Consider LARGE dataset for better scaling")
        end
    end
    
    # CSV output
    if output_csv
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        mkpath("results")
        filepath = "results/nussinov_$(dataset)_$(timestamp).csv"
        
        open(filepath, "w") do io
            println(io, "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified")
            
            for strat in ["sequential", "simd", "wavefront", "tiled"]
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
                
                # @printf(io, "nussinov,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
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