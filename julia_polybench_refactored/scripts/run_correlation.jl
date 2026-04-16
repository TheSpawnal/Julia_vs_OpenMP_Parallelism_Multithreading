#!/usr/bin/env julia
# Correlation Matrix Benchmark Runner - FIXED VERSION
# Usage: julia -t N scripts/run_correlation.jl --dataset MEDIUM

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
    "MINI" => (m=28, n=32),
    "SMALL" => (m=80, n=100),
    "MEDIUM" => (m=240, n=260),
    "LARGE" => (m=1200, n=1400),
    "EXTRALARGE" => (m=2600, n=3000)
)

# FLOPs calculation
flops_correlation(m, n) = n*m + 2*n*m + 2*n*m + n^2*m

# Strategy classification
# const THREADED_STRATEGIES = Set(["threads", "colmajor", "tiled"])

# function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)
#     if lowercase(strategy) in THREADED_STRATEGIES
#         return (speedup / max(nthreads, 1)) * 100.0
#     else
#         return speedup * 100.0
#     end
# end
const PARALLEL_STRATEGIES = Set(["threads", "colmajor", "tiled"])

function compute_efficiency(strategy::String, speedup::Float64, nthreads::Int)::Float64
    if !(lowercase(strategy) in PARALLEL_STRATEGIES)
        return NaN  # Efficiency not applicable for non-parallel strategies
    end
    return (speedup / max(nthreads, 1)) * 100.0
end
#=============================================================================
 INITIALIZATION
=============================================================================#
function init_data!(data::Matrix{Float64})
    m, n = size(data)
    @inbounds for j in 1:n
        for i in 1:m
            data[i, j] = Float64((i-1) * (j-1)) / m + Float64(i-1)
        end
    end
    return nothing
end

function verify_result(corr_ref::Matrix{Float64}, corr_test::Matrix{Float64})
    return norm(corr_ref - corr_test) / max(norm(corr_ref), 1e-10)
end

#=============================================================================
 KERNEL IMPLEMENTATIONS
 NOTE: Using col_mean/col_stddev to avoid shadowing Statistics.mean
=============================================================================#

# Strategy 1: Sequential baseline
function kernel_correlation_seq!(data::Matrix{Float64}, corr::Matrix{Float64},
                                  col_mean::Vector{Float64}, col_stddev::Vector{Float64})
    m, n = size(data)
    eps = 1e-10
    
    @inbounds for j in 1:n
        sum_val = 0.0
        for i in 1:m
            sum_val += data[i, j]
        end
        col_mean[j] = sum_val / m
    end
    
    @inbounds for j in 1:n
        sum_sq = 0.0
        for i in 1:m
            diff = data[i, j] - col_mean[j]
            sum_sq += diff * diff
        end
        col_stddev[j] = sqrt(sum_sq / m)
        if col_stddev[j] < eps
            col_stddev[j] = 1.0
        end
    end
    
    @inbounds for j in 1:n
        inv_std = 1.0 / (sqrt(Float64(m)) * col_stddev[j])
        for i in 1:m
            data[i, j] = (data[i, j] - col_mean[j]) * inv_std
        end
    end
    
    @inbounds for j in 1:n
        corr[j, j] = 1.0
        for i in 1:(j-1)
            sum_val = 0.0
            for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val
        end
    end
    
    return nothing
end

# Strategy 2: Threaded row-parallel
function kernel_correlation_threads!(data::Matrix{Float64}, corr::Matrix{Float64},
                                      col_mean::Vector{Float64}, col_stddev::Vector{Float64})
    m, n = size(data)
    eps = 1e-10
    
    Threads.@threads :static for j in 1:n
        sum_val = 0.0
        @inbounds @simd for i in 1:m
            sum_val += data[i, j]
        end
        col_mean[j] = sum_val / m
    end
    
    Threads.@threads :static for j in 1:n
        sum_sq = 0.0
        @inbounds @simd for i in 1:m
            diff = data[i, j] - col_mean[j]
            sum_sq += diff * diff
        end
        col_stddev[j] = sqrt(sum_sq / m)
        if col_stddev[j] < eps
            col_stddev[j] = 1.0
        end
    end
    
    Threads.@threads :static for j in 1:n
        inv_std = 1.0 / (sqrt(Float64(m)) * col_stddev[j])
        @inbounds @simd for i in 1:m
            data[i, j] = (data[i, j] - col_mean[j]) * inv_std
        end
    end
    
    Threads.@threads :static for j in 1:n
        @inbounds corr[j, j] = 1.0
        @inbounds for i in 1:(j-1)
            sum_val = 0.0
            @simd for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val
        end
    end
    
    return nothing
end

# Strategy 3: Column-major optimized with threading
function kernel_correlation_colmajor!(data::Matrix{Float64}, corr::Matrix{Float64},
                                       col_mean::Vector{Float64}, col_stddev::Vector{Float64})
    m, n = size(data)
    eps = 1e-10
    
    Threads.@threads :static for j in 1:n
        sum_val = 0.0
        sum_sq = 0.0
        @inbounds for i in 1:m
            val = data[i, j]
            sum_val += val
            sum_sq += val * val
        end
        col_mean[j] = sum_val / m
        variance = sum_sq / m - col_mean[j]^2
        col_stddev[j] = sqrt(max(variance, 0.0))
        if col_stddev[j] < eps
            col_stddev[j] = 1.0
        end
    end
    
    Threads.@threads :static for j in 1:n
        inv_std = 1.0 / (sqrt(Float64(m)) * col_stddev[j])
        @inbounds @simd for i in 1:m
            data[i, j] = (data[i, j] - col_mean[j]) * inv_std
        end
    end
    
    fill!(corr, 0.0)
    Threads.@threads :static for j in 1:n
        @inbounds corr[j, j] = 1.0
        @inbounds for i in 1:(j-1)
            sum_val = 0.0
            @simd for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val
        end
    end
    
    return nothing
end

# Strategy 4: Tiled for cache optimization (threaded)
function kernel_correlation_tiled!(data::Matrix{Float64}, corr::Matrix{Float64},
                                    col_mean::Vector{Float64}, col_stddev::Vector{Float64};
                                    tile_size::Int=64)
    m, n = size(data)
    eps = 1e-10
    ts = tile_size
    
    Threads.@threads :static for j in 1:n
        sum_val = 0.0
        sum_sq = 0.0
        @inbounds for i in 1:m
            val = data[i, j]
            sum_val += val
            sum_sq += val * val
        end
        col_mean[j] = sum_val / m
        variance = sum_sq / m - col_mean[j]^2
        col_stddev[j] = sqrt(max(variance, 0.0))
        if col_stddev[j] < eps
            col_stddev[j] = 1.0
        end
    end
    
    Threads.@threads :static for j in 1:n
        inv_std = 1.0 / (sqrt(Float64(m)) * col_stddev[j])
        @inbounds @simd for i in 1:m
            data[i, j] = (data[i, j] - col_mean[j]) * inv_std
        end
    end
    
    fill!(corr, 0.0)
    for j in 1:n
        corr[j, j] = 1.0
    end
    
    Threads.@threads :static for jj in 1:ts:n
        j_end = min(jj + ts - 1, n)
        
        for ii in 1:ts:n
            i_end = min(ii + ts - 1, n)
            
            @inbounds for j in jj:j_end
                for i in ii:min(i_end, j-1)
                    sum_val = 0.0
                    @simd for k in 1:m
                        sum_val += data[k, i] * data[k, j]
                    end
                    corr[i, j] = sum_val
                    corr[j, i] = sum_val
                end
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

function run_benchmark(kernel!, data_orig::Matrix{Float64}, corr::Matrix{Float64},
                       col_mean::Vector{Float64}, col_stddev::Vector{Float64};
                       warmup::Int=3, iterations::Int=10)
    times = Float64[]
    
    for _ in 1:warmup
        data = copy(data_orig)
        fill!(corr, 0.0)
        kernel!(data, corr, col_mean, col_stddev)
    end
    GC.gc()
    
    for _ in 1:iterations
        data = copy(data_orig)
        fill!(corr, 0.0)
        t = @elapsed kernel!(data, corr, col_mean, col_stddev)
        push!(times, t * 1000)
    end
    
    return times
end

function main()
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
    m, n = params.m, params.n
    flops = flops_correlation(m, n)
    memory_mb = (m * n + n * n) * 8 / 1024^2
    
    all_strategies = Dict(
        "sequential" => kernel_correlation_seq!,
        "threads" => kernel_correlation_threads!,
        "colmajor" => kernel_correlation_colmajor!,
        "tiled" => kernel_correlation_tiled!,
    )
    
    if strategies_arg == "all"
        strategies = ["sequential", "threads", "colmajor", "tiled"]
    else
        strategies = [strip(String(s)) for s in split(strategies_arg, ",")]
    end
    
    println("="^70)
    println("CORRELATION MATRIX BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset (m=$m, n=$n)")
    @printf("Memory: %.2f MB\n", memory_mb)
    println()
    
    data_orig = Matrix{Float64}(undef, m, n)
    init_data!(data_orig)
    
    corr = Matrix{Float64}(undef, n, n)
    col_mean = Vector{Float64}(undef, n)
    col_stddev = Vector{Float64}(undef, n)
    
    # Reference result
    data_ref = copy(data_orig)
    corr_ref = zeros(n, n)
    kernel_correlation_seq!(data_ref, corr_ref, col_mean, col_stddev)
    
    results = Dict{String, BenchmarkResult}()
    
    for strat in strategies
        if !haskey(all_strategies, strat)
            println("Unknown strategy: $strat")
            continue
        end
        
        kernel! = all_strategies[strat]
        times = run_benchmark(kernel!, data_orig, corr, col_mean, col_stddev;
                              warmup=warmup, iterations=iterations)
        
        data_test = copy(data_orig)
        fill!(corr, 0.0)
        kernel!(data_test, corr, col_mean, col_stddev)
        err = do_verify ? verify_result(corr_ref, corr) : 0.0
        passed = err < 1e-6
        
        results[strat] = BenchmarkResult(strat, times, passed, err)
        
        if do_verify && !passed
            @printf("  Verification FAILED: %s (err=%.2e)\n", strat, err)
        end
    end
    
    seq_time = haskey(results, "sequential") ? minimum(results["sequential"].times_ms) : nothing
    
    println()
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    for strat in ["sequential", "threads", "colmajor", "tiled"]
        if !haskey(results, strat)
            continue
        end
        
        r = results[strat]
        min_t = minimum(r.times_ms)
        med_t = median(r.times_ms)
        mean_t = Statistics.mean(r.times_ms)  # Fully qualified
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
    
    if output_csv
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        mkpath("results")
        filepath = "results/correlation_$(dataset)_$(timestamp).csv"
        
        open(filepath, "w") do io
            println(io, "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified")
            
            for strat in ["sequential", "threads", "colmajor", "tiled"]
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
                
                # @printf(io, "correlation,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
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