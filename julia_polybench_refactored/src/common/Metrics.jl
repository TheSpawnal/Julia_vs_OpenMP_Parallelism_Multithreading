module Metrics
#=
Metrics Collection Module for Julia PolyBench Benchmarks
REFACTORED: Scientifically correct speedup and efficiency formulas

=============================================================================
SPEEDUP AND EFFICIENCY - SCIENTIFIC COMPUTING STANDARDS
=============================================================================

SPEEDUP (S_p):
  S_p = T_1 / T_p
  Where:
    T_1 = execution time with 1 thread (sequential baseline)
    T_p = execution time with p threads/processors
  
  For non-threaded optimizations (BLAS, SIMD):
    S_opt = T_seq / T_opt
    This measures algorithmic/implementation improvement, NOT parallel efficiency

PARALLEL EFFICIENCY (E_p):
  E_p = S_p / p = T_1 / (p * T_p)
  Where:
    S_p = speedup with p threads
    p = number of threads used
  
  Interpretation:
    E_p = 1.0 (100%) = perfect linear scaling
    E_p < 1.0 = sublinear scaling (Amdahl's law, overhead, contention)
    E_p > 1.0 = superlinear scaling (cache effects, rare)

IMPORTANT DISTINCTIONS:
- Parallel efficiency ONLY applies to parallel strategies
- BLAS/SIMD speedup is NOT parallel efficiency - it's algorithmic speedup
- Report BLAS speedup as "speedup" only, not "efficiency"
- Sequential baseline always has efficiency = 100% by definition

AMDAHL'S LAW:
  S_max = 1 / ((1 - f) + f/p)
  Where f = parallelizable fraction
  
=============================================================================
=#

using Printf
using Statistics
using Dates

export BenchmarkResult, MetricsCollector, BenchmarkRunConfig
export record!, print_results, export_csv, export_json
export compute_speedup, compute_parallel_efficiency
export is_parallel_strategy, PARALLEL_STRATEGIES, NON_PARALLEL_STRATEGIES

#=============================================================================
 Strategy Classification
 
 PARALLEL: Uses Julia threads, benefits from more cores
 NON_PARALLEL: Single-threaded optimizations (BLAS handles its own threading)
=============================================================================#
const PARALLEL_STRATEGIES = Set([
    "threads", "threads_static", "threads_dynamic",
    "tiled", "blocked", "tasks", "wavefront",
    "redblack", "parallel"
])

const NON_PARALLEL_STRATEGIES = Set([
    "sequential", "seq", "simd", "blas", "colmajor"
])

function is_parallel_strategy(strategy::AbstractString)::Bool
    return lowercase(String(strategy)) in PARALLEL_STRATEGIES
end

#=============================================================================
 Result Structures
=============================================================================#
struct BenchmarkResult
    benchmark::String          # Kernel name: "2mm", "3mm", etc.
    dataset::String            # Size: "MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"
    strategy::String           # Strategy name
    threads::Int               # Julia threads available at runtime
    times_ns::Vector{Float64}  # All timing samples in nanoseconds
    allocations::Int           # Memory allocations during kernel
    flops::Float64             # Total FLOPs for the computation
    verified::Bool             # Verification passed
    max_error::Float64         # Maximum verification error
end

# Convenience constructor
function BenchmarkResult(
    benchmark::String,
    dataset::String,
    strategy::String,
    threads::Int,
    times_ns::Vector{Float64},
    allocations::Int,
    flops::Float64;
    verified::Bool=true,
    max_error::Float64=0.0
)
    return BenchmarkResult(
        benchmark, dataset, strategy, threads,
        times_ns, allocations, flops, verified, max_error
    )
end

# Configuration for a benchmark run
struct BenchmarkRunConfig
    warmup_iterations::Int
    timed_iterations::Int
    verify::Bool
    export_csv::Bool
    export_json::Bool
end

BenchmarkRunConfig() = BenchmarkRunConfig(5, 10, true, true, false)

#=============================================================================
 Metrics Collector
=============================================================================#
mutable struct MetricsCollector
    results::Vector{BenchmarkResult}
    timestamp::String
    benchmark_name::String
    dataset::String
    threads::Int
    
    function MetricsCollector(;benchmark::String="", dataset::String="", threads::Int=1)
        new(
            BenchmarkResult[],
            Dates.format(now(), "yyyymmdd_HHMMSS"),
            benchmark,
            dataset,
            threads
        )
    end
end

function record!(mc::MetricsCollector, result::BenchmarkResult)
    push!(mc.results, result)
end

#=============================================================================
 Speedup Calculation
 
 Computes speedup relative to sequential baseline
 S = T_baseline / T_current
=============================================================================#
function compute_speedup(baseline_time_ns::Float64, current_time_ns::Float64)::Float64
    if current_time_ns <= 0.0 || baseline_time_ns <= 0.0
        return 1.0
    end
    return baseline_time_ns / current_time_ns
end

#=============================================================================
 Parallel Efficiency Calculation
 
 ONLY meaningful for parallel strategies
 E = S / p = (T_1 / T_p) / p
 
 Returns:
   - For parallel strategies: (speedup / threads) * 100
   - For non-parallel strategies: NaN (not applicable)
=============================================================================#
function compute_parallel_efficiency(
    strategy::AbstractString,
    speedup::Float64,
    threads::Int
)::Float64
    if !is_parallel_strategy(strategy)
        return NaN  # Efficiency not meaningful for non-parallel strategies
    end
    
    if threads <= 0
        return NaN
    end
    
    return (speedup / threads) * 100.0
end

#=============================================================================
 Print Results - Console Output
=============================================================================#
function print_results(mc::MetricsCollector)
    isempty(mc.results) && return
    
    # Find sequential baseline
    seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], mc.results)
    seq_time_ns = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    println()
    println("="^100)
    @printf("%-18s | %10s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "Std(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("="^100)
    
    for r in mc.results
        min_t = minimum(r.times_ns) / 1e6
        med_t = Statistics.median(r.times_ns) / 1e6
        mean_t = Statistics.mean(r.times_ns) / 1e6
        std_t = length(r.times_ns) > 1 ? Statistics.std(r.times_ns) / 1e6 : 0.0
        gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
        
        speedup = seq_time_ns === nothing ? 1.0 : compute_speedup(seq_time_ns, minimum(r.times_ns))
        efficiency = compute_parallel_efficiency(r.strategy, speedup, r.threads)
        
        eff_str = isnan(efficiency) ? "  N/A" : @sprintf("%6.1f", efficiency)
        status = r.verified ? "" : " [FAIL]"
        
        @printf("%-18s | %10.3f | %10.3f | %10.3f | %10.3f | %8.2f | %7.2fx | %s%s\n",
                r.strategy, min_t, med_t, mean_t, std_t, gflops, speedup, eff_str, status)
    end
    println("="^100)
    
    # Legend
    println()
    println("Legend:")
    println("  Speedup = T_sequential / T_strategy")
    println("  Eff(%) = (Speedup / Threads) * 100  [parallel strategies only]")
    println("  N/A = Efficiency not applicable (non-parallel strategy)")
end

#=============================================================================
 CSV Export - For Visualization Pipeline
=============================================================================#
function export_csv(mc::MetricsCollector, filepath::String)
    # Find sequential baseline
    seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], mc.results)
    seq_time_ns = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    open(filepath, "w") do io
        # Header
        println(io, "benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations")
        
        for r in mc.results
            min_t = minimum(r.times_ns) / 1e6
            med_t = Statistics.median(r.times_ns) / 1e6
            mean_t = Statistics.mean(r.times_ns) / 1e6
            std_t = length(r.times_ns) > 1 ? Statistics.std(r.times_ns) / 1e6 : 0.0
            gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
            
            speedup = seq_time_ns === nothing ? 1.0 : compute_speedup(seq_time_ns, minimum(r.times_ns))
            efficiency = compute_parallel_efficiency(r.strategy, speedup, r.threads)
            is_parallel = is_parallel_strategy(r.strategy)
            
            # Use empty string for NaN efficiency in CSV
            eff_str = isnan(efficiency) ? "" : @sprintf("%.2f", efficiency)
            verified_str = r.verified ? "PASS" : "FAIL"
            
            @printf(io, "%s,%s,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%.2e,%d\n",
                    r.benchmark, r.dataset, r.strategy, r.threads,
                    is_parallel, min_t, med_t, mean_t, std_t,
                    gflops, speedup, eff_str, verified_str, r.max_error, r.allocations)
        end
    end
    
    println("CSV exported: $filepath")
end

#=============================================================================
 JSON Export - For Detailed Analysis
=============================================================================#
function export_json(mc::MetricsCollector, filepath::String)
    seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], mc.results)
    seq_time_ns = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"metadata\": {")
        println(io, "    \"timestamp\": \"$(mc.timestamp)\",")
        println(io, "    \"benchmark\": \"$(mc.benchmark_name)\",")
        println(io, "    \"dataset\": \"$(mc.dataset)\",")
        println(io, "    \"threads\": $(mc.threads),")
        println(io, "    \"julia_version\": \"$(VERSION)\"")
        println(io, "  },")
        println(io, "  \"results\": [")
        
        for (idx, r) in enumerate(mc.results)
            min_t = minimum(r.times_ns) / 1e6
            med_t = Statistics.median(r.times_ns) / 1e6
            mean_t = Statistics.mean(r.times_ns) / 1e6
            std_t = length(r.times_ns) > 1 ? Statistics.std(r.times_ns) / 1e6 : 0.0
            gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
            speedup = seq_time_ns === nothing ? 1.0 : compute_speedup(seq_time_ns, minimum(r.times_ns))
            efficiency = compute_parallel_efficiency(r.strategy, speedup, r.threads)
            
            println(io, "    {")
            println(io, "      \"strategy\": \"$(r.strategy)\",")
            println(io, "      \"is_parallel\": $(is_parallel_strategy(r.strategy)),")
            println(io, "      \"min_ms\": $min_t,")
            println(io, "      \"median_ms\": $med_t,")
            println(io, "      \"mean_ms\": $mean_t,")
            println(io, "      \"std_ms\": $std_t,")
            println(io, "      \"gflops\": $gflops,")
            println(io, "      \"speedup\": $speedup,")
            if !isnan(efficiency)
                println(io, "      \"efficiency_pct\": $efficiency,")
            else
                println(io, "      \"efficiency_pct\": null,")
            end
            println(io, "      \"verified\": $(r.verified),")
            println(io, "      \"max_error\": $(r.max_error),")
            println(io, "      \"allocations\": $(r.allocations),")
            println(io, "      \"all_times_ms\": [$(join(r.times_ns ./ 1e6, ", "))]")
            print(io, "    }")
            println(io, idx < length(mc.results) ? "," : "")
        end
        
        println(io, "  ]")
        println(io, "}")
    end
    
    println("JSON exported: $filepath")
end

end # module Metrics
