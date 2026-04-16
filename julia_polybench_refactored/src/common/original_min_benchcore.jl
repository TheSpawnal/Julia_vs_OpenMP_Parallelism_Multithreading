module BenchCore
#=
Benchmark Core Module for Julia PolyBench
REFACTORED: Proper JIT warmup, timing methodology, and allocation tracking

=============================================================================
TIMING METHODOLOGY
=============================================================================

1. WARMUP PHASE (Critical for JIT):
   - Run kernel warmup_iterations times WITHOUT timing
   - This ensures all JIT compilation is complete
   - First run compiles code, subsequent runs verify hot paths
   
2. GC COLLECTION:
   - Call GC.gc() between warmup and timed runs
   - This clears any garbage from warmup
   
3. TIMED ITERATIONS:
   - Time each iteration separately
   - Store all times for statistical analysis
   
4. REPORTING:
   - Minimum time: Best case, no OS interference (use for GFLOP/s)
   - Median time: Typical behavior
   - Mean time: Overall average including outliers
   - Standard deviation: Variability measure

5. ALLOCATION CHECKING:
   - Hot paths should have ZERO allocations
   - Use @allocated to verify
   
=============================================================================
=#

using Statistics
using Printf

export TimingResult, BenchmarkConfig
export benchmark_kernel, time_kernel_ns, check_allocations
export format_time, format_bytes

#=============================================================================
 Timing Result Structure
=============================================================================#
struct TimingResult
    times_ns::Vector{Float64}    # All timing samples
    allocations::Int             # Allocations in first timed run
    min_ns::Float64              # Minimum time
    median_ns::Float64           # Median time
    mean_ns::Float64             # Mean time
    std_ns::Float64              # Standard deviation
end

function TimingResult(times_ns::Vector{Float64}, allocations::Int)
    TimingResult(
        times_ns,
        allocations,
        minimum(times_ns),
        Statistics.median(times_ns),
        Statistics.mean(times_ns),
        length(times_ns) > 1 ? Statistics.std(times_ns) : 0.0
    )
end

#=============================================================================
 Benchmark Configuration
=============================================================================#
struct BenchmarkConfig
    warmup_iterations::Int
    timed_iterations::Int
    check_allocs::Bool
    verbose::Bool
end

BenchmarkConfig() = BenchmarkConfig(5, 10, true, false)

#=============================================================================
 High-Resolution Timing
 
 Uses time_ns() for nanosecond precision
=============================================================================#
function time_kernel_ns(kernel!::Function, args...; kwargs...)::Float64
    t_start = time_ns()
    kernel!(args...; kwargs...)
    t_end = time_ns()
    return Float64(t_end - t_start)
end

#=============================================================================
 Allocation Checking
 
 Verifies that hot path has zero allocations
=============================================================================#
function check_allocations(kernel!::Function, args...; kwargs...)::Int
    allocs = @allocated kernel!(args...; kwargs...)
    return allocs
end

#=============================================================================
 Main Benchmark Function
 
 Handles:
 - Warmup with JIT compilation
 - GC collection
 - Timed iterations
 - Allocation checking
 - Statistical analysis
=============================================================================#
function benchmark_kernel(
    kernel!::Function,
    reset!::Function,
    args...;
    warmup::Int=5,
    iterations::Int=10,
    check_allocs::Bool=true,
    verbose::Bool=false,
    kernel_kwargs...
)::TimingResult
    
    # Phase 1: Warmup (JIT compilation)
    verbose && println("  Warmup: $warmup iterations...")
    for i in 1:warmup
        reset!()  # Reset state before each run
        kernel!(args...; kernel_kwargs...)
    end
    
    # Phase 2: Garbage collection
    GC.gc()
    
    # Phase 3: Check allocations (on first timed run)
    allocations = 0
    if check_allocs
        reset!()
        allocations = check_allocations(kernel!, args...; kernel_kwargs...)
        if verbose
            if allocations > 0
                println("  WARNING: $allocations bytes allocated in hot path")
            else
                println("  Allocations: 0 (good)")
            end
        end
    end
    
    # Phase 4: Timed iterations
    verbose && println("  Timing: $iterations iterations...")
    times_ns = Float64[]
    sizehint!(times_ns, iterations)
    
    for i in 1:iterations
        reset!()  # Reset state before each timed run
        t_ns = time_kernel_ns(kernel!, args...; kernel_kwargs...)
        push!(times_ns, t_ns)
    end
    
    result = TimingResult(times_ns, allocations)
    
    if verbose
        println("  Results:")
        println("    Min:    $(format_time(result.min_ns))")
        println("    Median: $(format_time(result.median_ns))")
        println("    Mean:   $(format_time(result.mean_ns))")
        println("    Std:    $(format_time(result.std_ns))")
    end
    
    return result
end

#=============================================================================
 Simple Timing (for quick tests)
=============================================================================#
function time_kernel_simple(kernel!::Function, args...; kwargs...)
    # Single warmup
    kernel!(args...; kwargs...)
    GC.gc()
    
    # Single timed run
    t_ns = time_kernel_ns(kernel!, args...; kwargs...)
    return t_ns
end

#=============================================================================
 Formatting Utilities
=============================================================================#
function format_time(ns::Float64)::String
    if ns >= 1e9
        return @sprintf("%.3f s", ns / 1e9)
    elseif ns >= 1e6
        return @sprintf("%.3f ms", ns / 1e6)
    elseif ns >= 1e3
        return @sprintf("%.3f us", ns / 1e3)
    else
        return @sprintf("%.0f ns", ns)
    end
end

function format_bytes(bytes::Int)::String
    if bytes >= 1024^3
        return @sprintf("%.2f GB", bytes / 1024^3)
    elseif bytes >= 1024^2
        return @sprintf("%.2f MB", bytes / 1024^2)
    elseif bytes >= 1024
        return @sprintf("%.2f KB", bytes / 1024)
    else
        return @sprintf("%d B", bytes)
    end
end

end # module BenchCore