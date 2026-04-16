module BenchCore
#=
Enhanced Benchmark Core Module for Julia PolyBench
Improvements over base BenchCore.jl:
  A. Proper time unit handling with conversion utilities
  B. Statistical confidence intervals (Student's t-distribution)
  C. Amdahl's Law analysis and parallel fraction estimation

=============================================================================
WHY THESE IMPROVEMENTS MATTER
=============================================================================

A. TIME UNIT HANDLING
   - Raw nanoseconds are error-prone (off-by-1000 mistakes)
   - Explicit unit types prevent unit confusion
   - Formatted output adapts to magnitude automatically

B. CONFIDENCE INTERVALS
   - Mean alone is meaningless without variance context
   - CI tells you "how confident" you can be in the measurement
   - Essential for claiming "Strategy A is faster than B"

C. AMDAHL'S LAW OVERLAY
   - Without Amdahl context, low efficiency looks like failure
   - Reality: 50% efficiency at 16 threads can be EXCELLENT
   - Estimating parallel fraction reveals algorithm characteristics
   - Identifies whether you're hitting Amdahl limits vs overhead

=============================================================================
=#

using Statistics
using Printf
using Distributions  # For TDist (Student's t)

export TimingResult, EnhancedTimingResult, BenchmarkConfig
export benchmark_kernel, time_kernel_ns, check_allocations
export format_time, format_bytes

# Confidence interval exports
export ConfidenceInterval, compute_ci, ci_overlaps

# Amdahl exports  
export AmdahlAnalysis, amdahl_speedup, amdahl_efficiency
export estimate_parallel_fraction, theoretical_max_speedup
export analyze_scaling, ScalingAnalysis

#=============================================================================
 SECTION A: TIME UNIT HANDLING
 
 Problem: Working with raw nanoseconds leads to errors
   - Is 1500000 ns fast or slow? (answer: 1.5ms)
   - Converting ns -> ms -> s manually is error-prone
   
 Solution: Structured time handling with automatic formatting
==============================================================================#

"""
    TimeValue

Wrapper for time measurements that preserves the raw value
while providing convenient accessors for different units.
"""
struct TimeValue
    ns::Float64  # Canonical storage: nanoseconds
end

# Constructors from different units
TimeValue(; ns::Real=0, us::Real=0, ms::Real=0, s::Real=0) = 
    TimeValue(Float64(ns) + us*1e3 + ms*1e6 + s*1e9)

# Accessors - zero allocation, just arithmetic
nanoseconds(t::TimeValue) = t.ns
microseconds(t::TimeValue) = t.ns / 1e3
milliseconds(t::TimeValue) = t.ns / 1e6
seconds(t::TimeValue) = t.ns / 1e9

# Arithmetic
Base.:+(a::TimeValue, b::TimeValue) = TimeValue(a.ns + b.ns)
Base.:-(a::TimeValue, b::TimeValue) = TimeValue(a.ns - b.ns)
Base.:/(a::TimeValue, b::Real) = TimeValue(a.ns / b)
Base.:*(a::TimeValue, b::Real) = TimeValue(a.ns * b)
Base.:*(a::Real, b::TimeValue) = TimeValue(a * b.ns)

# Comparison
Base.:<(a::TimeValue, b::TimeValue) = a.ns < b.ns
Base.isless(a::TimeValue, b::TimeValue) = a.ns < b.ns

"""
Auto-format time to appropriate unit based on magnitude.
"""
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

format_time(t::TimeValue) = format_time(t.ns)

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

#=============================================================================
 SECTION B: CONFIDENCE INTERVALS
 
 Problem: Reporting mean=100ms is incomplete
   - What if std=50ms? Then 100ms is very uncertain
   - What if std=0.1ms? Then 100ms is very reliable
   
 Solution: Confidence intervals using Student's t-distribution
   - CI = mean +/- t_crit * (std / sqrt(n))
   - t_crit depends on confidence level (typically 95%) and sample size
   
 Why Student's t (not Normal)?
   - Normal distribution requires known population variance
   - We only have sample variance from N measurements
   - Student's t accounts for this uncertainty
   - As N -> infinity, t -> Normal
==============================================================================#

"""
    ConfidenceInterval

Represents a confidence interval for a measurement.
"""
struct ConfidenceInterval
    lower::Float64      # Lower bound
    upper::Float64      # Upper bound  
    mean::Float64       # Point estimate
    confidence::Float64 # Confidence level (e.g., 0.95)
    n::Int              # Sample size
end

"""
    compute_ci(samples; confidence=0.95)

Compute confidence interval using Student's t-distribution.

Mathematical basis:
  CI = x_bar +/- t_(alpha/2, n-1) * (s / sqrt(n))
  
Where:
  x_bar = sample mean
  s = sample standard deviation  
  n = sample size
  t_(alpha/2, n-1) = critical value from t-distribution
  alpha = 1 - confidence
"""
function compute_ci(samples::Vector{Float64}; confidence::Float64=0.95)::ConfidenceInterval
    n = length(samples)
    
    if n < 2
        m = n > 0 ? samples[1] : NaN
        return ConfidenceInterval(m, m, m, confidence, n)
    end
    
    mean_val = mean(samples)
    std_val = std(samples)
    
    # Student's t critical value
    # For 95% CI: alpha = 0.05, we need t_(0.025, n-1)
    alpha = 1 - confidence
    t_dist = TDist(n - 1)  # t-distribution with n-1 degrees of freedom
    t_crit = quantile(t_dist, 1 - alpha/2)
    
    # Margin of error
    margin = t_crit * std_val / sqrt(n)
    
    return ConfidenceInterval(
        mean_val - margin,
        mean_val + margin,
        mean_val,
        confidence,
        n
    )
end

"""
    ci_overlaps(ci1, ci2)

Check if two confidence intervals overlap.
Non-overlapping CIs indicate statistically significant difference.
"""
function ci_overlaps(ci1::ConfidenceInterval, ci2::ConfidenceInterval)::Bool
    return !(ci1.upper < ci2.lower || ci2.upper < ci1.lower)
end

"""
    ci_width(ci)

Width of confidence interval (uncertainty measure).
Narrower = more precise measurement.
"""
ci_width(ci::ConfidenceInterval) = ci.upper - ci.lower

"""
    ci_relative_width(ci)

Relative width as percentage of mean.
Useful for comparing precision across different magnitudes.
"""
ci_relative_width(ci::ConfidenceInterval) = 
    ci.mean != 0 ? 100.0 * ci_width(ci) / abs(ci.mean) : Inf

#=============================================================================
 SECTION C: AMDAHL'S LAW ANALYSIS
 
 Problem: Parallel efficiency always decreases with thread count
   - 16 threads, 50% efficiency -> Is this good or bad?
   - Without context, this looks like failure
   
 Reality: Amdahl's Law defines THEORETICAL LIMITS
   - If 5% of code is serial, max speedup is 20x (not infinity)
   - At 16 threads with f=0.95, theoretical efficiency = 54%
   - So 50% measured means you're at 93% of theoretical max!
   
 Key Insight:
   S_max = 1 / ((1-f) + f/p)
   
 Where:
   f = parallelizable fraction (0 to 1)
   p = number of processors/threads
   1-f = serial fraction (the bottleneck)
   
 As p -> infinity:
   S_max -> 1/(1-f)
   
 Example: f=0.95 (95% parallel)
   p=1:  S=1.0
   p=4:  S=3.48  (not 4.0)
   p=16: S=8.67  (not 16.0)
   p=inf: S=20.0 (the hard ceiling)
==============================================================================#

"""
    amdahl_speedup(p, f)

Theoretical maximum speedup according to Amdahl's Law.

Arguments:
  p: Number of processors/threads
  f: Parallel fraction (0.0 to 1.0)

Returns:
  Maximum achievable speedup
  
Formula: S = 1 / ((1-f) + f/p)
"""
function amdahl_speedup(p::Real, f::Real)::Float64
    @assert 0 <= f <= 1 "Parallel fraction must be in [0,1], got $f"
    @assert p >= 1 "Thread count must be >= 1, got $p"
    
    serial_fraction = 1 - f
    return 1.0 / (serial_fraction + f / p)
end

# Vectorized version for plotting
amdahl_speedup(p::AbstractVector, f::Real) = [amdahl_speedup(pi, f) for pi in p]

"""
    amdahl_efficiency(p, f)

Theoretical parallel efficiency according to Amdahl's Law.

Returns:
  E = S/p as percentage (100% = perfect scaling)
"""
function amdahl_efficiency(p::Real, f::Real)::Float64
    return 100.0 * amdahl_speedup(p, f) / p
end

amdahl_efficiency(p::AbstractVector, f::Real) = [amdahl_efficiency(pi, f) for pi in p]

"""
    theoretical_max_speedup(f)

Maximum possible speedup as threads -> infinity.
This is the hard ceiling imposed by serial code.

S_max = 1/(1-f)

Example:
  f=0.90 (90% parallel) -> S_max = 10x
  f=0.95 (95% parallel) -> S_max = 20x  
  f=0.99 (99% parallel) -> S_max = 100x
"""
function theoretical_max_speedup(f::Real)::Float64
    @assert 0 <= f < 1 "Parallel fraction must be in [0,1), got $f"
    return 1.0 / (1 - f)
end

"""
    estimate_parallel_fraction(speedups, threads)

Estimate the parallel fraction f from measured speedup data.
Uses least-squares fitting to Amdahl's model.

This answers: "What fraction of my code is actually parallelizable?"

Method:
  Rearrange Amdahl: S = 1/((1-f) + f/p)
  Let s = 1/S (inverse speedup)
  Then: s = (1-f) + f/p
  This is linear in 1/p with:
    intercept = (1-f)
    slope = f
    
  So: f = slope, and we verify (1-f) = intercept
"""
function estimate_parallel_fraction(
    speedups::Vector{Float64},
    threads::Vector{Int}
)::NamedTuple{(:f, :f_std, :r_squared, :residuals), Tuple{Float64, Float64, Float64, Vector{Float64}}}
    
    n = length(speedups)
    @assert n == length(threads) "speedups and threads must have same length"
    @assert n >= 2 "Need at least 2 data points"
    
    # Transform to linear form: 1/S = (1-f) + f*(1/p)
    y = 1.0 ./ speedups           # Inverse speedup
    x = 1.0 ./ Float64.(threads)  # Inverse thread count
    
    # Linear regression: y = a + b*x
    # where a = (1-f), b = f
    x_mean = mean(x)
    y_mean = mean(y)
    
    # Slope and intercept via least squares
    numerator = sum((x .- x_mean) .* (y .- y_mean))
    denominator = sum((x .- x_mean).^2)
    
    b = numerator / denominator  # slope = f
    a = y_mean - b * x_mean      # intercept = (1-f)
    
    # Parallel fraction estimate
    f_from_slope = b
    f_from_intercept = 1 - a
    
    # Average the two estimates (they should be close if Amdahl fits well)
    f_est = (f_from_slope + f_from_intercept) / 2
    f_est = clamp(f_est, 0.0, 1.0)  # Ensure valid range
    
    # Residuals and R-squared
    y_pred = a .+ b .* x
    residuals = y .- y_pred
    ss_res = sum(residuals.^2)
    ss_tot = sum((y .- y_mean).^2)
    r_squared = 1 - ss_res / ss_tot
    
    # Standard error of f estimate
    if n > 2
        mse = ss_res / (n - 2)
        se_b = sqrt(mse / denominator)
        f_std = se_b
    else
        f_std = NaN
    end
    
    return (f=f_est, f_std=f_std, r_squared=r_squared, residuals=residuals)
end

"""
    AmdahlAnalysis

Complete Amdahl's Law analysis for a set of measurements.
"""
struct AmdahlAnalysis
    parallel_fraction::Float64      # Estimated f
    parallel_fraction_std::Float64  # Uncertainty in f
    r_squared::Float64              # Goodness of fit (1.0 = perfect)
    theoretical_max::Float64        # S_max = 1/(1-f)
    theoretical_speedups::Vector{Float64}  # At measured thread counts
    theoretical_efficiencies::Vector{Float64}
    measured_speedups::Vector{Float64}
    measured_efficiencies::Vector{Float64}
    threads::Vector{Int}
    efficiency_ratio::Vector{Float64}  # measured/theoretical
end

"""
    analyze_scaling(speedups, threads)

Comprehensive Amdahl's Law analysis.

Returns analysis showing:
  - Estimated parallel fraction
  - Theoretical limits at each thread count
  - How close measurements are to theoretical limits
"""
function analyze_scaling(
    speedups::Vector{Float64},
    threads::Vector{Int}
)::AmdahlAnalysis
    
    # Estimate parallel fraction
    est = estimate_parallel_fraction(speedups, threads)
    f = est.f
    
    # Theoretical predictions
    theo_speedups = [amdahl_speedup(t, f) for t in threads]
    theo_effs = [amdahl_efficiency(t, f) for t in threads]
    
    # Measured efficiencies
    meas_effs = [100.0 * s / t for (s, t) in zip(speedups, threads)]
    
    # How close to theoretical limit?
    eff_ratio = meas_effs ./ theo_effs
    
    return AmdahlAnalysis(
        f,
        est.f_std,
        est.r_squared,
        theoretical_max_speedup(min(f, 0.9999)),  # Avoid div by zero
        theo_speedups,
        theo_effs,
        speedups,
        meas_effs,
        threads,
        eff_ratio
    )
end

#=============================================================================
 ScalingAnalysis: High-Level Summary
==============================================================================#

struct ScalingAnalysis
    amdahl::AmdahlAnalysis
    interpretation::String
    recommendations::Vector{String}
end

"""
    interpret_scaling(analysis)

Generate human-readable interpretation of scaling behavior.
"""
function interpret_scaling(a::AmdahlAnalysis)::ScalingAnalysis
    f = a.parallel_fraction
    r2 = a.r_squared
    max_threads = maximum(a.threads)
    max_eff_ratio = minimum(a.efficiency_ratio)  # Worst case
    
    # Build interpretation
    lines = String[]
    
    # Parallel fraction interpretation
    if f >= 0.99
        push!(lines, "Excellent parallelization: $(round(f*100, digits=1))% parallel")
        push!(lines, "Theoretical max speedup: $(round(theoretical_max_speedup(f), digits=0))x")
    elseif f >= 0.95
        push!(lines, "Good parallelization: $(round(f*100, digits=1))% parallel")
        push!(lines, "Serial bottleneck limits speedup to $(round(theoretical_max_speedup(f), digits=1))x")
    elseif f >= 0.90
        push!(lines, "Moderate parallelization: $(round(f*100, digits=1))% parallel")
        push!(lines, "Significant serial portion limits scaling")
    else
        push!(lines, "Limited parallelization: $(round(f*100, digits=1))% parallel")
        push!(lines, "Serial bottleneck dominates at high thread counts")
    end
    
    # Fit quality
    if r2 >= 0.95
        push!(lines, "Amdahl model fits well (R²=$(round(r2, digits=3)))")
    elseif r2 >= 0.80
        push!(lines, "Amdahl model fits moderately (R²=$(round(r2, digits=3)))")
        push!(lines, "Some non-Amdahl effects present (cache, NUMA, etc.)")
    else
        push!(lines, "Poor Amdahl fit (R²=$(round(r2, digits=3)))")
        push!(lines, "Significant non-Amdahl effects dominate")
    end
    
    interpretation = join(lines, "\n")
    
    # Recommendations
    recs = String[]
    
    if f < 0.95
        push!(recs, "Profile serial sections - $(round((1-f)*100, digits=1))% serial is the bottleneck")
    end
    
    if max_eff_ratio < 0.7
        push!(recs, "Investigate parallel overhead - achieving only $(round(max_eff_ratio*100, digits=0))% of theoretical efficiency")
    end
    
    theo_eff_at_max = amdahl_efficiency(max_threads, f)
    if theo_eff_at_max < 50
        push!(recs, "Consider reducing thread count - theoretical efficiency at $max_threads threads is only $(round(theo_eff_at_max, digits=0))%")
    end
    
    if r2 < 0.9
        push!(recs, "Non-Amdahl effects detected - check for cache effects, NUMA, or load imbalance")
    end
    
    if isempty(recs)
        push!(recs, "Scaling behavior is optimal for this algorithm's parallel fraction")
    end
    
    return ScalingAnalysis(a, interpretation, recs)
end

#=============================================================================
 ENHANCED TIMING RESULT
 Combines all improvements: time units, CI, Amdahl context
==============================================================================#

struct EnhancedTimingResult
    # Raw data
    times_ns::Vector{Float64}
    allocations::Int
    
    # Basic stats (in nanoseconds)
    min_ns::Float64
    median_ns::Float64
    mean_ns::Float64
    std_ns::Float64
    
    # Confidence interval
    ci::ConfidenceInterval
    
    # Convenience accessors populated
    min_ms::Float64
    median_ms::Float64
    mean_ms::Float64
end

function EnhancedTimingResult(times_ns::Vector{Float64}, allocations::Int)
    ci = compute_ci(times_ns)
    
    EnhancedTimingResult(
        times_ns,
        allocations,
        minimum(times_ns),
        Statistics.median(times_ns),
        Statistics.mean(times_ns),
        length(times_ns) > 1 ? Statistics.std(times_ns) : 0.0,
        ci,
        minimum(times_ns) / 1e6,
        Statistics.median(times_ns) / 1e6,
        Statistics.mean(times_ns) / 1e6
    )
end

#=============================================================================
 CORE BENCHMARKING FUNCTIONS (same as BenchCore.jl)
==============================================================================#

struct BenchmarkConfig
    warmup_iterations::Int
    timed_iterations::Int
    check_allocs::Bool
    verbose::Bool
end

BenchmarkConfig() = BenchmarkConfig(5, 10, true, false)

function time_kernel_ns(kernel!::Function, args...; kwargs...)::Float64
    t_start = time_ns()
    kernel!(args...; kwargs...)
    t_end = time_ns()
    return Float64(t_end - t_start)
end

function check_allocations(kernel!::Function, args...; kwargs...)::Int
    allocs = @allocated kernel!(args...; kwargs...)
    return allocs
end

function benchmark_kernel(
    kernel!::Function,
    reset!::Function,
    args...;
    warmup::Int=5,
    iterations::Int=10,
    check_allocs::Bool=true,
    verbose::Bool=false,
    kernel_kwargs...
)::EnhancedTimingResult
    
    # Phase 1: Warmup
    verbose && println("  Warmup: $warmup iterations...")
    for i in 1:warmup
        reset!()
        kernel!(args...; kernel_kwargs...)
    end
    
    # Phase 2: GC
    GC.gc()
    
    # Phase 3: Allocation check
    allocations = 0
    if check_allocs
        reset!()
        allocations = check_allocations(kernel!, args...; kernel_kwargs...)
        verbose && allocations > 0 && println("  WARNING: $allocations bytes allocated")
    end
    
    # Phase 4: Timed runs
    verbose && println("  Timing: $iterations iterations...")
    times_ns = Float64[]
    sizehint!(times_ns, iterations)
    
    for i in 1:iterations
        reset!()
        t_ns = time_kernel_ns(kernel!, args...; kernel_kwargs...)
        push!(times_ns, t_ns)
    end
    
    result = EnhancedTimingResult(times_ns, allocations)
    
    if verbose
        println("  Results:")
        println("    Min:    $(format_time(result.min_ns))")
        println("    Median: $(format_time(result.median_ns))")
        println("    Mean:   $(format_time(result.mean_ns)) +/- $(format_time(result.std_ns))")
        println("    95% CI: [$(format_time(result.ci.lower)), $(format_time(result.ci.upper))]")
        println("    CI width: $(round(ci_relative_width(result.ci), digits=1))% of mean")
    end
    
    return result
end

#=============================================================================
 PRINTING UTILITIES
==============================================================================#

function print_amdahl_analysis(a::AmdahlAnalysis)
    println("\n" * "="^70)
    println("AMDAHL'S LAW ANALYSIS")
    println("="^70)
    
    @printf("Estimated parallel fraction: %.1f%% (f = %.4f)\n", 
            a.parallel_fraction * 100, a.parallel_fraction)
    @printf("Theoretical maximum speedup: %.1fx\n", a.theoretical_max)
    @printf("Model fit quality (R²): %.3f\n", a.r_squared)
    
    println("\nThread | Measured | Theoretical | Meas Eff | Theo Eff | Ratio")
    println("-"^70)
    
    for i in eachindex(a.threads)
        @printf("%6d | %8.2fx | %11.2fx | %7.1f%% | %7.1f%% | %.1f%%\n",
                a.threads[i],
                a.measured_speedups[i],
                a.theoretical_speedups[i],
                a.measured_efficiencies[i],
                a.theoretical_efficiencies[i],
                a.efficiency_ratio[i] * 100)
    end
    
    println("="^70)
end

function print_scaling_analysis(s::ScalingAnalysis)
    print_amdahl_analysis(s.amdahl)
    
    println("\nINTERPRETATION:")
    println(s.interpretation)
    
    println("\nRECOMMENDATIONS:")
    for (i, rec) in enumerate(s.recommendations)
        println("  $i. $rec")
    end
    println()
end

end # module BenchCore