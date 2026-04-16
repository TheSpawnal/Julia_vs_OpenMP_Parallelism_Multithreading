# Julia PolyBench Timing Mechanics and Multithreading Analysis

## Executive Summary

This report analyzes the time measurement methodology, JIT warmup handling, and parallelization strategies implemented in the Julia PolyBench benchmarking suite. The framework employs a principled approach to scientific benchmarking but reveals several areas requiring attention for DAS-5 deployment.

---

## 1. Timing Methodology Architecture

### 1.1 Core Timing Pipeline (BenchCore.jl)

The benchmarking framework implements a four-phase timing protocol:

```
Phase 1: WARMUP (JIT Compilation)
    |-- Run kernel N times WITHOUT timing
    |-- Ensures LLVM IR -> machine code complete
    |-- First run compiles, subsequent verify hot paths
    
Phase 2: GARBAGE COLLECTION
    |-- GC.gc() clears warmup allocations
    |-- Prevents GC pauses during measurement
    
Phase 3: ALLOCATION CHECK
    |-- @allocated macro on first timed run
    |-- Hot paths MUST show ZERO allocations
    |-- Non-zero = type instability or temporary arrays
    
Phase 4: TIMED ITERATIONS
    |-- time_ns() for nanosecond precision
    |-- Store ALL samples for statistical analysis
    |-- Report: min, median, mean, std
```

### 1.2 Critical Timing Functions

```julia
function time_kernel_ns(kernel!::Function, args...; kwargs...)::Float64
    t_start = time_ns()         # Nanosecond clock
    kernel!(args...; kwargs...) # Execute kernel
    t_end = time_ns()
    return Float64(t_end - t_start)
end
```

**Why `time_ns()` over `@time`?**

| Method | Resolution | Overhead | JIT Included |
|--------|-----------|----------|--------------|
| `@time` | ~1ms | High (IO) | Yes if first call |
| `@elapsed` | ~1us | Low | Yes if first call |
| `time_ns()` | 1ns | Minimal | No (manual control) |

The framework uses `time_ns()` because it provides the finest granularity with minimal measurement overhead, and crucially, allows separation of JIT compilation from measurement.

### 1.3 Statistical Metrics Reported

```julia
struct TimingResult
    times_ns::Vector{Float64}  # All timing samples
    allocations::Int           # Allocations in first timed run
    min_ns::Float64            # Best case (use for GFLOP/s)
    median_ns::Float64         # Typical behavior
    mean_ns::Float64           # Overall average
    std_ns::Float64            # Variability measure
end
```

**Metric Selection Guidelines:**

- **Minimum time**: Use for GFLOP/s calculation (represents best-case without OS interference)
- **Median time**: Use for comparing implementations (robust to outliers)
- **Mean/Std**: Use for understanding variability (cache effects, contention)

---

## 2. JIT Warmup Mechanics

### 2.1 Why Warmup is Critical

Julia's Just-In-Time compilation model means:

1. First invocation triggers LLVM compilation
2. Compilation time can exceed execution time for small problems
3. Subsequent calls hit cached native code

```
Example Timeline (no warmup):
+-------------------+------------------------+
|  JIT Compilation  |   Actual Execution     |
|   (50-500ms)      |      (10ms)            |
+-------------------+------------------------+
                    ^ Measurement would include JIT
                    
Example Timeline (with warmup):
Warmup: |--JIT--|--exec--|--exec--|--exec--|--exec--|
        ^ All compilation done
        
Measured: |--exec--|--exec--|--exec--|--exec--|--exec--|
          ^ Pure execution time
```

### 2.2 Warmup Configuration

```julia
BenchmarkConfig() = BenchmarkConfig(
    warmup_iterations = 5,  # Default warmup count
    timed_iterations = 10,  # Default measurement count
    check_allocs = true,    # Verify zero allocations
    verbose = false
)
```

**Warmup Count Selection:**

| Problem Size | Recommended Warmup | Rationale |
|--------------|-------------------|-----------|
| MINI | 3-5 | Fast compilation, verify code paths |
| SMALL/MEDIUM | 5 | Standard |
| LARGE | 3 | Compilation amortized over long runtime |
| EXTRALARGE | 2-3 | Runtime dominates |

### 2.3 Warmup with Reset Pattern

The framework uses a reset function to ensure consistent state:

```julia
for i in 1:warmup
    reset!()  # Reset arrays to initial state
    kernel!(args...; kernel_kwargs...)
end
```

This is critical because kernels modify data in-place. Without reset, warmup runs would operate on corrupted data.

---

## 3. @BenchmarkTools vs Manual Timing

### 3.1 Framework Comparison

The project uses **manual timing** rather than `@benchmark`:

**Manual Approach (Current):**
```julia
# Used in BenchCore.jl
for i in 1:iterations
    reset!()
    t_ns = time_kernel_ns(kernel!, args...)
    push!(times_ns, t_ns)
end
```


### 3.2 Trade-off Analysis

| Aspect | Manual Timing | @benchmark |
|--------|--------------|------------|
| Control | Full (explicit warmup, GC) | Automatic |
| Overhead | Minimal | Higher (tuning, sampling) |
| Statistical rigor | Basic (min/med/mean/std) | Advanced (confidence intervals) |
| Reset handling | Manual | Via `setup` kwarg |
| Interpolation | Not needed | `$var` required for globals |
| Execution time | Predictable | Adaptive (runs until stable) |

**Recommendation:** The current manual approach is appropriate for this project because:
1. Explicit control over warmup count
2. Predictable job duration for SLURM
3. Sufficient statistical metrics for comparison

## 4. Parallelization Strategies

### 4.1 Strategy Classification

The framework distinguishes between parallel and non-parallel strategies:

```julia
const PARALLEL_STRATEGIES = Set([
    "threads", "threads_static", "threads_dynamic",
    "tiled", "blocked", "tasks", "wavefront",
    "redblack", "parallel"
])

const NON_PARALLEL_STRATEGIES = Set([
    "sequential", "seq", "simd", "blas", "colmajor"
])
```

**Critical Distinction:**
- Parallel strategies: Efficiency = Speedup / threads (Amdahl measure)
- Non-parallel strategies: Efficiency = N/A (algorithmic speedup)

### 4.2 Strategy Implementations (2MM/3MM)

#### Sequential (Baseline)

```julia
function kernel_2mm_seq!(alpha, beta, A, B, tmp, C, D)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B (column-major: j outer)
    @inbounds for j in 1:nj
        for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    # ... second multiplication follows
end
```

**Key optimizations:**
- Column-major loop order (j outer for Julia arrays)
- Scalar hoisting (`b_kj`) outside SIMD loop
- `@inbounds` removes bounds checking
- `@simd` enables vectorization

#### Threads Static

```julia
function kernel_2mm_threads_static!(alpha, beta, A, B, tmp, C, D)
    Threads.@threads :static for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
end
```

**Characteristics:**
- `:static` scheduler divides iterations evenly at compile time
- Thread-to-iteration mapping is deterministic
- Good for uniform workloads (matrix operations)

#### Threads Dynamic

```julia
Threads.@threads :dynamic for i in 1:ni
    # ... work ...
end
```

**Characteristics:**
- Work-stealing scheduler
- Better for non-uniform workloads
- Higher overhead than static
- Thread migration possible (task may move between threads)

#### Tiled/Blocked

```julia
function kernel_2mm_tiled!(alpha, beta, A, B, tmp, C, D; tile_size::Int=64)
    ts = tile_size
    
    @threads :static for jj in 1:ts:nj
        j_end = min(jj + ts - 1, nj)
        @inbounds for kk in 1:ts:nk
            k_end = min(kk + ts - 1, nk)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        b_kj = alpha * B[k, j]
                        @simd for i in ii:i_end
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
end
```

**Cache optimization rationale:**
- L2 cache: ~256KB per core on Xeon E5-2630-v3
- Tile 64x64 of Float64 = 32KB (fits in L1/L2)
- Reduces cache misses by reusing loaded data

#### BLAS Reference

```julia
function kernel_2mm_blas!(alpha, beta, A, B, tmp, C, D)
    mul!(tmp, A, B, alpha, 0.0)   # tmp = alpha * A * B
    mul!(D, tmp, C, 1.0, beta)   # D = tmp * C + beta * D
end
```

**Critical note:** BLAS uses its own internal threading model (OpenBLAS/MKL). The speedup from BLAS is NOT parallel efficiency - it's algorithmic efficiency from optimized assembly + vectorization.

#### Task-Based

```julia
function kernel_2mm_tasks!(alpha, beta, A, B, tmp, C, D; num_tasks::Int=nthreads())
    chunk_j = cld(nj, num_tasks)
    @sync begin
        for t in 1:num_tasks
            j_start = (t - 1) * chunk_j + 1
            j_end = min(t * chunk_j, nj)
            @spawn begin
                @inbounds for j in j_start:j_end
                    # ... column computation ...
                end
            end
        end
    end
end
```

**Characteristics:**
- Explicit task granularity control
- Good for heterogeneous workloads
- `@sync` ensures all tasks complete before return

### 4.3 BLAS Thread Configuration

Critical insight from `Config.jl`:

```julia
function configure_blas_threads(;verbose=false, for_blas_benchmark=false)
    if for_blas_benchmark
        BLAS.set_num_threads(Sys.CPU_THREADS)  # BLAS uses all cores
    elseif Threads.nthreads() > 1
        BLAS.set_num_threads(1)  # Avoid oversubscription
    else
        BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
    end
end
```

**Why disable BLAS threading when Julia is multi-threaded?**

Without this:
- Julia uses 16 threads
- BLAS uses 16 threads internally
- 16 x 16 = 256 threads competing for 16 cores
- Massive context switching overhead

---

## 5. Metrics Calculation

### 5.1 Speedup Formula

```julia
function compute_speedup(baseline_time_ns, current_time_ns)
    return baseline_time_ns / current_time_ns
end
```

```
S_p = T_1 / T_p

Where:
  T_1 = Sequential baseline time
  T_p = Parallel execution time with p threads
```

### 5.2 Parallel Efficiency

```julia
function compute_parallel_efficiency(strategy, speedup, threads)
    if !is_parallel_strategy(strategy)
        return NaN  # Not meaningful for BLAS, sequential
    end
    return (speedup / threads) * 100.0
end
```

```
E_p = S_p / p = T_1 / (p * T_p)

Interpretation:
  E_p = 100%  -> Perfect linear scaling
  E_p < 100%  -> Sublinear (Amdahl's law, overhead)
  E_p > 100%  -> Superlinear (rare, cache effects)
```

### 5.3 GFLOP/s Calculation

```julia
gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
```

Using minimum time for GFLOP/s provides the peak achievable throughput, representing the kernel's best performance without OS interference.

### 5.4 FLOP Counting (Config.jl)

```julia
# 2MM: D = alpha*A*B*C + beta*D
function flops_2mm(ni, nj, nk, nl)
    flops_tmp = 2.0 * ni * nj * nk     # tmp = A * B (multiply + add)
    flops_d = 2.0 * ni * nl * nj       # D = tmp * C
    flops_scale = Float64(ni * nl)     # beta * D
    return flops_tmp + flops_d + flops_scale
end

# 3MM: G = (A*B)*(C*D)
function flops_3mm(ni, nj, nk, nl, nm)
    flops_e = 2.0 * ni * nj * nk       # E = A * B
    flops_f = 2.0 * nj * nl * nm       # F = C * D
    flops_g = 2.0 * ni * nl * nj       # G = E * F
    return flops_e + flops_f + flops_g
end
```

---

## 6. Identified Issues and Recommendations

### 6.1 Efficiency Calculation Bug (CRITICAL)

From the scaling study output:
```
threads,1,261.071,...,4.21,421.3,PASS
```

**Problem:** Efficiency of 421% for 1 thread is mathematically impossible.

**Root cause:** The CSV export is not properly populating `is_parallel` column, and efficiency is being calculated as `speedup * 100` for all strategies instead of using `NaN` for non-parallel.

**Fix required in Metrics.jl export:**
```julia
# Current (broken)
efficiency = compute_efficiency(strategy, speedup, nthreads)

# Should be
efficiency = is_parallel_strategy(strategy) ? 
             (speedup / threads) * 100.0 : NaN
```

### 6.2 CSV Header Mismatch

The CSV output shows:
```csv
threads,is_parallel,...
1,261.071,...
```

The `is_parallel` column contains timing data instead of boolean flag. This corrupts downstream analysis.

### 6.3 Recommended Improvements

#### A. Add Proper Time Unit Handling
```julia
struct TimingResult
    times_ns::Vector{Float64}    # Raw nanoseconds
    times_ms::Vector{Float64}    # Convenience: ms
    times_s::Vector{Float64}     # Convenience: seconds
end
```

#### B. Add Confidence Intervals
```julia
function compute_confidence_interval(times, confidence=0.95)
    n = length(times)
    mean_t = mean(times)
    std_t = std(times)
    t_crit = quantile(TDist(n-1), 1 - (1 - confidence)/2)
    margin = t_crit * std_t / sqrt(n)
    return (mean_t - margin, mean_t + margin)
end
```

#### C. Amdahl's Law Overlay
```julia
function amdahl_speedup(f, p)
    return 1 / ((1 - f) + f/p)
end

# Estimate parallel fraction from measurements
function estimate_parallel_fraction(speedups, threads)
    # Least squares fit to Amdahl model
    # ...
end
```

---

## 7. DAS-5 Deployment Considerations

### 7.1 Node Specifications

| Component | Value |
|-----------|-------|
| CPU | Dual Intel Xeon E5-2630-v3 |
| Cores/node | 16 (8 per socket) |
| Hyperthreading | 32 logical (avoid for compute) |
| Memory | 64 GB |
| L2 cache | 256 KB/core |
| L3 cache | 20 MB shared |

### 7.2 Thread Count Recommendation

**Use 16 threads maximum per node:**

```
Physical cores = 16
Hyperthreads = 32 (do NOT use for FP-bound)

Reason: E5-2630-v3 has ONE FPU per core.
Two hyperthreads share the same FPU.
For DGEMM-heavy workloads, using 32 threads
causes FPU contention and REDUCES performance.
```

### 7.3 SLURM Configuration Template

```bash
#!/bin/bash
#SBATCH --job-name=julia_bench
#SBATCH --output=julia_bench_%j.out
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --constraint=cpunode

# Source environment
source ~/.bashrc
module load julia/1.11

# Set threads explicitly
export JULIA_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1  # Disable BLAS threading

# Run benchmark
julia --threads=16 scripts/run_2mm.jl --dataset LARGE --output csv
```

---

## 8. Summary: Time Measurement State of the Art

The current framework implements:

| Aspect | Implementation | Status |
|--------|---------------|--------|
| JIT separation | Explicit warmup phase | Correct |
| Timing precision | `time_ns()` (nanoseconds) | Correct |
| GC isolation | `GC.gc()` between phases | Correct |
| Allocation tracking | `@allocated` on first run | Correct |
| Statistical metrics | min/median/mean/std | Adequate |
| BLAS thread control | Explicit configuration | Correct |
| Parallel efficiency | Conditional calculation | BUG |
| CSV export | Strategy classification | BUG |

**Priority fixes required:**
1. Efficiency calculation for non-parallel strategies (return NaN)
2. CSV column alignment (`is_parallel` field)
3. Verification tolerance scale-awareness

---

*Report generated for Julia PolyBench Benchmarking Suite*
*Project: TheSpawnal/Julia_versus_OpenMP*
