# Manual Timing vs @time/@btime: Design Rationale

## The Question

Why does Julia PolyBench use manual `time_ns()` timing instead of:
- `@time` (built-in Julia macro)
- `@btime` / `@benchmark` (from BenchmarkTools.jl)

## TL;DR

| Aspect | Manual `time_ns()` | `@time` | `@btime` |
|--------|-------------------|---------|----------|
| JIT control | Explicit separation | Included in first call | Handled automatically |
| Overhead | ~50ns | ~10us (I/O) | ~100us (setup) |
| Iteration count | Fixed (predictable) | 1 | Adaptive (unpredictable) |
| SLURM compatibility | Excellent | Poor | Poor |
| GC control | Explicit | None | Automatic |
| Reset between runs | Manual | N/A | Via `setup=` |
| Statistical output | Customizable | None | Fixed format |
| Dependencies | None | None | BenchmarkTools.jl |

**Verdict:** Manual timing is superior for HPC cluster deployment with fixed-duration jobs.

---

## Detailed Analysis

### 1. `@time` Macro

```julia
julia> @time sum(rand(1000000))
  0.003421 seconds (2 allocations: 7.629 MiB)
499766.5433109988
```

**What @time does:**
1. Measures wall-clock time (including compilation on first call)
2. Reports allocations
3. Prints to stdout

**Problems for benchmarking:**

```julia
# Problem 1: JIT included in first call
julia> @time expensive_function(x)  
  0.523421 seconds (50k allocations)  # Includes JIT compilation!

julia> @time expensive_function(x)
  0.003421 seconds (2 allocations)    # Actual runtime
```

```julia
# Problem 2: I/O overhead in measurement
function time_with_macro()
    @time kernel!(args...)  # Prints to stdout = I/O = ~10us overhead
end

# For fast kernels (<1ms), this overhead is significant
```

```julia
# Problem 3: No statistical aggregation
# You get ONE number. Is it representative? Who knows.
```

### 2. `@btime` / `@benchmark` (BenchmarkTools.jl)

```julia
using BenchmarkTools

julia> @btime sum($x)
  449.162 us (0 allocations: 0 bytes)

julia> @benchmark sum($x)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min ... max):  447.300 us ... 892.100 us
 Time  (median):       449.200 us
 Time  (mean +/- std): 453.912 us +/- 18.234 us
```

**What BenchmarkTools does:**
1. Automatic warmup until stable
2. Adaptive iteration count
3. Statistical analysis (min, median, mean, std)
4. Handles interpolation with `$` for globals

**Problems for HPC benchmarking:**

```julia
# Problem 1: Unpredictable duration
@benchmark expensive_kernel($A, $B)
# Could run 100 iterations or 10000 - adaptive based on variance
# SLURM job might timeout or waste allocation
```

```julia
# Problem 2: No control over reset between iterations
# If kernel modifies data in-place, subsequent iterations
# operate on corrupted state

# Workaround exists but is clunky:
@benchmark kernel!($C, $A, $B) setup=(C = copy(C_orig))
# This copies before EVERY iteration - significant overhead for large arrays
```

```julia
# Problem 3: The $ interpolation trap
x = rand(1000, 1000)
@btime sum(x)     # WRONG: measures global lookup overhead
@btime sum($x)    # RIGHT: interpolates value

# Easy to forget, silent performance degradation
```

```julia
# Problem 4: Fixed statistical output
# BenchmarkTools decides what stats to report
# Can't easily add confidence intervals or Amdahl analysis
```

### 3. Manual `time_ns()` Approach

```julia
function benchmark_kernel(kernel!, reset!, args...; warmup=5, iterations=10)
    # Phase 1: Explicit warmup (JIT compilation)
    for _ in 1:warmup
        reset!()
        kernel!(args...)
    end
    
    # Phase 2: Explicit GC
    GC.gc()
    
    # Phase 3: Allocation check
    reset!()
    allocs = @allocated kernel!(args...)
    
    # Phase 4: Timed iterations with reset
    times = Float64[]
    for _ in 1:iterations
        reset!()
        t0 = time_ns()
        kernel!(args...)
        t1 = time_ns()
        push!(times, Float64(t1 - t0))
    end
    
    return times
end
```

**Advantages:**

```julia
# Advantage 1: Predictable SLURM duration
# warmup=5, iterations=10 -> exactly 15 kernel executions
# Job duration = 15 * kernel_time + overhead
# Can accurately set #SBATCH --time=
```

```julia
# Advantage 2: Proper reset between iterations
reset!() = begin
    fill!(tmp, 0.0)
    copy!(D, D_orig)
end
# Each timed iteration starts from identical state
```

```julia
# Advantage 3: Minimal measurement overhead
# time_ns() is ~50 nanoseconds
# No I/O, no formatting, just clock read
# For 1ms kernel: 0.005% overhead
# For @time: ~1% overhead (I/O)
```

```julia
# Advantage 4: Full statistical control
times = benchmark_kernel(...)

# Compute whatever stats you need:
min_t = minimum(times)
med_t = median(times)
ci = compute_confidence_interval(times, 0.95)
# Feed into Amdahl analysis
# Export to custom CSV format
```

```julia
# Advantage 5: Zero external dependencies
# BenchmarkTools.jl must be installed and loaded
# On HPC clusters, module availability varies
# time_ns() is Base Julia - always available
```

---

## Why Not Both?

You might ask: "Can we use BenchmarkTools for development and manual timing for production?"

**Answer:** Yes, but consistency matters for scientific computing.

The issue is that BenchmarkTools and manual timing can give slightly different results:
- Different warmup strategies
- Different GC timing  
- Different iteration counts

For a paper comparing Julia vs OpenMP, you want ONE methodology applied consistently across all measurements.

---

## The Manual Timing Pipeline in Detail

### Phase 1: Warmup (JIT Separation)

```
First invocation of kernel!(args...):
+------------------+----------------------+
|  LLVM Compile    |  Execute Machine Code|
|  (type inference,|  (actual computation)|
|   IR generation, |                      |
|   optimization)  |                      |
+------------------+----------------------+
       ~50-500ms          ~1-100ms

Subsequent invocations:
+----------------------+
|  Execute Machine Code|  <- Cached, no compilation
+----------------------+
         ~1-100ms
```

The warmup phase ensures all code paths are compiled before measurement begins.

Why multiple warmup iterations?
- First run: Compiles main code path
- Subsequent runs: May trigger compilation of edge cases, branch targets
- 5 iterations is typically sufficient

### Phase 2: GC Collection

```julia
GC.gc()  # Force garbage collection
```

Purpose:
- Clear any allocations from warmup phase
- Start measurement with clean heap
- Reduce probability of GC pause during timed runs

Without this:
```
Warmup creates garbage -> GC triggers during measurement -> Outlier timing
```

### Phase 3: Allocation Check

```julia
allocs = @allocated kernel!(args...)
```

Zero allocations in hot path is CRITICAL for performance:
- Each allocation = malloc() call = ~100ns minimum
- Allocations trigger GC = unpredictable pauses
- Type instability causes allocations

This check verifies your kernel is properly optimized.

### Phase 4: Timed Iterations

```julia
for _ in 1:iterations
    reset!()  # Restore initial state
    
    t0 = time_ns()       # Read clock (TSC register)
    kernel!(args...)     # Execute kernel
    t1 = time_ns()       # Read clock again
    
    push!(times, t1 - t0)
end
```

Key points:
- `time_ns()` uses CPU timestamp counter (TSC) - hardware precision
- ~50ns overhead per timing call
- Reset between iterations ensures reproducibility

---

## Statistical Considerations

### Why Report Multiple Statistics?

| Statistic | Use Case | Meaning |
|-----------|----------|---------|
| Minimum | GFLOP/s calculation | Best-case, no interference |
| Median | Implementation comparison | Typical behavior |
| Mean | Workload estimation | Average including outliers |
| Std Dev | Stability assessment | Variability measure |
| 95% CI | Significance testing | "Are A and B really different?" |

### Why Minimum for GFLOP/s?

```
GFLOP/s = FLOPs / time

Using minimum time gives PEAK throughput:
- No OS scheduling interference
- No GC pauses
- No cache misses from other processes

This is what the hardware CAN achieve.
Mean/median include noise - not representative of capability.
```

### Why Median for Comparison?

```
Comparing Strategy A vs Strategy B:

If mean(A) < mean(B), but median(A) > median(B):
- A has lower outliers pulling mean down
- B is actually faster in typical use

Median is robust to outliers - better for comparison.
```

---

## Implementation Checklist

When implementing manual timing:

1. [ ] Separate warmup from measurement
2. [ ] Call GC.gc() before timed runs
3. [ ] Check allocations with @allocated
4. [ ] Reset state between iterations
5. [ ] Use time_ns() for precision
6. [ ] Store all samples for statistical analysis
7. [ ] Report minimum (for GFLOP/s) and median (for comparison)
8. [ ] Calculate confidence intervals for significance testing

---

## Conclusion

Manual `time_ns()` timing is chosen for Julia PolyBench because:

1. **Predictable duration** - Essential for SLURM job scheduling
2. **Explicit JIT control** - Warmup is separated from measurement
3. **Proper state reset** - Each iteration starts clean
4. **Minimal overhead** - ~50ns vs ~10us for @time
5. **Full statistical control** - Custom metrics, CI, Amdahl analysis
6. **Zero dependencies** - Works on any Julia installation
7. **Scientific reproducibility** - Same methodology for all benchmarks

The trade-off is more boilerplate code, but for HPC benchmarking where reproducibility and precision matter, this is the correct choice.