# OpenMP PolyBench Benchmark Suite

A refactored OpenMP implementation of PolyBench kernels, designed for direct comparison with Julia implementations.

## Overview

This benchmark suite provides OpenMP/C implementations of selected PolyBench kernels with multiple parallelization strategies. The output format and metrics are aligned with the Julia PolyBench implementation for unified analysis.

## Benchmarks

| Kernel | Category | Description | Parallelism |
|--------|----------|-------------|-------------|
| 2mm | Linear Algebra | D = alpha*A*B*C + beta*D | High (compute-bound) |
| 3mm | Linear Algebra | G = (A*B)*(C*D) | High (compute-bound) |
| cholesky | Solver | Cholesky decomposition | Limited (row dependencies) |
| correlation | Data Mining | Pearson correlation matrix | Medium (triangular) |
| nussinov | Dynamic Programming | RNA secondary structure | Low (anti-diagonal only) |

## Strategies Implemented

| Strategy | Description | is_parallel | Efficiency Reported |
|----------|-------------|-------------|---------------------|
| sequential | Baseline, no parallelization | false | N/A |
| threads_static | `#pragma omp parallel for schedule(static)` | true | Yes |
| threads_dynamic | `#pragma omp parallel for schedule(dynamic)` | true | Yes |
| tiled | Cache-blocked with parallel tiles | true | Yes |
| simd | SIMD vectorization | false | N/A |
| tasks | Task-based parallelism | true | Yes |
| wavefront | Anti-diagonal parallelism (Nussinov) | true | Yes |
| collapsed | Collapsed loop parallelization | true | Yes |

## Build

```bash
# Portable build (DEFAULT - works on any x86-64 CPU)
make

# For your specific machine (compile on target!)
make native

# For AVX2-capable CPUs (Haswell 2013+, Ryzen+)
make avx2

# For older CPUs (SSE4.2, Nehalem 2008+)
make sse4

# For DAS-5 cluster (Intel Xeon E5-2630-v3)
make das5

# Debug build
make debug
```

**Important**: The default `make` produces portable binaries. Use `make native` on your target machine for best performance (compiles with `-march=native`).

## Usage

```bash
# Run single benchmark
./benchmark_2mm --dataset LARGE --threads 8 --output csv

# Options
  --dataset SIZE     MINI, SMALL, MEDIUM, LARGE, EXTRALARGE
  --iterations N     Number of timed iterations (default: 10)
  --warmup N         Number of warmup iterations (default: 3)
  --threads N        Number of OpenMP threads
  --output csv       Export results to CSV

# Run all benchmarks
make run

# Quick test
make test
```

## CSV Output Format

The CSV output is aligned with Julia implementation:

```csv
benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations
2mm,LARGE,sequential,16,false,245.123,247.891,248.432,2.341,12.45,1.00,,PASS,0.00e+00,0
2mm,LARGE,threads_static,16,true,21.456,22.103,22.567,0.892,84.23,11.42,71.37,PASS,0.00e+00,0
```

Key fields:
- `is_parallel`: Whether strategy uses OpenMP threading
- `efficiency_pct`: Empty for non-parallel strategies (sequential, simd, blas)
- `speedup`: Relative to sequential baseline (T_seq / T_p)
- `verified`: PASS if max_error < tolerance

## Metrics Calculation

### Speedup
```
S_p = T_1 / T_p
```
Where T_1 is sequential time and T_p is parallel time with p threads.

### Parallel Efficiency
```
E_p = (S_p / p) * 100%
```
Only calculated for parallel strategies. Sequential, SIMD, and BLAS strategies report N/A.

### GFLOP/s
```
GFLOP/s = FLOPS / (min_time_ms / 1000) / 1e9
```
Uses minimum time for peak achievable throughput.

## Visualization

Generate plots from CSV data using the visualization script:

```bash
# Generate all plots from results
python3 scripts/visualize_benchmarks.py results/*.csv

# Specify output directory
python3 scripts/visualize_benchmarks.py results/*.csv -o ./plots

# Add title prefix
python3 scripts/visualize_benchmarks.py results/*.csv -o ./plots -t "DAS-5"

# Focus on thread scaling
python3 scripts/visualize_benchmarks.py --scaling results/scaling_*.csv

# Compare Julia and OpenMP results
python3 scripts/visualize_benchmarks.py julia/*.csv openmp/*.csv --compare
```

**Generated Plots:**
- `summary_*.png` - 4-panel dashboard (time, speedup, GFLOP/s, efficiency)
- `heatmap_*.png` - Strategy performance heatmap
- `scaling_*.png` - Thread scaling with Amdahl's Law overlays
- `comparison_*.png` - Multi-benchmark comparison
- `julia_vs_openmp_*.png` - Language comparison (when both present)

**Requirements:**
```bash
pip install pandas numpy matplotlib
```

## Comparison with Julia

Use the comparison script to analyze results from both implementations:

```bash
# Compare Julia and OpenMP results
python3 scripts/compare_benchmarks.py julia_results/*.csv openmp_results/*.csv

# Generate Markdown report
python3 scripts/compare_benchmarks.py *.csv --report comparison.md
```

## DAS-5 Cluster Usage

```bash
# Submit job
sbatch slurm/das5_benchmark.slurm 2mm LARGE

# Scaling study
sbatch slurm/das5_scaling.slurm

# Check results
cat results/*.csv
```

## Project Structure

```
openmp_polybench_refactored/
├── include/
│   ├── benchmark_common.h   # Common types, macros, FLOP formulas
│   └── metrics.h            # Metrics collection declarations
├── src/
│   ├── metrics.c            # Metrics implementation
│   ├── benchmark_2mm.c      # 2MM kernel
│   ├── benchmark_3mm.c      # 3MM kernel
│   ├── benchmark_cholesky.c # Cholesky decomposition
│   ├── benchmark_correlation.c # Correlation matrix
│   └── benchmark_nussinov.c # Nussinov RNA folding
├── scripts/
│   └── compare_benchmarks.py # Julia/OpenMP comparison tool
├── slurm/
│   ├── das5_benchmark.slurm  # Single benchmark job
│   └── das5_scaling.slurm    # Scaling study
├── results/                   # CSV output directory
├── Makefile
└── README.md
```

## References

1. PolyBench/C 4.2.1 - Ohio State University
2. "Structured Parallel Programming" - McCool, Robison, Reinders
3. OpenMP 5.0 Specification
4. Julia PolyBench Implementation - TheSpawnal/Julia_versus_OpenMP

## Author

SpawnAl / Falkor collaboration
Project: Julia vs OpenMP Performance Comparison

## Troubleshooting

### "Illegal instruction" Error
If you get this error, you're running a binary compiled for a different CPU architecture.

**Solution**: Recompile on your machine:
```bash
make clean
make          # Portable build
# OR
make native   # Optimized for your specific CPU
```

### Poor Performance
- Use `make native` for machine-specific optimizations
- Set thread count: `export OMP_NUM_THREADS=8`
- Check CPU affinity: `export OMP_PROC_BIND=close OMP_PLACES=cores`

### Verification Failures
Small numerical differences are expected with different optimization levels. Tolerance is set to 1e-6.
