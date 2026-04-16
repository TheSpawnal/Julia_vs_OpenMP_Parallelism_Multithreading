# Julia PolyBench - Refactored Benchmarking Suite

High-performance Julia implementations of PolyBench kernels for parallel computing research.

## Scientific Computing Standards

### Speedup and Efficiency Formulas

```
SPEEDUP (S_p):
  S_p = T_1 / T_p
  
  Where:
    T_1 = execution time with 1 thread (sequential baseline)
    T_p = execution time with p threads

PARALLEL EFFICIENCY (E_p):
  E_p = S_p / p = T_1 / (p * T_p)
  
  Where:
    S_p = speedup with p threads
    p   = number of threads used
  
  Interpretation:
    E_p = 100% (1.0) = perfect linear scaling
    E_p < 100%       = sublinear scaling (Amdahl's law, overhead)
    E_p > 100%       = superlinear scaling (cache effects, rare)

AMDAHL'S LAW (theoretical maximum speedup):
  S_max = 1 / ((1 - f) + f/p)
  
  Where:
    f = parallelizable fraction of the workload
    p = number of processors
```

### Critical Distinctions

| Strategy Type | Speedup Meaning | Efficiency Meaning |
|---------------|-----------------|-------------------|
| Parallel (threads_*, tiled, tasks) | T_seq / T_parallel | (Speedup / threads) * 100 |
| Non-parallel (sequential, BLAS) | T_seq / T_optimized | NOT APPLICABLE |

**BLAS speedup is NOT parallel efficiency** - BLAS uses internal vectorization and its own threading model. Reporting "efficiency" for BLAS is semantically incorrect.

### FLOP Counting

```julia
# Matrix multiplication C = A * B (A is m x k, B is k x n)
FLOPs = 2 * m * n * k   # One multiply + one add per output element

# 2MM: D = alpha*A*B*C + beta*D
flops_2mm(ni, nj, nk, nl) = 2*ni*nj*nk + 2*ni*nl*nj + ni*nl

# 3MM: G = (A*B)*(C*D)
flops_3mm(ni, nj, nk, nl, nm) = 2*ni*nj*nk + 2*nj*nl*nm + 2*ni*nl*nj
```

## Quick Start

```bash
# Local development (8 threads)
julia -t 8 scripts/run_2mm.jl --dataset MEDIUM

# With CSV output
julia -t 16 scripts/run_2mm.jl --dataset LARGE --output csv

# Specific strategies only
julia -t 8 scripts/run_2mm.jl --strategies sequential,threads_static,blas
```

## DAS-5 Cluster Deployment

### Node Specifications

| Component | Value |
|-----------|-------|
| CPU | Dual Intel Xeon E5-2630-v3 |
| Cores per node | 16 (8 per socket) |
| Memory | 64 GB |
| Partition | defq |
| Max daytime job | 15 minutes |

### Thread Count: Why 16 (not 32)?

DAS-5 nodes have **16 physical cores**. While hyperthreading could provide 32 logical cores, for compute-bound benchmarks like PolyBench:

- Hyperthreading provides minimal benefit for FP-heavy workloads
- Two threads competing for the same FPU causes resource contention
- Benchmarks typically see degraded efficiency beyond physical core count

**Recommendation**: Use 16 threads maximum on a single DAS-5 node. For 32+ threads, use multi-node MPI deployment.

### SLURM Commands

```bash
# Check cluster status
sinfo
squeue -u $USER

# Interactive session (16 cores)
srun -N 1 -c 16 --time=00:15:00 --partition=defq --pty bash

# Submit scaling study
sbatch slurm/das5_scaling_study.slurm 2mm LARGE

# Submit EXTRALARGE (off-hours only)
sbatch --begin=22:00 slurm/das5_extralarge.slurm all

# Cancel jobs
scancel -u $USER
```

```

```bash
# Single benchmark, LARGE dataset, all strategies
sbatch --job-name=2mm_L \
       --output=2mm_L_%j.out \
       --time=00:15:00 \
       -N 1 --cpus-per-task=16 --partition=defq \
       --wrap='. /etc/bashrc; . /etc/profile.d/lmod.sh; \
               module load prun julia/1.11.4; \
               export JULIA_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1; \
               cd ~/Julia_versus_OpenMP/julia_polybench; \
               julia -t 16 scripts/run_2mm.jl --dataset LARGE --output csv'

# Complete scaling study (all thread counts)
for bench in 2mm 3mm cholesky correlation jacobi2d nussinov; do
    sbatch --job-name=scale_${bench} \
           --output=scale_${bench}_%j.out \
           --time=00:15:00 \
           -N 1 --cpus-per-task=16 --partition=defq \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                   module load prun julia/1.11.4; \
                   export OPENBLAS_NUM_THREADS=1; \
                   cd ~/Julia_versus_OpenMP/julia_polybench; \
                   for t in 1 2 4 8 16; do \
                       export JULIA_NUM_THREADS=\$t; \
                       julia -t \$t scripts/run_${bench}.jl --dataset LARGE --output csv; \
                   done"
done

# EXTRALARGE on weekend (4 hour limit)
sbatch --job-name=XL_all \
       --output=XL_%j.out \
       --time=04:00:00 \
       --begin=saturday \
       -N 1 --cpus-per-task=16 --partition=defq \
       --wrap='. /etc/bashrc; . /etc/profile.d/lmod.sh; \
               module load prun julia/1.11.4; \
               export JULIA_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1; \
               cd ~/Julia_versus_OpenMP/julia_polybench; \
               for b in 2mm 3mm cholesky correlation jacobi2d nussinov; do \
                   julia -t 16 scripts/run_${b}.jl --dataset EXTRALARGE --output csv; \
               done'


### Extensive Command Lines for Heavy Jobs

mkdir -p results

for bench in 2mm 3mm cholesky correlation jacobi2d nussinov; do
    sbatch --job-name="scale_${bench}" \
           --output="results/scale_${bench}_%j.out" \
           --error="results/scale_${bench}_%j.err" \
           --time=00:30:00 \
           -N 1 \
           --ntasks=1 \
           --cpus-per-task=16 \
           --exclusive \
           --partition=defq \
           -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                   module load prun julia/1.11.4; \
                   export OPENBLAS_NUM_THREADS=1; \
                   cd ~/Julia_versus_OpenMP/julia_polybench_refactored; \
                   for t in 1 2 4 8 16; do \
                       echo '=== Running with '\$t' threads ==='; \
                       export JULIA_NUM_THREADS=\$t; \
                       julia -t \$t scripts/run_${bench}.jl --dataset LARGE --output csv; \
                   done; \
                   echo '=== Scaling study complete for ${bench} ==='"
done


## Key Changes

| Change | Reason |
|--------|--------|
| `--exclusive` | No interference from other jobs during timing |
| `-C cpunode` | Ensure CPU-only node |
| `--time=00:30:00` | More buffer for 5 thread counts |
| `--ntasks=1` | Explicit single process |
| `echo` statements | Progress tracking in .out file |
| Fixed path | Point to refactored code |
| `results/` prefix | Organized output |

## How It Works
```
Job 1 (scale_2mm) on Node A:
  -> julia -t 1 run_2mm.jl ...   (sequential baseline)
  -> julia -t 2 run_2mm.jl ...   
  -> julia -t 4 run_2mm.jl ...   
  -> julia -t 8 run_2mm.jl ...   
  -> julia -t 16 run_2mm.jl ...  (full node)

Job 2 (scale_3mm) on Node B:
  -> [same pattern]
  
...6 jobs total, one per benchmark

```

## Flame Graph Profiling

### Strategy 1: Built-in Profile Flag (Recommended)

```bash
# Add --profile flag to benchmark runner
julia -t 8 scripts/run_2mm.jl --dataset LARGE --profile

# Output: results/flamegraph_2mm_LARGE_TIMESTAMP.svg
```

### Strategy 2: Manual Profiling

```julia
using Profile
using ProfileSVG

# Include kernel
include("src/kernels/TwoMM.jl")
using .TwoMM

# Setup data
ni, nj, nk, nl = 800, 900, 1100, 1200
A = rand(Float64, ni, nk)
B = rand(Float64, nk, nj)
# ... etc

# CRITICAL: Warmup first (exclude JIT from profile)
kernel_2mm_threads_static!(alpha, beta, A, B, tmp, C, D)

# Clear and profile
Profile.clear()
@profile for _ in 1:50
    fill!(tmp, 0.0)
    kernel_2mm_threads_static!(alpha, beta, A, B, tmp, C, D)
end

# Export
ProfileSVG.save("flamegraph.svg")
```

### Strategy 3: DAS-5 Profiling Job

```bash
sbatch --job-name=profile \
       --output=profile_%j.out \
       --time=00:15:00 \
       -N 1 --cpus-per-task=16 --partition=defq \
       --wrap='. /etc/bashrc; . /etc/profile.d/lmod.sh; \
               module load prun julia/1.11.4; \
               export JULIA_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1; \
               cd ~/Julia_versus_OpenMP/julia_polybench; \
               julia -t 16 scripts/run_2mm.jl --dataset LARGE --profile'
```

### Flame Graph Dependencies

```julia
# Install once
using Pkg
Pkg.add("Profile")      # Built-in
Pkg.add("ProfileSVG")   # For SVG output
Pkg.add("ProfileView")  # For interactive viewing (local only)
```

## CSV Output Format

```csv
benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations
2mm,LARGE,sequential,16,false,145.234,147.891,148.432,2.341,12.45,1.00,,PASS,0.00e+00,0
2mm,LARGE,threads_static,16,true,21.456,22.103,22.567,0.892,84.23,6.77,42.31,PASS,0.00e+00,0
2mm,LARGE,blas,16,false,8.234,8.567,8.891,0.234,219.45,17.64,,PASS,1.2e-12,0
```

Key fields:
- `is_parallel`: Whether strategy uses Julia threading
- `efficiency_pct`: Empty for non-parallel strategies (N/A)
- `max_error`: Verification error vs reference
- `allocations`: Should be 0 for optimized hot paths

## Visualization

```bash
# Generate all plots from CSV data
python3 visualize_benchmarks.py results/*.csv

# Specific output directory
python3 visualize_benchmarks.py results/*.csv --output-dir ./plots

# Scaling study visualization
python3 visualize_benchmarks.py results/scaling_study_*.csv --scaling
```

Color palette: Patagonia 80s retro pastels (coral, teal, sage, gold, lavender).

## About the .out Files

Files like `julia_bench_12345.out` are SLURM stdout captures created by:
```bash
#SBATCH --output=julia_bench_%j.out
```

They contain all console output from your job. Useful for:
- Debugging failed jobs
- Reviewing intermediate results
- Checking Julia/BLAS configuration

To exploit them:
```bash
# Parse timing info from .out files
grep "threads_static" *.out | grep -E "[0-9]+\.[0-9]+ ms"

# Find errors
grep -i "error\|fail" *.out

# Extract all speedup values
grep "Speedup" *.out
```

## Project Structure

```
julia_polybench/
├── src/
│   ├── PolyBenchJulia.jl       # Main module
│   ├── common/
│   │   ├── Config.jl           # Dataset configs, FLOP formulas
│   │   ├── Metrics.jl          # BenchmarkResult, efficiency calculation
│   │   └── BenchCore.jl        # Timing, warmup, allocation checking
│   └── kernels/
│       ├── TwoMM.jl            # 2MM implementations
│       ├── ThreeMM.jl          # 3MM implementations
│       ├── Cholesky.jl
│       ├── Correlation.jl
│       ├── Jacobi2D.jl
│       └── Nussinov.jl
├── scripts/
│   ├── run_2mm.jl              # Standalone benchmark runners
│   ├── run_3mm.jl
│   └── ...
├── slurm/
│   ├── das5_scaling_study.slurm
│   └── das5_extralarge.slurm
├── results/                    # CSV output
├── visualize_benchmarks.py     # Patagonia-style plots
└── README.md
```

## Command Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | MINI, SMALL, MEDIUM, LARGE, EXTRALARGE | MEDIUM |
| `--strategies` | Comma-separated list or "all" | all |
| `--iterations` | Timed iterations | 10 |
| `--warmup` | Warmup iterations (JIT) | 5 |
| `--no-verify` | Skip verification | false |
| `--output csv` | Export CSV | false |
| `--profile` | Generate flame graph | false |

## References

- PolyBench/C 4.2.1 Benchmark Suite
- Julia Manual: Multi-Threading
- Amdahl's Law: "Validity of the Single Processor Approach" (1967)
- Intel Xeon E5-2630-v3 Specifications
