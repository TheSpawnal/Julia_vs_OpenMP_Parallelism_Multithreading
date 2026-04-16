module PolyBenchJulia
#=
PolyBench Julia - High Performance Computing Benchmark Suite
REFACTORED VERSION

A Julia implementation of selected PolyBench kernels for benchmarking
parallel computing strategies against OpenMP implementations.

=============================================================================
KERNELS IMPLEMENTED
=============================================================================
- 2MM: D = alpha*A*B*C + beta*D (chained matrix multiplication)
- 3MM: G = (A*B)*(C*D) (triple matrix multiplication)
- Cholesky: Cholesky decomposition
- Correlation: Correlation matrix computation
- Jacobi2D: 2D Jacobi stencil iteration
- Nussinov: RNA secondary structure prediction (dynamic programming)

=============================================================================
PARALLELIZATION STRATEGIES
=============================================================================
Each kernel implements multiple strategies:
- sequential: Baseline with SIMD optimization
- threads_static: Static thread scheduling (@threads :static)
- threads_dynamic: Dynamic thread scheduling (@threads :dynamic)
- tiled: Cache-blocked with parallel tiles
- blas: BLAS/LAPACK reference (uses internal BLAS threading)
- tasks: Task-based parallelism (@spawn)

=============================================================================
USAGE
=============================================================================
    using PolyBenchJulia
    
    # Configure BLAS (call once at startup)
    configure_blas_threads()
    
    # Get dataset parameters
    params = DATASETS_2MM["MEDIUM"]
    
    # Calculate expected FLOPs
    flops = flops_2mm(params.ni, params.nj, params.nk, params.nl)
    
    # Run benchmark
    julia -t 16 scripts/run_2mm.jl --dataset LARGE

=============================================================================
Author: SpawnAl / Falkor collaboration
Project: Scientific Parallel Computing & Multithreading with Julia
=============================================================================
=#

# Common utilities (load first)
include("common/Config.jl")
include("common/Metrics.jl")
include("common/BenchCore.jl")

# Kernel implementations
include("kernels/TwoMM.jl")
include("kernels/ThreeMM.jl")
include("kernels/Cholesky.jl")
include("kernels/Correlation.jl")
include("kernels/Jacobi2D.jl")
include("kernels/Nussinov.jl")

# Import submodules
using .Config
using .Metrics
using .BenchCore
using .TwoMM
using .ThreeMM
using .Cholesky
using .Correlation
using .Jacobi2D
using .Nussinov

# Export submodules for qualified access
export Config, Metrics, BenchCore
export TwoMM, ThreeMM, Cholesky, Correlation, Jacobi2D, Nussinov

# Re-export from Config
export configure_blas_threads, print_system_info
export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY
export DATASETS_CORRELATION, DATASETS_JACOBI2D, DATASETS_NUSSINOV
export flops_2mm, flops_3mm, flops_cholesky
export flops_correlation, flops_jacobi2d, flops_nussinov
export memory_bytes_2mm, memory_bytes_3mm

# Re-export from Metrics
export BenchmarkResult, MetricsCollector, BenchmarkRunConfig
export record!, print_results, export_csv, export_json
export compute_speedup, compute_parallel_efficiency
export is_parallel_strategy, PARALLEL_STRATEGIES, NON_PARALLEL_STRATEGIES

# Re-export from BenchCore
export TimingResult, BenchmarkConfig
export benchmark_kernel, time_kernel_ns, check_allocations
export format_time, format_bytes

# Re-export kernel strategies
export STRATEGIES_2MM, STRATEGIES_3MM
export STRATEGIES_CHOLESKY, STRATEGIES_CORRELATION
export STRATEGIES_JACOBI2D, STRATEGIES_NUSSINOV

end # module PolyBenchJulia
