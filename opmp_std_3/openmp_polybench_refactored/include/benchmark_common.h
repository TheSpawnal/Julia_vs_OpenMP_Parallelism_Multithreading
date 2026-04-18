#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>

// Dataset sizes - aligned with PolyBench specifications
typedef enum {
    DATASET_MINI = 0,
    DATASET_SMALL = 1,
    DATASET_MEDIUM = 2,
    DATASET_LARGE = 3,
    DATASET_EXTRALARGE = 4
} DatasetSize;

static const char* DATASET_NAMES[] __attribute__((unused)) = {"MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"};

// Strategy classification - aligned with Julia implementation
typedef enum {
    STRATEGY_SEQUENTIAL = 0,
    STRATEGY_THREADS_STATIC,
    STRATEGY_THREADS_DYNAMIC,
    STRATEGY_TILED,
    STRATEGY_SIMD,
    STRATEGY_TASKS,
    STRATEGY_BLAS,
    STRATEGY_WAVEFRONT,
    STRATEGY_PIPELINE,
    STRATEGY_COLLAPSED
} StrategyType;

// Strategy info structure
typedef struct {
    const char* name;
    StrategyType type;
    int is_parallel;  // 1 for parallel strategies, 0 for sequential/SIMD/BLAS
} StrategyInfo;

// Parallel strategies (use Julia threads) - efficiency is meaningful
// Non-parallel strategies (sequential, SIMD, BLAS) - efficiency is NaN
static inline int strategy_is_parallel(const char* name) {
    // Non-parallel strategies
    if (strcmp(name, "sequential") == 0 || strcmp(name, "seq") == 0) return 0;
    if (strcmp(name, "simd") == 0) return 0;
    if (strcmp(name, "blas") == 0) return 0;
    // All others are parallel
    return 1;
}

// Benchmark result structure - aligned with Julia's BenchmarkResult
typedef struct {
    char benchmark[64];
    char dataset[32];
    char strategy[64];
    int threads;
    int is_parallel;
    double min_ms;
    double median_ms;
    double mean_ms;
    double std_ms;
    double gflops;
    double speedup;
    double efficiency_pct;  // NaN for non-parallel strategies
    int verified;
    double max_error;
    int64_t allocations;  // Memory allocations (bytes)
} BenchmarkResult;

// Timing data collection
#define MAX_ITERATIONS 100

typedef struct {
    double times_ms[MAX_ITERATIONS];
    int count;
} TimingData;

// Memory allocation with alignment for SIMD
#define CACHE_LINE_SIZE 64

static inline void* aligned_alloc_safe(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "ERROR: Failed to allocate %zu bytes with %zu alignment\n", size, alignment);
        exit(1);
    }
    return ptr;
}

#define ALLOC_1D(type, n) ((type*)aligned_alloc_safe(CACHE_LINE_SIZE, (n) * sizeof(type)))
#define ALLOC_2D(type, n1, n2) ((type*)aligned_alloc_safe(CACHE_LINE_SIZE, (n1) * (n2) * sizeof(type)))
#define ALLOC_3D(type, n1, n2, n3) ((type*)aligned_alloc_safe(CACHE_LINE_SIZE, (n1) * (n2) * (n3) * sizeof(type)))

#define FREE_ARRAY(ptr) free(ptr)

// 2D array indexing (row-major)
#define IDX2(i, j, ncols) ((i) * (ncols) + (j))

// 3D array indexing (row-major)
#define IDX3(i, j, k, nj, nk) (((i) * (nj) + (j)) * (nk) + (k))

// Min/Max macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Verification tolerance
#define VERIFY_TOLERANCE 1e-6
#define VERIFY_TOLERANCE_STRICT 1e-10

// FLOP counting helpers (aligned with Julia's Config.jl)
static inline double flops_2mm(int ni, int nj, int nk, int nl) {
    // D = alpha*A*B*C + beta*D
    // tmp = A*B: 2*ni*nj*nk (multiply + add)
    // D = tmp*C: 2*ni*nl*nj
    // beta*D: ni*nl
    return 2.0 * ni * nj * nk + 2.0 * ni * nl * nj + (double)(ni * nl);
}

static inline double flops_3mm(int ni, int nj, int nk, int nl, int nm) {
    // G = (A*B)*(C*D)
    // E = A*B: 2*ni*nj*nk
    // F = C*D: 2*nj*nl*nm
    // G = E*F: 2*ni*nl*nj
    return 2.0 * ni * nj * nk + 2.0 * nj * nl * nm + 2.0 * ni * nl * nj;
}

static inline double flops_cholesky(int n) {
    // Approximately n^3/3 operations
    return (double)n * n * n / 3.0;
}

static inline double flops_correlation(int m, int n) {
    // Mean: n*m, Stddev: 2*n*m, Normalize: n*m, Correlation: n*n*m
    return (double)n * m + 2.0 * n * m + (double)n * m + (double)n * n * m;
}

static inline double flops_nussinov(int n) {
    // Approximately n^3 / 6 operations
    return (double)n * n * n / 6.0;
}

static inline double flops_jacobi2d(int tsteps, int n) {
    // 5-point stencil: 5 FLOPs per point, 2 sweeps per timestep
    int interior = n - 2;
    return (double)tsteps * 2 * interior * interior * 5.0;
}

static inline double flops_heat3d(int tsteps, int n) {
    // 7-point stencil: 13 FLOPs per point, 2 sweeps per timestep
    int interior = n - 2;
    return (double)tsteps * 2 * interior * interior * interior * 13.0;
}

// Environment setup
static inline void setup_openmp_env(void) {
    // Ensure consistent thread binding
    #ifdef _OPENMP
    omp_set_dynamic(0);  // Disable dynamic adjustment of threads
    #endif
}

#endif // BENCHMARK_COMMON_H
