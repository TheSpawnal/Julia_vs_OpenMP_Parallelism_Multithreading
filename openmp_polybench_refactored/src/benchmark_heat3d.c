/*
 * Heat-3D Benchmark - 3D Heat Equation Stencil (PolyBench)
 * 7-point stencil, explicit Euler, Jacobi double-buffer iteration.
 *
 * Physical model: du/dt = alpha * laplacian(u)
 * Discretization: 7-point finite difference stencil
 * CFL stability : alpha * dt/dx^2 = 0.125 < 1/6  (stable in 3D)
 *
 * Memory-bound kernel. Arithmetic intensity ~0.2-0.8 FLOP/byte
 * (13 FLOPs/point, 7 reads + 1 write; 16 B best-case with full reuse,
 *  64 B worst-case cold). Performance limited by memory bandwidth.
 *
 * Strategies (aligned with Julia heat3d when present):
 *   sequential       - Baseline Jacobi iteration (double-buffered)
 *   threads_static   - collapse(2) on i,j with static scheduling
 *   threads_dynamic  - collapse(2) on i,j with dynamic(4) scheduling
 *   tiled            - Spatial cache-blocking on i,j tile loops
 *   simd             - #pragma omp simd on k-loop + collapse(2) on i,j
 *   collapsed        - collapse(3) full parallelism
 *   red_black        - Gauss-Seidel red-black ordering (in-place)
 *
 * Verification:
 *   Jacobi-family strategies compare against the sequential Jacobi reference.
 *   red_black compares against a sequential red-black reference, since it
 *   follows a different numerical trajectory than Jacobi. Both use the strict
 *   VERIFY_TOLERANCE (1e-6) from benchmark_common.h.
 *
 * References:
 *   PolyBench/C 4.2.1 - heat-3d kernel
 *   "Structured Parallel Programming" - McCool, Robison, Reinders
 *   "Optimization of Stencil Computations" - Datta et al.
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>
#include <strings.h>   /* strcasecmp */

/* ------------------------------------------------------------------ */
/*  Dataset configurations (PolyBench standard sizes)                 */
/* ------------------------------------------------------------------ */

typedef struct {
    int n;
    int tsteps;
} DatasetHeat3D;

static const DatasetHeat3D DATASETS[] = {
    {10,   20},    /* MINI       */
    {20,   40},    /* SMALL      */
    {40,   100},   /* MEDIUM     */
    {120,  500},   /* LARGE      */
    {200,  1000}   /* EXTRALARGE */
};

/* Stencil coefficient (thermal_diffusivity * dt / dx^2) */
#define COEFF 0.125

/* Tile size for spatial cache blocking (L1-friendly: TILE_I*TILE_J*n doubles) */
#define TILE_I 8
#define TILE_J 8

/* ------------------------------------------------------------------ */
/*  7-point stencil                                                   */
/* ------------------------------------------------------------------ */

static inline double stencil_7pt(const double* restrict src,
                                 int i, int j, int k, int n) {
    double center = src[IDX3(i, j, k, n, n)];
    return COEFF * (src[IDX3(i+1, j, k, n, n)] - 2.0*center + src[IDX3(i-1, j, k, n, n)])
         + COEFF * (src[IDX3(i, j+1, k, n, n)] - 2.0*center + src[IDX3(i, j-1, k, n, n)])
         + COEFF * (src[IDX3(i, j, k+1, n, n)] - 2.0*center + src[IDX3(i, j, k-1, n, n)])
         + center;
}

/* ------------------------------------------------------------------ */
/*  Initialization (PolyBench standard)                               */
/* ------------------------------------------------------------------ */

static void init_array(int n, double* A, double* B) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                double val = (double)(i + j + (n - k)) * 10.0 / n;
                A[IDX3(i, j, k, n, n)] = val;
                B[IDX3(i, j, k, n, n)] = val;
            }
}

/* ------------------------------------------------------------------ */
/*  Verification: L-infinity relative error vs reference              */
/* ------------------------------------------------------------------ */

static double verify_result(int n, const double* ref, const double* out) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                int idx = IDX3(i, j, k, n, n);
                double r = ref[idx];
                double v = out[idx];
                double err = fabs(r - v);
                if (fabs(r) > 1e-15) err /= fabs(r);
                if (err > max_err) max_err = err;
            }
    return max_err;
}

/* ------------------------------------------------------------------ */
/*  Strategy 1: Sequential baseline (Jacobi double-buffer)            */
/* ------------------------------------------------------------------ */

static void kernel_heat3d_sequential(int tsteps, int n,
                                     double* restrict A,
                                     double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    B[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);

        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(B, i, j, k, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 2: threads_static                                        */
/*  collapse(2) on i,j keeps k contiguous for vectorization           */
/* ------------------------------------------------------------------ */

static void kernel_heat3d_threads_static(int tsteps, int n,
                                         double* restrict A,
                                         double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    B[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(B, i, j, k, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 3: threads_dynamic                                       */
/*  Dynamic scheduling absorbs load imbalance from OS jitter          */
/* ------------------------------------------------------------------ */

static void kernel_heat3d_threads_dynamic(int tsteps, int n,
                                          double* restrict A,
                                          double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for collapse(2) schedule(dynamic, 4)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    B[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);

        #pragma omp parallel for collapse(2) schedule(dynamic, 4)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(B, i, j, k, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 4: Tiled (spatial cache blocking)                        */
/*  Tile i,j for L1/L2 plane-neighbor reuse; k stays linear.          */
/* ------------------------------------------------------------------ */

static void kernel_heat3d_tiled(int tsteps, int n,
                                double* restrict A,
                                double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 1; ii < n-1; ii += TILE_I) {
            for (int jj = 1; jj < n-1; jj += TILE_J) {
                int i_end = MIN(ii + TILE_I, n-1);
                int j_end = MIN(jj + TILE_J, n-1);
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        for (int k = 1; k < n-1; k++)
                            B[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 1; ii < n-1; ii += TILE_I) {
            for (int jj = 1; jj < n-1; jj += TILE_J) {
                int i_end = MIN(ii + TILE_I, n-1);
                int j_end = MIN(jj + TILE_J, n-1);
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        for (int k = 1; k < n-1; k++)
                            A[IDX3(i,j,k,n,n)] = stencil_7pt(B, i, j, k, n);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 5: SIMD                                                  */
/*  #pragma omp simd on k-loop + collapse(2) thread-parallel on i,j   */
/* ------------------------------------------------------------------ */

static void kernel_heat3d_simd(int tsteps, int n,
                               double* restrict A,
                               double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++) {
                #pragma omp simd
                for (int k = 1; k < n-1; k++)
                    B[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);
            }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++) {
                #pragma omp simd
                for (int k = 1; k < n-1; k++)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(B, i, j, k, n);
            }
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 6: Collapsed                                             */
/*  collapse(3) exposes maximum parallelism; may inhibit k-SIMD       */
/* ------------------------------------------------------------------ */

static void kernel_heat3d_collapsed(int tsteps, int n,
                                    double* restrict A,
                                    double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    B[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);

        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                for (int k = 1; k < n-1; k++)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(B, i, j, k, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Red-Black Gauss-Seidel                                            */
/*                                                                    */
/*  In-place update; no double buffer needed during iteration.        */
/*  Red phase:   points where (i+j+k) % 2 == 0                        */
/*  Black phase: points where (i+j+k) % 2 == 1                        */
/*  Stride-2 k-loop avoids branch inside inner loop.                  */
/*  Runs 2*tsteps sweeps to match Jacobi FLOP count.                  */
/*                                                                    */
/*  In a 7-point stencil, red cells have only black neighbors and     */
/*  vice versa, so within a color phase updates are race-free and     */
/*  the parallel version matches the sequential red-black reference   */
/*  within strict tolerance.                                          */
/* ------------------------------------------------------------------ */

/* Sequential red-black reference (not parallelized; ground truth) */
static void kernel_heat3d_redblack_seq(int tsteps, int n,
                                       double* restrict A) {
    int total_iters = 2 * tsteps;
    for (int t = 0; t < total_iters; t++) {
        /* Red phase: (i+j+k) % 2 == 0 */
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                int k0 = 1 + ((i + j + 1) & 1);
                for (int k = k0; k < n-1; k += 2)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);
            }
        }
        /* Black phase: (i+j+k) % 2 == 1 */
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                int k0 = 1 + ((i + j) & 1);
                for (int k = k0; k < n-1; k += 2)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);
            }
        }
    }
}

/* Strategy 7: parallel red-black (verified against sequential red-black) */
static void kernel_heat3d_red_black(int tsteps, int n,
                                    double* restrict A,
                                    double* restrict B) {
    int total_iters = 2 * tsteps;

    for (int t = 0; t < total_iters; t++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                int k0 = 1 + ((i + j + 1) & 1);
                for (int k = k0; k < n-1; k += 2)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                int k0 = 1 + ((i + j) & 1);
                for (int k = k0; k < n-1; k += 2)
                    A[IDX3(i,j,k,n,n)] = stencil_7pt(A, i, j, k, n);
            }
        }
    }

    /* Keep B in sync with A for interface consistency with driver */
    size_t sz = (size_t)n * n * n * sizeof(double);
    memcpy(B, A, sz);
}

/* ------------------------------------------------------------------ */
/*  Strategy dispatch table                                           */
/* ------------------------------------------------------------------ */

typedef void (*KernelFunc)(int, int, double*, double*);

typedef enum {
    REF_JACOBI   = 0,   /* verify against sequential Jacobi reference  */
    REF_REDBLACK = 1    /* verify against sequential red-black ref.    */
} RefKind;

typedef struct {
    const char* name;
    KernelFunc  func;
    RefKind     ref;
} Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential",      kernel_heat3d_sequential,      REF_JACOBI},
    {"threads_static",  kernel_heat3d_threads_static,  REF_JACOBI},
    {"threads_dynamic", kernel_heat3d_threads_dynamic, REF_JACOBI},
    {"tiled",           kernel_heat3d_tiled,           REF_JACOBI},
    {"simd",            kernel_heat3d_simd,            REF_JACOBI},
    {"collapsed",       kernel_heat3d_collapsed,       REF_JACOBI},
    {"red_black",       kernel_heat3d_red_black,       REF_REDBLACK}
};

static const int NUM_STRATEGIES = sizeof(STRATEGIES) / sizeof(STRATEGIES[0]);

/* Returns 1 if the strategy matching `name` should run given CLI arg */
static int strategy_selected(const char* requested, const char* name) {
    if (!requested || strcmp(requested, "all") == 0) return 1;
    return strstr(requested, name) != NULL;
}

/* ------------------------------------------------------------------ */
/*  CLI                                                               */
/* ------------------------------------------------------------------ */

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --dataset SIZE     MINI, SMALL, MEDIUM, LARGE, EXTRALARGE (default: LARGE)\n");
    printf("  --iterations N     Timed iterations (default: 10)\n");
    printf("  --warmup N         Warmup iterations (default: 3)\n");
    printf("  --threads N        OpenMP thread count (default: all)\n");
    printf("  --output csv       Export results to CSV\n");
    printf("  --strategies LIST  Comma-separated or 'all' (default: all)\n");
    printf("  --help             Show this help\n");
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    DatasetSize dataset_size = DATASET_LARGE;
    int iterations = 10;
    int warmup     = 3;
    int threads    = omp_get_max_threads();
    int output_csv = 0;
    char* strategies_arg = NULL;

    static struct option long_options[] = {
        {"dataset",    required_argument, 0, 'd'},
        {"iterations", required_argument, 0, 'i'},
        {"warmup",     required_argument, 0, 'w'},
        {"threads",    required_argument, 0, 't'},
        {"output",     required_argument, 0, 'o'},
        {"strategies", required_argument, 0, 's'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:i:w:t:o:s:h",
                              long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                for (int i = 0; i <= DATASET_EXTRALARGE; i++) {
                    if (strcasecmp(optarg, DATASET_NAMES[i]) == 0) {
                        dataset_size = (DatasetSize)i;
                        break;
                    }
                }
                break;
            case 'i': iterations = atoi(optarg); break;
            case 'w': warmup     = atoi(optarg); break;
            case 't': threads    = atoi(optarg); break;
            case 'o': output_csv = (strcmp(optarg, "csv") == 0); break;
            case 's': strategies_arg = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
        }
    }

    setup_openmp_env();
    omp_set_num_threads(threads);

    const DatasetHeat3D* ds = &DATASETS[dataset_size];
    int n      = ds->n;
    int tsteps = ds->tsteps;
    double flops = flops_heat3d(tsteps, n);

    size_t array_bytes = (size_t)n * n * n * sizeof(double);

    printf("Heat-3D Benchmark\n");
    printf("Dataset: %s (N=%d, TSTEPS=%d)\n",
           DATASET_NAMES[dataset_size], n, tsteps);
    printf("Threads: %d | Iterations: %d | Warmup: %d\n",
           threads, iterations, warmup);
    printf("FLOPS: %.2e | Memory: %.1f MB (per array)\n",
           flops, array_bytes / (1024.0 * 1024.0));
    printf("NOTE: memory-bound, AI ~0.2-0.8 FLOP/byte (cache-dependent)\n\n");

    /* Allocate arrays (flat 1D, cache-aligned) */
    double* A        = ALLOC_3D(double, n, n, n);
    double* B        = ALLOC_3D(double, n, n, n);
    double* A_jacobi = ALLOC_3D(double, n, n, n);
    double* A_rb     = NULL;

    /* Decide whether the red-black reference is needed */
    int need_rb_ref = 0;
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        if (STRATEGIES[s].ref == REF_REDBLACK &&
            strategy_selected(strategies_arg, STRATEGIES[s].name)) {
            need_rb_ref = 1;
            break;
        }
    }

    /* Jacobi reference */
    init_array(n, A, B);
    kernel_heat3d_sequential(tsteps, n, A, B);
    memcpy(A_jacobi, A, array_bytes);

    /* Red-black reference (only if used) */
    if (need_rb_ref) {
        A_rb = ALLOC_3D(double, n, n, n);
        init_array(n, A, B);
        kernel_heat3d_redblack_seq(tsteps, n, A);
        memcpy(A_rb, A, array_bytes);
    }

    /* Metrics collector */
    MetricsCollector mc;
    metrics_init(&mc, "heat3d", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();

    /* Run strategies */
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        if (!strategy_selected(strategies_arg, STRATEGIES[s].name))
            continue;

        TimingData timing;
        timing_init(&timing);

        /* Warmup */
        for (int w = 0; w < warmup; w++) {
            init_array(n, A, B);
            STRATEGIES[s].func(tsteps, n, A, B);
        }

        /* Timed iterations */
        for (int iter = 0; iter < iterations; iter++) {
            init_array(n, A, B);

            double t0 = omp_get_wtime();
            STRATEGIES[s].func(tsteps, n, A, B);
            double t1 = omp_get_wtime();

            timing_record(&timing, (t1 - t0) * 1000.0);
        }

        /* Verification run */
        init_array(n, A, B);
        STRATEGIES[s].func(tsteps, n, A, B);

        const double* ref = (STRATEGIES[s].ref == REF_REDBLACK) ? A_rb : A_jacobi;
        double max_err = verify_result(n, ref, A);
        int verified = (max_err < VERIFY_TOLERANCE);

        metrics_record(&mc, STRATEGIES[s].name, &timing, flops,
                       verified, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }

    /* CSV export */
    if (output_csv) {
        char filename[256];
        char ts[64];
        get_timestamp(ts, sizeof(ts));
        snprintf(filename, sizeof(filename),
                 "results/heat3d_%s_%s.csv",
                 DATASET_NAMES[dataset_size], ts);
        metrics_export_csv(&mc, filename);
    }

    /* Cleanup */
    FREE_ARRAY(A);
    FREE_ARRAY(B);
    FREE_ARRAY(A_jacobi);
    if (A_rb) FREE_ARRAY(A_rb);

    return 0;
}