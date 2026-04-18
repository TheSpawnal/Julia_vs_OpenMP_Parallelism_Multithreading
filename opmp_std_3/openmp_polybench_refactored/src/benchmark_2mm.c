/*
 * 2MM Benchmark - Chained Matrix Multiplication
 * D = alpha * A * B * C + beta * D
 * 
 * Strategies aligned with Julia implementation:
 * - sequential: Baseline (no parallelization)
 * - threads_static: OpenMP parallel for with static scheduling
 * - threads_dynamic: OpenMP parallel for with dynamic scheduling  
 * - tiled: Cache-blocked with parallel tiles
 * - simd: Explicit SIMD vectorization
 * - tasks: Task-based parallelism
 * - collapsed: Collapsed loop parallelization
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>

// Dataset configurations
typedef struct {
    int ni, nj, nk, nl;
} Dataset2MM;

static const Dataset2MM DATASETS[] = {
    {16, 18, 22, 24},       // MINI
    {40, 50, 70, 80},       // SMALL
    {180, 190, 210, 220},   // MEDIUM
    {800, 900, 1100, 1200}, // LARGE
    {1600, 1800, 2200, 2400} // EXTRALARGE
};

static double ALPHA = 1.5;
static double BETA = 1.2;

// Initialize arrays
static void init_arrays(int ni, int nj, int nk, int nl,
                       double* A, double* B, double* C, double* D) {
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nk; j++)
            A[IDX2(i, j, nk)] = (double)((i * j + 1) % ni) / ni;
    
    for (int i = 0; i < nk; i++)
        for (int j = 0; j < nj; j++)
            B[IDX2(i, j, nj)] = (double)(i * (j + 1) % nj) / nj;
    
    for (int i = 0; i < nj; i++)
        for (int j = 0; j < nl; j++)
            C[IDX2(i, j, nl)] = (double)((i * (j + 3) + 1) % nl) / nl;
    
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nl; j++)
            D[IDX2(i, j, nl)] = (double)(i * (j + 2) % nk) / nk;
}

// Strategy 1: Sequential baseline
static void kernel_2mm_sequential(int ni, int nj, int nk, int nl,
                                  double alpha, double beta,
                                  double* A, double* B, double* tmp,
                                  double* C, double* D) {
    // tmp = alpha * A * B
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                tmp[IDX2(i, j, nj)] += alpha * A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    
    // D = tmp * C + beta * D
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[IDX2(i, j, nl)] *= beta;
            for (int k = 0; k < nj; k++) {
                D[IDX2(i, j, nl)] += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
            }
        }
    }
}

// Strategy 2: threads_static
static void kernel_2mm_threads_static(int ni, int nj, int nk, int nl,
                                      double alpha, double beta,
                                      double* A, double* B, double* tmp,
                                      double* C, double* D) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                tmp[IDX2(i, j, nj)] += alpha * A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[IDX2(i, j, nl)] *= beta;
            for (int k = 0; k < nj; k++) {
                D[IDX2(i, j, nl)] += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
            }
        }
    }
}

// Strategy 3: threads_dynamic
static void kernel_2mm_threads_dynamic(int ni, int nj, int nk, int nl,
                                       double alpha, double beta,
                                       double* A, double* B, double* tmp,
                                       double* C, double* D) {
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                tmp[IDX2(i, j, nj)] += alpha * A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[IDX2(i, j, nl)] *= beta;
            for (int k = 0; k < nj; k++) {
                D[IDX2(i, j, nl)] += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
            }
        }
    }
}

// Strategy 4: Tiled (cache-blocked)
#define TILE_SIZE 32

static void kernel_2mm_tiled(int ni, int nj, int nk, int nl,
                            double alpha, double beta,
                            double* A, double* B, double* tmp,
                            double* C, double* D) {
    // tmp = alpha * A * B (tiled)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < ni; ii += TILE_SIZE) {
        for (int jj = 0; jj < nj; jj += TILE_SIZE) {
            int i_end = MIN(ii + TILE_SIZE, ni);
            int j_end = MIN(jj + TILE_SIZE, nj);
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    tmp[IDX2(i, j, nj)] = 0.0;
                }
            }
            
            for (int kk = 0; kk < nk; kk += TILE_SIZE) {
                int k_end = MIN(kk + TILE_SIZE, nk);
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_end; k++) {
                            sum += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
                        }
                        tmp[IDX2(i, j, nj)] += alpha * sum;
                    }
                }
            }
        }
    }
    
    // D = tmp * C + beta * D (tiled)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < ni; ii += TILE_SIZE) {
        for (int jj = 0; jj < nl; jj += TILE_SIZE) {
            int i_end = MIN(ii + TILE_SIZE, ni);
            int j_end = MIN(jj + TILE_SIZE, nl);
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    D[IDX2(i, j, nl)] *= beta;
                }
            }
            
            for (int kk = 0; kk < nj; kk += TILE_SIZE) {
                int k_end = MIN(kk + TILE_SIZE, nj);
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_end; k++) {
                            sum += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
                        }
                        D[IDX2(i, j, nl)] += sum;
                    }
                }
            }
        }
    }
}

// Strategy 5: SIMD vectorization
static void kernel_2mm_simd(int ni, int nj, int nk, int nl,
                           double alpha, double beta,
                           double* A, double* B, double* tmp,
                           double* C, double* D) {
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nk; k++) {
                sum += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
            tmp[IDX2(i, j, nj)] = alpha * sum;
        }
    }
    
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[IDX2(i, j, nl)];
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
            }
            D[IDX2(i, j, nl)] = sum;
        }
    }
}

// Strategy 6: Task-based
static void kernel_2mm_tasks(int ni, int nj, int nk, int nl,
                            double alpha, double beta,
                            double* A, double* B, double* tmp,
                            double* C, double* D) {
    int chunk = MAX(ni / (omp_get_max_threads() * 4), 1);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // tmp = alpha * A * B
            for (int ii = 0; ii < ni; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    int i_end = MIN(ii + chunk, ni);
                    for (int i = ii; i < i_end; i++) {
                        for (int j = 0; j < nj; j++) {
                            tmp[IDX2(i, j, nj)] = 0.0;
                            for (int k = 0; k < nk; k++) {
                                tmp[IDX2(i, j, nj)] += alpha * A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
            
            // D = tmp * C + beta * D
            for (int ii = 0; ii < ni; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    int i_end = MIN(ii + chunk, ni);
                    for (int i = ii; i < i_end; i++) {
                        for (int j = 0; j < nl; j++) {
                            D[IDX2(i, j, nl)] *= beta;
                            for (int k = 0; k < nj; k++) {
                                D[IDX2(i, j, nl)] += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Strategy 7: Collapsed loops
static void kernel_2mm_collapsed(int ni, int nj, int nk, int nl,
                                 double alpha, double beta,
                                 double* A, double* B, double* tmp,
                                 double* C, double* D) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                tmp[IDX2(i, j, nj)] += alpha * A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[IDX2(i, j, nl)] *= beta;
            for (int k = 0; k < nj; k++) {
                D[IDX2(i, j, nl)] += tmp[IDX2(i, k, nj)] * C[IDX2(k, j, nl)];
            }
        }
    }
}

// Verification
static double verify_result(int ni, int nl, const double* D_ref, const double* D) {
    double max_err = 0.0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double ref = D_ref[IDX2(i, j, nl)];
            double val = D[IDX2(i, j, nl)];
            double err = fabs(ref - val);
            if (ref != 0.0) err /= fabs(ref);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// Strategy dispatcher
typedef void (*KernelFunc)(int, int, int, int, double, double, 
                           double*, double*, double*, double*, double*);

typedef struct {
    const char* name;
    KernelFunc func;
} Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential", kernel_2mm_sequential},
    {"threads_static", kernel_2mm_threads_static},
    {"threads_dynamic", kernel_2mm_threads_dynamic},
    {"tiled", kernel_2mm_tiled},
    {"simd", kernel_2mm_simd},
    {"tasks", kernel_2mm_tasks},
    {"collapsed", kernel_2mm_collapsed}
};

static const int NUM_STRATEGIES = sizeof(STRATEGIES) / sizeof(STRATEGIES[0]);

// Print usage
static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --dataset SIZE     Dataset size: MINI, SMALL, MEDIUM, LARGE, EXTRALARGE (default: LARGE)\n");
    printf("  --iterations N     Number of timed iterations (default: 10)\n");
    printf("  --warmup N         Number of warmup iterations (default: 3)\n");
    printf("  --threads N        Number of threads (default: all available)\n");
    printf("  --output csv       Export results to CSV\n");
    printf("  --strategies LIST  Comma-separated strategies or 'all' (default: all)\n");
    printf("  --help             Show this help\n");
}

int main(int argc, char* argv[]) {
    // Default parameters
    DatasetSize dataset_size = DATASET_LARGE;
    int iterations = 10;
    int warmup = 3;
    int threads = omp_get_max_threads();
    int output_csv = 0;
    char* strategies_arg = NULL;
    
    // Parse arguments
    static struct option long_options[] = {
        {"dataset", required_argument, 0, 'd'},
        {"iterations", required_argument, 0, 'i'},
        {"warmup", required_argument, 0, 'w'},
        {"threads", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},
        {"strategies", required_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "d:i:w:t:o:s:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                for (int i = 0; i <= DATASET_EXTRALARGE; i++) {
                    if (strcasecmp(optarg, DATASET_NAMES[i]) == 0) {
                        dataset_size = i;
                        break;
                    }
                }
                break;
            case 'i': iterations = atoi(optarg); break;
            case 'w': warmup = atoi(optarg); break;
            case 't': threads = atoi(optarg); break;
            case 'o': output_csv = (strcmp(optarg, "csv") == 0); break;
            case 's': strategies_arg = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
        }
    }
    
    // Setup
    setup_openmp_env();
    omp_set_num_threads(threads);
    
    const Dataset2MM* ds = &DATASETS[dataset_size];
    int ni = ds->ni, nj = ds->nj, nk = ds->nk, nl = ds->nl;
    double flops = flops_2mm(ni, nj, nk, nl);
    
    printf("2MM Benchmark\n");
    printf("Dataset: %s (NI=%d, NJ=%d, NK=%d, NL=%d)\n", 
           DATASET_NAMES[dataset_size], ni, nj, nk, nl);
    printf("Threads: %d | Iterations: %d | Warmup: %d\n", threads, iterations, warmup);
    printf("FLOPS: %.2e\n\n", flops);
    
    // Allocate arrays
    double* A = ALLOC_2D(double, ni, nk);
    double* B = ALLOC_2D(double, nk, nj);
    double* C = ALLOC_2D(double, nj, nl);
    double* D = ALLOC_2D(double, ni, nl);
    double* D_ref = ALLOC_2D(double, ni, nl);
    double* tmp = ALLOC_2D(double, ni, nj);
    
    // Initialize
    init_arrays(ni, nj, nk, nl, A, B, C, D);
    memcpy(D_ref, D, ni * nl * sizeof(double));
    
    // Compute reference (sequential)
    kernel_2mm_sequential(ni, nj, nk, nl, ALPHA, BETA, A, B, tmp, C, D_ref);
    
    // Initialize metrics collector
    MetricsCollector mc;
    metrics_init(&mc, "2mm", DATASET_NAMES[dataset_size], threads);
    
    // Run all strategies
    metrics_print_header();
    
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        // Check if strategy is requested
        if (strategies_arg && strcmp(strategies_arg, "all") != 0) {
            if (strstr(strategies_arg, STRATEGIES[s].name) == NULL) continue;
        }
        
        TimingData timing;
        timing_init(&timing);
        
        // Warmup
        for (int w = 0; w < warmup; w++) {
            init_arrays(ni, nj, nk, nl, A, B, C, D);
            memset(tmp, 0, ni * nj * sizeof(double));
            STRATEGIES[s].func(ni, nj, nk, nl, ALPHA, BETA, A, B, tmp, C, D);
        }
        
        // Timed iterations
        for (int iter = 0; iter < iterations; iter++) {
            init_arrays(ni, nj, nk, nl, A, B, C, D);
            memset(tmp, 0, ni * nj * sizeof(double));
            
            double start = omp_get_wtime();
            STRATEGIES[s].func(ni, nj, nk, nl, ALPHA, BETA, A, B, tmp, C, D);
            double end = omp_get_wtime();
            
            timing_record(&timing, (end - start) * 1000.0);
        }
        
        // Verify
        init_arrays(ni, nj, nk, nl, A, B, C, D);
        memset(tmp, 0, ni * nj * sizeof(double));
        STRATEGIES[s].func(ni, nj, nk, nl, ALPHA, BETA, A, B, tmp, C, D);
        double max_err = verify_result(ni, nl, D_ref, D);
        int verified = (max_err < VERIFY_TOLERANCE);
        
        // Record and print
        metrics_record(&mc, STRATEGIES[s].name, &timing, flops, verified, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }
    
    // Export results
    if (output_csv) {
        char filename[256];
        char timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));
        snprintf(filename, sizeof(filename), "results/2mm_%s_%s.csv", 
                 DATASET_NAMES[dataset_size], timestamp);
        metrics_export_csv(&mc, filename);
    }
    
    // Cleanup
    FREE_ARRAY(A);
    FREE_ARRAY(B);
    FREE_ARRAY(C);
    FREE_ARRAY(D);
    FREE_ARRAY(D_ref);
    FREE_ARRAY(tmp);
    
    return 0;
}
