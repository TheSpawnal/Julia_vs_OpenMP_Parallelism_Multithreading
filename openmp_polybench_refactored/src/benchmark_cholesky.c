/*
 * Cholesky Decomposition Benchmark
 * A = L * L^T where L is lower triangular
 * 
 * Dependency-limited benchmark with row-level parallelism
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>

static const int DATASETS[] = {
    40,    // MINI
    120,   // SMALL
    400,   // MEDIUM
    2000,  // LARGE
    4000   // EXTRALARGE
};

static void init_array(int n, double* A) {
    // Create positive-definite matrix: A = B * B^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[IDX2(i, j, n)] = (double)(-(j % n)) / n + 1;
        }
        for (int j = i + 1; j < n; j++) {
            A[IDX2(i, j, n)] = 0.0;
        }
        A[IDX2(i, i, n)] = 1.0;
    }
    
    // Make it positive definite: A = A * A^T
    double* B = ALLOC_2D(double, n, n);
    memcpy(B, A, n * n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += B[IDX2(i, k, n)] * B[IDX2(j, k, n)];
            A[IDX2(i, j, n)] = sum;
        }
    }
    
    FREE_ARRAY(B);
}

// Sequential Cholesky-Banachiewicz
static void kernel_cholesky_sequential(int n, double* A) {
    for (int i = 0; i < n; i++) {
        // Off-diagonal elements
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++)
                A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
            A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
        }
        // Diagonal element
        for (int k = 0; k < i; k++)
            A[IDX2(i, i, n)] -= A[IDX2(i, k, n)] * A[IDX2(i, k, n)];
        A[IDX2(i, i, n)] = sqrt(A[IDX2(i, i, n)]);
    }
}

// threads_static - row-level parallelism where possible
static void kernel_cholesky_threads_static(int n, double* A) {
    for (int i = 0; i < n; i++) {
        // Off-diagonal elements (j-loop can be parallelized for large i)
        if (i > 64) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < j; k++)
                    A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
            }
        } else {
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < j; k++)
                    A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
            }
        }
        
        // Diagonal element (sequential due to dependency)
        for (int k = 0; k < i; k++)
            A[IDX2(i, i, n)] -= A[IDX2(i, k, n)] * A[IDX2(i, k, n)];
        A[IDX2(i, i, n)] = sqrt(A[IDX2(i, i, n)]);
    }
}

// Tiled/blocked Cholesky
#define TILE_SIZE 64

static void kernel_cholesky_tiled(int n, double* A) {
    for (int ii = 0; ii < n; ii += TILE_SIZE) {
        int i_end = MIN(ii + TILE_SIZE, n);
        
        // Factor diagonal block
        for (int i = ii; i < i_end; i++) {
            for (int j = ii; j < i; j++) {
                for (int k = ii; k < j; k++)
                    A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
            }
            for (int k = ii; k < i; k++)
                A[IDX2(i, i, n)] -= A[IDX2(i, k, n)] * A[IDX2(i, k, n)];
            A[IDX2(i, i, n)] = sqrt(A[IDX2(i, i, n)]);
        }
        
        // Update trailing blocks (parallel)
        #pragma omp parallel for schedule(dynamic)
        for (int jj = ii + TILE_SIZE; jj < n; jj += TILE_SIZE) {
            int j_end = MIN(jj + TILE_SIZE, n);
            
            // Update block (jj, ii) using diagonal block
            for (int i = jj; i < j_end; i++) {
                for (int j = ii; j < i_end; j++) {
                    for (int k = ii; k < j; k++)
                        A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                    A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
                }
            }
        }
    }
}

// SIMD for inner k-loop
static void kernel_cholesky_simd(int n, double* A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double sum = A[IDX2(i, j, n)];
            #pragma omp simd reduction(-:sum)
            for (int k = 0; k < j; k++)
                sum -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
            A[IDX2(i, j, n)] = sum / A[IDX2(j, j, n)];
        }
        
        double diag = A[IDX2(i, i, n)];
        #pragma omp simd reduction(-:diag)
        for (int k = 0; k < i; k++)
            diag -= A[IDX2(i, k, n)] * A[IDX2(i, k, n)];
        A[IDX2(i, i, n)] = sqrt(diag);
    }
}

// Task-based with row dependencies
static void kernel_cholesky_tasks(int n, double* A) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < n; i++) {
                #pragma omp task depend(inout: A[IDX2(i,0,n):i])
                {
                    for (int j = 0; j < i; j++) {
                        for (int k = 0; k < j; k++)
                            A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                        A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
                    }
                    for (int k = 0; k < i; k++)
                        A[IDX2(i, i, n)] -= A[IDX2(i, k, n)] * A[IDX2(i, k, n)];
                    A[IDX2(i, i, n)] = sqrt(A[IDX2(i, i, n)]);
                }
            }
        }
    }
}

static double verify_result(int n, const double* A_ref, const double* A) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double ref = A_ref[IDX2(i, j, n)];
            double val = A[IDX2(i, j, n)];
            double err = fabs(ref - val);
            if (fabs(ref) > 1e-10) err /= fabs(ref);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

typedef void (*KernelFunc)(int, double*);
typedef struct { const char* name; KernelFunc func; } Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential", kernel_cholesky_sequential},
    {"threads_static", kernel_cholesky_threads_static},
    {"tiled", kernel_cholesky_tiled},
    {"simd", kernel_cholesky_simd},
    {"tasks", kernel_cholesky_tasks}
};

static const int NUM_STRATEGIES = sizeof(STRATEGIES) / sizeof(STRATEGIES[0]);

int main(int argc, char* argv[]) {
    DatasetSize dataset_size = DATASET_LARGE;
    int iterations = 10, warmup = 3;
    int threads = omp_get_max_threads();
    int output_csv = 0;
    
    static struct option long_options[] = {
        {"dataset", required_argument, 0, 'd'},
        {"iterations", required_argument, 0, 'i'},
        {"warmup", required_argument, 0, 'w'},
        {"threads", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "d:i:w:t:o:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                for (int i = 0; i <= DATASET_EXTRALARGE; i++)
                    if (strcasecmp(optarg, DATASET_NAMES[i]) == 0) { dataset_size = i; break; }
                break;
            case 'i': iterations = atoi(optarg); break;
            case 'w': warmup = atoi(optarg); break;
            case 't': threads = atoi(optarg); break;
            case 'o': output_csv = (strcmp(optarg, "csv") == 0); break;
        }
    }
    
    setup_openmp_env();
    omp_set_num_threads(threads);
    
    int n = DATASETS[dataset_size];
    double flops = flops_cholesky(n);
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Dataset: %s (N=%d)\n", DATASET_NAMES[dataset_size], n);
    printf("Threads: %d | FLOPS: %.2e\n", threads, flops);
    printf("NOTE: Dependency-limited benchmark, row-level parallelism only\n\n");
    
    double* A = ALLOC_2D(double, n, n);
    double* A_copy = ALLOC_2D(double, n, n);
    double* A_ref = ALLOC_2D(double, n, n);
    
    init_array(n, A);
    memcpy(A_ref, A, n * n * sizeof(double));
    kernel_cholesky_sequential(n, A_ref);
    
    MetricsCollector mc;
    metrics_init(&mc, "cholesky", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();
    
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        TimingData timing;
        timing_init(&timing);
        
        for (int w = 0; w < warmup; w++) {
            init_array(n, A);
            STRATEGIES[s].func(n, A);
        }
        
        for (int iter = 0; iter < iterations; iter++) {
            init_array(n, A);
            
            double start = omp_get_wtime();
            STRATEGIES[s].func(n, A);
            double end = omp_get_wtime();
            
            timing_record(&timing, (end - start) * 1000.0);
        }
        
        init_array(n, A);
        STRATEGIES[s].func(n, A);
        double max_err = verify_result(n, A_ref, A);
        
        metrics_record(&mc, STRATEGIES[s].name, &timing, flops, max_err < VERIFY_TOLERANCE, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }
    
    if (output_csv) {
        char filename[256], timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));
        snprintf(filename, sizeof(filename), "results/cholesky_%s_%s.csv",
                 DATASET_NAMES[dataset_size], timestamp);
        metrics_export_csv(&mc, filename);
    }
    
    FREE_ARRAY(A);
    FREE_ARRAY(A_copy);
    FREE_ARRAY(A_ref);
    
    return 0;
}
