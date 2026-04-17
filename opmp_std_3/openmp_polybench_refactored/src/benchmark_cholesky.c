/*
 * Cholesky Decomposition Benchmark
 * A = L * L^T where L is lower triangular
 *
 * Right-looking (Crout) formulation for correct parallel trailing updates.
 *
 * Strategies:
 * - sequential:     Banachiewicz (left-looking) baseline
 * - threads_static: Right-looking with parallel trailing SYRK
 * - tiled:          Blocked POTRF/TRSM/SYRK (LAPACK-style)
 * - simd:           SIMD on inner k-reduction
 * - tasks:          Blocked with task-based TRSM + SYRK and barriers
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>

static const int DATASETS[] = {
    40, 120, 400, 2000, 4000
};

static void init_array(int n, double* A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++)
            A[IDX2(i, j, n)] = (double)(-(j % n)) / n + 1;
        for (int j = i + 1; j < n; j++)
            A[IDX2(i, j, n)] = 0.0;
        A[IDX2(i, i, n)] = 1.0;
    }
    double* B = ALLOC_2D(double, n, n);
    memcpy(B, A, (size_t)n * n * sizeof(double));
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

/* Sequential Banachiewicz (left-looking) */
static void kernel_cholesky_sequential(int n, double* A) {
    for (int i = 0; i < n; i++) {
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

/* Right-looking with parallel trailing rank-1 update.
 * Column j: compute diagonal, scale column, then update trailing
 * submatrix in parallel. Each thread owns a distinct row -> no race. */
static void kernel_cholesky_threads_static(int n, double* A) {
    for (int j = 0; j < n; j++) {
        A[IDX2(j, j, n)] = sqrt(A[IDX2(j, j, n)]);
        double Ljj = A[IDX2(j, j, n)];

        for (int i = j + 1; i < n; i++)
            A[IDX2(i, j, n)] /= Ljj;

        #pragma omp parallel for schedule(static) if(n - j > 128)
        for (int i = j + 1; i < n; i++) {
            double Lij = A[IDX2(i, j, n)];
            for (int k = j + 1; k <= i; k++)
                A[IDX2(i, k, n)] -= Lij * A[IDX2(k, j, n)];
        }
    }
}

/* Blocked POTRF/TRSM/SYRK */
#define CHOL_TILE 64

static void kernel_cholesky_tiled(int n, double* A) {
    for (int jj = 0; jj < n; jj += CHOL_TILE) {
        int jb = MIN(CHOL_TILE, n - jj);

        /* POTRF: factor diagonal block */
        for (int j = jj; j < jj + jb; j++) {
            A[IDX2(j, j, n)] = sqrt(A[IDX2(j, j, n)]);
            double Ljj = A[IDX2(j, j, n)];
            for (int i = j + 1; i < jj + jb; i++)
                A[IDX2(i, j, n)] /= Ljj;
            for (int i = j + 1; i < jj + jb; i++) {
                double Lij = A[IDX2(i, j, n)];
                for (int k = j + 1; k <= i; k++)
                    A[IDX2(i, k, n)] -= Lij * A[IDX2(k, j, n)];
            }
        }

        if (jj + jb >= n) break;

        /* TRSM: solve column panel below (parallel across rows) */
        #pragma omp parallel for schedule(dynamic) if(n - jj - jb > 64)
        for (int i = jj + jb; i < n; i++) {
            for (int j = jj; j < jj + jb; j++) {
                for (int k = jj; k < j; k++)
                    A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
            }
        }

        /* SYRK: update trailing submatrix (parallel across rows) */
        #pragma omp parallel for schedule(dynamic) if(n - jj - jb > 64)
        for (int i = jj + jb; i < n; i++) {
            for (int k = jj + jb; k <= i; k++) {
                double sum = 0.0;
                for (int j = jj; j < jj + jb; j++)
                    sum += A[IDX2(i, j, n)] * A[IDX2(k, j, n)];
                A[IDX2(i, k, n)] -= sum;
            }
        }
    }
}

/* SIMD on inner k-reduction */
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

/* Task-based blocked with barriers between TRSM and SYRK */
static void kernel_cholesky_tasks(int n, double* A) {
    for (int jj = 0; jj < n; jj += CHOL_TILE) {
        int jb = MIN(CHOL_TILE, n - jj);

        /* POTRF: sequential on diagonal block */
        for (int j = jj; j < jj + jb; j++) {
            A[IDX2(j, j, n)] = sqrt(A[IDX2(j, j, n)]);
            double Ljj = A[IDX2(j, j, n)];
            for (int i = j + 1; i < jj + jb; i++)
                A[IDX2(i, j, n)] /= Ljj;
            for (int i = j + 1; i < jj + jb; i++) {
                double Lij = A[IDX2(i, j, n)];
                for (int k = j + 1; k <= i; k++)
                    A[IDX2(i, k, n)] -= Lij * A[IDX2(k, j, n)];
            }
        }

        if (jj + jb >= n) break;

        #pragma omp parallel
        #pragma omp single
        {
            /* TRSM tasks: one per row block */
            for (int ii = jj + jb; ii < n; ii += CHOL_TILE) {
                int ib = MIN(CHOL_TILE, n - ii);
                #pragma omp task firstprivate(ii, ib, jj, jb)
                {
                    for (int i = ii; i < ii + ib; i++) {
                        for (int j = jj; j < jj + jb; j++) {
                            for (int k = jj; k < j; k++)
                                A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
                            A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
                        }
                    }
                }
            }
            #pragma omp taskwait

            /* SYRK tasks: one per row block */
            for (int ii = jj + jb; ii < n; ii += CHOL_TILE) {
                int ib = MIN(CHOL_TILE, n - ii);
                #pragma omp task firstprivate(ii, ib, jj, jb)
                {
                    for (int i = ii; i < ii + ib; i++) {
                        for (int k = jj + jb; k <= i; k++) {
                            double sum = 0.0;
                            for (int j = jj; j < jj + jb; j++)
                                sum += A[IDX2(i, j, n)] * A[IDX2(k, j, n)];
                            A[IDX2(i, k, n)] -= sum;
                        }
                    }
                }
            }
            #pragma omp taskwait
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
    {"sequential",     kernel_cholesky_sequential},
    {"threads_static", kernel_cholesky_threads_static},
    {"tiled",          kernel_cholesky_tiled},
    {"simd",           kernel_cholesky_simd},
    {"tasks",          kernel_cholesky_tasks}
};
static const int NUM_STRATEGIES = sizeof(STRATEGIES) / sizeof(STRATEGIES[0]);

int main(int argc, char* argv[]) {
    DatasetSize dataset_size = DATASET_LARGE;
    int iterations = 10, warmup = 3;
    int threads = omp_get_max_threads();
    int output_csv = 0;

    static struct option long_options[] = {
        {"dataset", required_argument, 0, 'd'}, {"iterations", required_argument, 0, 'i'},
        {"warmup", required_argument, 0, 'w'},  {"threads", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},  {"help", no_argument, 0, 'h'}, {0,0,0,0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "d:i:w:t:o:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                for (int i = 0; i <= DATASET_EXTRALARGE; i++)
                    if (strcasecmp(optarg, DATASET_NAMES[i]) == 0)
                        { dataset_size = (DatasetSize)i; break; }
                break;
            case 'i': iterations = atoi(optarg); break;
            case 'w': warmup     = atoi(optarg); break;
            case 't': threads    = atoi(optarg); break;
            case 'o': output_csv = (strcmp(optarg, "csv") == 0); break;
            case 'h': return 0;
        }
    }

    setup_openmp_env();
    omp_set_num_threads(threads);
    int n = DATASETS[dataset_size];
    double flops = flops_cholesky(n);

    printf("Cholesky Decomposition Benchmark\n");
    printf("Dataset: %s (N=%d)\n", DATASET_NAMES[dataset_size], n);
    printf("Threads: %d | FLOPS: %.2e\n\n", threads, flops);

    double* A     = ALLOC_2D(double, n, n);
    double* A_ref = ALLOC_2D(double, n, n);

    init_array(n, A);
    memcpy(A_ref, A, (size_t)n * n * sizeof(double));
    kernel_cholesky_sequential(n, A_ref);

    MetricsCollector mc;
    metrics_init(&mc, "cholesky", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();

    for (int s = 0; s < NUM_STRATEGIES; s++) {
        TimingData timing;
        timing_init(&timing);
        for (int w = 0; w < warmup; w++) { init_array(n, A); STRATEGIES[s].func(n, A); }
        for (int iter = 0; iter < iterations; iter++) {
            init_array(n, A);
            double t0 = omp_get_wtime();
            STRATEGIES[s].func(n, A);
            double t1 = omp_get_wtime();
            timing_record(&timing, (t1 - t0) * 1000.0);
        }
        init_array(n, A);
        STRATEGIES[s].func(n, A);
        double max_err = verify_result(n, A_ref, A);
        metrics_record(&mc, STRATEGIES[s].name, &timing, flops, max_err < VERIFY_TOLERANCE, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }

    if (output_csv) {
        char filename[256], ts[64];
        get_timestamp(ts, sizeof(ts));
        snprintf(filename, sizeof(filename), "results/cholesky_%s_%s.csv", DATASET_NAMES[dataset_size], ts);
        metrics_export_csv(&mc, filename);
    }

    FREE_ARRAY(A);
    FREE_ARRAY(A_ref);
    return 0;
}