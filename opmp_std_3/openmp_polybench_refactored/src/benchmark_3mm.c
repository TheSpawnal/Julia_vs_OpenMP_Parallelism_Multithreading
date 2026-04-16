/*
 * 3MM Benchmark - Triple Matrix Multiplication
 * G = (A*B) * (C*D)
 * E = A*B, F = C*D, G = E*F
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>

typedef struct {
    int ni, nj, nk, nl, nm;
} Dataset3MM;

static const Dataset3MM DATASETS[] = {
    {16, 18, 20, 22, 24},         // MINI
    {40, 50, 60, 70, 80},         // SMALL
    {180, 190, 200, 210, 220},    // MEDIUM
    {800, 900, 1000, 1100, 1200}, // LARGE
    {1600, 1800, 2000, 2200, 2400} // EXTRALARGE
};

static void init_arrays(int ni, int nj, int nk, int nl, int nm,
                       double* A, double* B, double* C, double* D) {
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nk; j++)
            A[IDX2(i, j, nk)] = (double)((i * j + 1) % ni) / (5 * ni);
    
    for (int i = 0; i < nk; i++)
        for (int j = 0; j < nj; j++)
            B[IDX2(i, j, nj)] = (double)((i * (j + 1) + 2) % nj) / (5 * nj);
    
    for (int i = 0; i < nj; i++)
        for (int j = 0; j < nm; j++)
            C[IDX2(i, j, nm)] = (double)(i * (j + 3) % nl) / (5 * nl);
    
    for (int i = 0; i < nm; i++)
        for (int j = 0; j < nl; j++)
            D[IDX2(i, j, nl)] = (double)((i * (j + 2) + 2) % nk) / (5 * nk);
}

// Sequential baseline
static void kernel_3mm_sequential(int ni, int nj, int nk, int nl, int nm,
                                  double* A, double* B, double* C, double* D,
                                  double* E, double* F, double* G) {
    // E = A * B
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            E[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                E[IDX2(i, j, nj)] += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    // F = C * D
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nl; j++) {
            F[IDX2(i, j, nl)] = 0.0;
            for (int k = 0; k < nm; k++) {
                F[IDX2(i, j, nl)] += C[IDX2(i, k, nm)] * D[IDX2(k, j, nl)];
            }
        }
    }
    // G = E * F
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            G[IDX2(i, j, nl)] = 0.0;
            for (int k = 0; k < nj; k++) {
                G[IDX2(i, j, nl)] += E[IDX2(i, k, nj)] * F[IDX2(k, j, nl)];
            }
        }
    }
}

// threads_static
static void kernel_3mm_threads_static(int ni, int nj, int nk, int nl, int nm,
                                      double* A, double* B, double* C, double* D,
                                      double* E, double* F, double* G) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            E[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                E[IDX2(i, j, nj)] += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nl; j++) {
            F[IDX2(i, j, nl)] = 0.0;
            for (int k = 0; k < nm; k++) {
                F[IDX2(i, j, nl)] += C[IDX2(i, k, nm)] * D[IDX2(k, j, nl)];
            }
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            G[IDX2(i, j, nl)] = 0.0;
            for (int k = 0; k < nj; k++) {
                G[IDX2(i, j, nl)] += E[IDX2(i, k, nj)] * F[IDX2(k, j, nl)];
            }
        }
    }
}

// threads_dynamic
static void kernel_3mm_threads_dynamic(int ni, int nj, int nk, int nl, int nm,
                                       double* A, double* B, double* C, double* D,
                                       double* E, double* F, double* G) {
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            E[IDX2(i, j, nj)] = 0.0;
            for (int k = 0; k < nk; k++) {
                E[IDX2(i, j, nj)] += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nl; j++) {
            F[IDX2(i, j, nl)] = 0.0;
            for (int k = 0; k < nm; k++) {
                F[IDX2(i, j, nl)] += C[IDX2(i, k, nm)] * D[IDX2(k, j, nl)];
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            G[IDX2(i, j, nl)] = 0.0;
            for (int k = 0; k < nj; k++) {
                G[IDX2(i, j, nl)] += E[IDX2(i, k, nj)] * F[IDX2(k, j, nl)];
            }
        }
    }
}

// tiled
#define TILE_SIZE 32

static void kernel_3mm_tiled(int ni, int nj, int nk, int nl, int nm,
                            double* A, double* B, double* C, double* D,
                            double* E, double* F, double* G) {
    // E = A * B (tiled)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += TILE_SIZE) {
        for (int jj = 0; jj < nj; jj += TILE_SIZE) {
            int i_end = MIN(ii + TILE_SIZE, ni);
            int j_end = MIN(jj + TILE_SIZE, nj);
            for (int i = ii; i < i_end; i++)
                for (int j = jj; j < j_end; j++)
                    E[IDX2(i, j, nj)] = 0.0;
            
            for (int kk = 0; kk < nk; kk += TILE_SIZE) {
                int k_end = MIN(kk + TILE_SIZE, nk);
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_end; k++)
                            sum += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
                        E[IDX2(i, j, nj)] += sum;
                    }
                }
            }
        }
    }
    
    // F = C * D (tiled)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < nj; ii += TILE_SIZE) {
        for (int jj = 0; jj < nl; jj += TILE_SIZE) {
            int i_end = MIN(ii + TILE_SIZE, nj);
            int j_end = MIN(jj + TILE_SIZE, nl);
            for (int i = ii; i < i_end; i++)
                for (int j = jj; j < j_end; j++)
                    F[IDX2(i, j, nl)] = 0.0;
            
            for (int kk = 0; kk < nm; kk += TILE_SIZE) {
                int k_end = MIN(kk + TILE_SIZE, nm);
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_end; k++)
                            sum += C[IDX2(i, k, nm)] * D[IDX2(k, j, nl)];
                        F[IDX2(i, j, nl)] += sum;
                    }
                }
            }
        }
    }
    
    // G = E * F (tiled)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += TILE_SIZE) {
        for (int jj = 0; jj < nl; jj += TILE_SIZE) {
            int i_end = MIN(ii + TILE_SIZE, ni);
            int j_end = MIN(jj + TILE_SIZE, nl);
            for (int i = ii; i < i_end; i++)
                for (int j = jj; j < j_end; j++)
                    G[IDX2(i, j, nl)] = 0.0;
            
            for (int kk = 0; kk < nj; kk += TILE_SIZE) {
                int k_end = MIN(kk + TILE_SIZE, nj);
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_end; k++)
                            sum += E[IDX2(i, k, nj)] * F[IDX2(k, j, nl)];
                        G[IDX2(i, j, nl)] += sum;
                    }
                }
            }
        }
    }
}

// tasks
static void kernel_3mm_tasks(int ni, int nj, int nk, int nl, int nm,
                            double* A, double* B, double* C, double* D,
                            double* E, double* F, double* G) {
    int chunk = MAX(ni / (omp_get_max_threads() * 4), 1);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // E = A * B and F = C * D can be computed in parallel
            #pragma omp task
            {
                for (int ii = 0; ii < ni; ii += chunk) {
                    #pragma omp task firstprivate(ii)
                    {
                        int i_end = MIN(ii + chunk, ni);
                        for (int i = ii; i < i_end; i++) {
                            for (int j = 0; j < nj; j++) {
                                E[IDX2(i, j, nj)] = 0.0;
                                for (int k = 0; k < nk; k++)
                                    E[IDX2(i, j, nj)] += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
                            }
                        }
                    }
                }
            }
            
            #pragma omp task
            {
                for (int ii = 0; ii < nj; ii += chunk) {
                    #pragma omp task firstprivate(ii)
                    {
                        int i_end = MIN(ii + chunk, nj);
                        for (int i = ii; i < i_end; i++) {
                            for (int j = 0; j < nl; j++) {
                                F[IDX2(i, j, nl)] = 0.0;
                                for (int k = 0; k < nm; k++)
                                    F[IDX2(i, j, nl)] += C[IDX2(i, k, nm)] * D[IDX2(k, j, nl)];
                            }
                        }
                    }
                }
            }
            
            #pragma omp taskwait
            
            // G = E * F
            for (int ii = 0; ii < ni; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    int i_end = MIN(ii + chunk, ni);
                    for (int i = ii; i < i_end; i++) {
                        for (int j = 0; j < nl; j++) {
                            G[IDX2(i, j, nl)] = 0.0;
                            for (int k = 0; k < nj; k++)
                                G[IDX2(i, j, nl)] += E[IDX2(i, k, nj)] * F[IDX2(k, j, nl)];
                        }
                    }
                }
            }
        }
    }
}

static double verify_result(int ni, int nl, const double* G_ref, const double* G) {
    double max_err = 0.0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double ref = G_ref[IDX2(i, j, nl)];
            double val = G[IDX2(i, j, nl)];
            double err = fabs(ref - val);
            if (ref != 0.0) err /= fabs(ref);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

typedef void (*KernelFunc)(int, int, int, int, int, double*, double*, double*, double*,
                           double*, double*, double*);

typedef struct { const char* name; KernelFunc func; } Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential", kernel_3mm_sequential},
    {"threads_static", kernel_3mm_threads_static},
    {"threads_dynamic", kernel_3mm_threads_dynamic},
    {"tiled", kernel_3mm_tiled},
    {"tasks", kernel_3mm_tasks}
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
    
    const Dataset3MM* ds = &DATASETS[dataset_size];
    int ni = ds->ni, nj = ds->nj, nk = ds->nk, nl = ds->nl, nm = ds->nm;
    double flops = flops_3mm(ni, nj, nk, nl, nm);
    
    printf("3MM Benchmark\n");
    printf("Dataset: %s (NI=%d, NJ=%d, NK=%d, NL=%d, NM=%d)\n",
           DATASET_NAMES[dataset_size], ni, nj, nk, nl, nm);
    printf("Threads: %d | FLOPS: %.2e\n\n", threads, flops);
    
    double* A = ALLOC_2D(double, ni, nk);
    double* B = ALLOC_2D(double, nk, nj);
    double* C = ALLOC_2D(double, nj, nm);
    double* D = ALLOC_2D(double, nm, nl);
    double* E = ALLOC_2D(double, ni, nj);
    double* F = ALLOC_2D(double, nj, nl);
    double* G = ALLOC_2D(double, ni, nl);
    double* G_ref = ALLOC_2D(double, ni, nl);
    
    init_arrays(ni, nj, nk, nl, nm, A, B, C, D);
    kernel_3mm_sequential(ni, nj, nk, nl, nm, A, B, C, D, E, F, G_ref);
    
    MetricsCollector mc;
    metrics_init(&mc, "3mm", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();
    
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        TimingData timing;
        timing_init(&timing);
        
        for (int w = 0; w < warmup; w++) {
            memset(E, 0, ni * nj * sizeof(double));
            memset(F, 0, nj * nl * sizeof(double));
            memset(G, 0, ni * nl * sizeof(double));
            STRATEGIES[s].func(ni, nj, nk, nl, nm, A, B, C, D, E, F, G);
        }
        
        for (int iter = 0; iter < iterations; iter++) {
            memset(E, 0, ni * nj * sizeof(double));
            memset(F, 0, nj * nl * sizeof(double));
            memset(G, 0, ni * nl * sizeof(double));
            
            double start = omp_get_wtime();
            STRATEGIES[s].func(ni, nj, nk, nl, nm, A, B, C, D, E, F, G);
            double end = omp_get_wtime();
            
            timing_record(&timing, (end - start) * 1000.0);
        }
        
        memset(E, 0, ni * nj * sizeof(double));
        memset(F, 0, nj * nl * sizeof(double));
        memset(G, 0, ni * nl * sizeof(double));
        STRATEGIES[s].func(ni, nj, nk, nl, nm, A, B, C, D, E, F, G);
        double max_err = verify_result(ni, nl, G_ref, G);
        
        metrics_record(&mc, STRATEGIES[s].name, &timing, flops, max_err < VERIFY_TOLERANCE, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }
    
    if (output_csv) {
        char filename[256], timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));
        snprintf(filename, sizeof(filename), "results/3mm_%s_%s.csv",
                 DATASET_NAMES[dataset_size], timestamp);
        metrics_export_csv(&mc, filename);
    }
    
    FREE_ARRAY(A); FREE_ARRAY(B); FREE_ARRAY(C); FREE_ARRAY(D);
    FREE_ARRAY(E); FREE_ARRAY(F); FREE_ARRAY(G); FREE_ARRAY(G_ref);
    
    return 0;
}
