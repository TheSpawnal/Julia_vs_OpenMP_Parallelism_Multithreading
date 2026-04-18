/*
 * Correlation Benchmark - Pearson Correlation Matrix
 * Computes correlation matrix of M data points with N features
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>

#define EPS 1.0e-10

typedef struct { int m, n; } DatasetCorr;

static const DatasetCorr DATASETS[] = {
    {28, 32},      // MINI
    {80, 100},     // SMALL
    {240, 260},    // MEDIUM
    {1200, 1400},  // LARGE
    {2600, 3000}   // EXTRALARGE
};

static void init_array(int m, int n, double* data) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            data[IDX2(i, j, m)] = (double)(i * j) / m + i;
}

// Sequential baseline
static void kernel_correlation_sequential(int m, int n, double* data, double* corr,
                                          double* mean, double* stddev) {
    double sqrt_m = sqrt((double)m);
    
    // Calculate mean
    for (int j = 0; j < n; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < m; i++)
            mean[j] += data[IDX2(j, i, m)];
        mean[j] /= m;
    }
    
    // Calculate stddev
    for (int j = 0; j < n; j++) {
        stddev[j] = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = data[IDX2(j, i, m)] - mean[j];
            stddev[j] += diff * diff;
        }
        stddev[j] = sqrt(stddev[j] / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize data
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            data[IDX2(j, i, m)] = (data[IDX2(j, i, m)] - mean[j]) / (sqrt_m * stddev[j]);
    
    // Calculate correlation matrix
    for (int i = 0; i < n; i++) {
        corr[IDX2(i, i, n)] = 1.0;
        for (int j = i + 1; j < n; j++) {
            corr[IDX2(i, j, n)] = 0.0;
            for (int k = 0; k < m; k++)
                corr[IDX2(i, j, n)] += data[IDX2(i, k, m)] * data[IDX2(j, k, m)];
            corr[IDX2(j, i, n)] = corr[IDX2(i, j, n)];
        }
    }
}

// threads_static
static void kernel_correlation_threads_static(int m, int n, double* data, double* corr,
                                              double* mean, double* stddev) {
    double sqrt_m = sqrt((double)m);
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < m; i++)
            mean[j] += data[IDX2(j, i, m)];
        mean[j] /= m;
    }
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {
        stddev[j] = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = data[IDX2(j, i, m)] - mean[j];
            stddev[j] += diff * diff;
        }
        stddev[j] = sqrt(stddev[j] / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            data[IDX2(j, i, m)] = (data[IDX2(j, i, m)] - mean[j]) / (sqrt_m * stddev[j]);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        corr[IDX2(i, i, n)] = 1.0;
        for (int j = i + 1; j < n; j++) {
            corr[IDX2(i, j, n)] = 0.0;
            for (int k = 0; k < m; k++)
                corr[IDX2(i, j, n)] += data[IDX2(i, k, m)] * data[IDX2(j, k, m)];
            corr[IDX2(j, i, n)] = corr[IDX2(i, j, n)];
        }
    }
}

// threads_dynamic
static void kernel_correlation_threads_dynamic(int m, int n, double* data, double* corr,
                                               double* mean, double* stddev) {
    double sqrt_m = sqrt((double)m);
    
    #pragma omp parallel for schedule(dynamic, 32)
    for (int j = 0; j < n; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < m; i++)
            mean[j] += data[IDX2(j, i, m)];
        mean[j] /= m;
    }
    
    #pragma omp parallel for schedule(dynamic, 32)
    for (int j = 0; j < n; j++) {
        stddev[j] = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = data[IDX2(j, i, m)] - mean[j];
            stddev[j] += diff * diff;
        }
        stddev[j] = sqrt(stddev[j] / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    #pragma omp parallel for schedule(dynamic, 32)
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            data[IDX2(j, i, m)] = (data[IDX2(j, i, m)] - mean[j]) / (sqrt_m * stddev[j]);
    
    // Dynamic scheduling helps with triangular work distribution
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        corr[IDX2(i, i, n)] = 1.0;
        for (int j = i + 1; j < n; j++) {
            corr[IDX2(i, j, n)] = 0.0;
            for (int k = 0; k < m; k++)
                corr[IDX2(i, j, n)] += data[IDX2(i, k, m)] * data[IDX2(j, k, m)];
            corr[IDX2(j, i, n)] = corr[IDX2(i, j, n)];
        }
    }
}

// tiled
#define TILE_SIZE 32

static void kernel_correlation_tiled(int m, int n, double* data, double* corr,
                                     double* mean, double* stddev) {
    double sqrt_m = sqrt((double)m);
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < m; i++)
            mean[j] += data[IDX2(j, i, m)];
        mean[j] /= m;
        
        stddev[j] = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = data[IDX2(j, i, m)] - mean[j];
            stddev[j] += diff * diff;
        }
        stddev[j] = sqrt(stddev[j] / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
        
        double inv_factor = 1.0 / (sqrt_m * stddev[j]);
        for (int i = 0; i < m; i++)
            data[IDX2(j, i, m)] = (data[IDX2(j, i, m)] - mean[j]) * inv_factor;
    }
    
    // Tiled correlation computation
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += TILE_SIZE) {
        for (int jj = ii; jj < n; jj += TILE_SIZE) {
            int i_end = MIN(ii + TILE_SIZE, n);
            int j_end = MIN(jj + TILE_SIZE, n);
            
            for (int i = ii; i < i_end; i++) {
                int j_start = (ii == jj) ? i : jj;
                for (int j = j_start; j < j_end; j++) {
                    if (i == j) {
                        corr[IDX2(i, i, n)] = 1.0;
                    } else {
                        double sum = 0.0;
                        for (int k = 0; k < m; k++)
                            sum += data[IDX2(i, k, m)] * data[IDX2(j, k, m)];
                        corr[IDX2(i, j, n)] = sum;
                        corr[IDX2(j, i, n)] = sum;
                    }
                }
            }
        }
    }
}

// tasks
static void kernel_correlation_tasks(int m, int n, double* data, double* corr,
                                     double* mean, double* stddev) {
    double sqrt_m = sqrt((double)m);
    int chunk = MAX(n / (omp_get_max_threads() * 4), 1);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Mean and stddev tasks
            for (int jj = 0; jj < n; jj += chunk) {
                #pragma omp task firstprivate(jj)
                {
                    int j_end = MIN(jj + chunk, n);
                    for (int j = jj; j < j_end; j++) {
                        mean[j] = 0.0;
                        for (int i = 0; i < m; i++)
                            mean[j] += data[IDX2(j, i, m)];
                        mean[j] /= m;
                        
                        stddev[j] = 0.0;
                        for (int i = 0; i < m; i++) {
                            double diff = data[IDX2(j, i, m)] - mean[j];
                            stddev[j] += diff * diff;
                        }
                        stddev[j] = sqrt(stddev[j] / m);
                        if (stddev[j] <= EPS) stddev[j] = 1.0;
                        
                        double inv_factor = 1.0 / (sqrt_m * stddev[j]);
                        for (int i = 0; i < m; i++)
                            data[IDX2(j, i, m)] = (data[IDX2(j, i, m)] - mean[j]) * inv_factor;
                    }
                }
            }
            #pragma omp taskwait
            
            // Correlation tasks
            for (int ii = 0; ii < n; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    int i_end = MIN(ii + chunk, n);
                    for (int i = ii; i < i_end; i++) {
                        corr[IDX2(i, i, n)] = 1.0;
                        for (int j = i + 1; j < n; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < m; k++)
                                sum += data[IDX2(i, k, m)] * data[IDX2(j, k, m)];
                            corr[IDX2(i, j, n)] = sum;
                            corr[IDX2(j, i, n)] = sum;
                        }
                    }
                }
            }
        }
    }
}

static double verify_result(int n, const double* corr_ref, const double* corr) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double ref = corr_ref[IDX2(i, j, n)];
            double val = corr[IDX2(i, j, n)];
            double err = fabs(ref - val);
            if (fabs(ref) > EPS) err /= fabs(ref);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

typedef void (*KernelFunc)(int, int, double*, double*, double*, double*);
typedef struct { const char* name; KernelFunc func; } Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential", kernel_correlation_sequential},
    {"threads_static", kernel_correlation_threads_static},
    {"threads_dynamic", kernel_correlation_threads_dynamic},
    {"tiled", kernel_correlation_tiled},
    {"tasks", kernel_correlation_tasks}
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
    
    const DatasetCorr* ds = &DATASETS[dataset_size];
    int m = ds->m, n = ds->n;
    double flops = flops_correlation(m, n);
    
    printf("Correlation Benchmark\n");
    printf("Dataset: %s (M=%d, N=%d)\n", DATASET_NAMES[dataset_size], m, n);
    printf("Threads: %d | FLOPS: %.2e\n\n", threads, flops);
    
    double* data = ALLOC_2D(double, n, m);
    double* data_copy = ALLOC_2D(double, n, m);
    double* corr = ALLOC_2D(double, n, n);
    double* corr_ref = ALLOC_2D(double, n, n);
    double* mean = ALLOC_1D(double, n);
    double* stddev = ALLOC_1D(double, n);
    
    init_array(m, n, data);
    memcpy(data_copy, data, n * m * sizeof(double));
    kernel_correlation_sequential(m, n, data_copy, corr_ref, mean, stddev);
    
    MetricsCollector mc;
    metrics_init(&mc, "correlation", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();
    
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        TimingData timing;
        timing_init(&timing);
        
        for (int w = 0; w < warmup; w++) {
            init_array(m, n, data);
            memset(corr, 0, n * n * sizeof(double));
            STRATEGIES[s].func(m, n, data, corr, mean, stddev);
        }
        
        for (int iter = 0; iter < iterations; iter++) {
            init_array(m, n, data);
            memset(corr, 0, n * n * sizeof(double));
            
            double start = omp_get_wtime();
            STRATEGIES[s].func(m, n, data, corr, mean, stddev);
            double end = omp_get_wtime();
            
            timing_record(&timing, (end - start) * 1000.0);
        }
        
        init_array(m, n, data);
        memset(corr, 0, n * n * sizeof(double));
        STRATEGIES[s].func(m, n, data, corr, mean, stddev);
        double max_err = verify_result(n, corr_ref, corr);
        
        metrics_record(&mc, STRATEGIES[s].name, &timing, flops, max_err < VERIFY_TOLERANCE, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }
    
    if (output_csv) {
        char filename[256], timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));
        snprintf(filename, sizeof(filename), "results/correlation_%s_%s.csv",
                 DATASET_NAMES[dataset_size], timestamp);
        metrics_export_csv(&mc, filename);
    }
    
    FREE_ARRAY(data); FREE_ARRAY(data_copy); FREE_ARRAY(corr);
    FREE_ARRAY(corr_ref); FREE_ARRAY(mean); FREE_ARRAY(stddev);
    
    return 0;
}
