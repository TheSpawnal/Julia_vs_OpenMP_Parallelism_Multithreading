/*
 * Nussinov Benchmark - RNA Secondary Structure Prediction
 * Dynamic programming algorithm for maximum base pairing
 * 
 * This is a dependency-limited benchmark with anti-diagonal parallelism
 */

 /*
 * FIX PATCH — benchmark_nussinov.c
 *
 * Drop-in replacement for:
 *   - kernel_nussinov_tiled   (FAIL max_error = 1, off-by-one at tile bdry)
 *
 * APPLICATION:
 *   In src/benchmark_nussinov.c, locate `static void kernel_nussinov_tiled(...)`
 *   and replace its entire body with the block below. The signature and
 *   STRATEGIES[] table are unchanged.
 *
 *   The existing file has  #define TILE_SIZE 64  at the top of the kernels.
 *   This patch uses NUSSINOV_TILE_SIZE (independent macro) so the original
 *   TILE_SIZE can remain if used elsewhere. If it is unused elsewhere,
 *   either macro may be removed.
 *
 * DEPENDENCIES ALREADY IN THE FILE:
 *   #include "benchmark_common.h"  (provides IDX2, MIN, MAX)
 *   Kernel-local helpers: max_score, match, base type
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>
#include <omp.h>

typedef int base;

#ifndef NUSSINOV_TILE_SIZE
#define NUSSINOV_TILE_SIZE 64
#endif

static const int DATASETS[] = {
    60,    // MINI
    180,   // SMALL
    500,   // MEDIUM
    2500,  // LARGE
    5500   // EXTRALARGE
};

// Base pairing: A-U (0-3) and C-G (1-2)
static inline int match(base b1, base b2) {
    return (b1 + b2) == 3 ? 1 : 0;
}

static inline int max_score(int s1, int s2) {
    return s1 >= s2 ? s1 : s2;
}

static void init_array(int n, base* seq, int* table) {
    for (int i = 0; i < n; i++)
        seq[i] = i % 4;
    
    for (int i = 0; i < n * n; i++)
        table[i] = 0;
}

// Sequential baseline (anti-diagonal sweep)
static void kernel_nussinov_sequential(int n, base* seq, int* table) {
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            if (j - 1 >= 0)
                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i, j-1, n)]);
            
            if (i + 1 < n)
                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i+1, j, n)]);
            
            if (j - 1 >= 0 && i + 1 < n) {
                if (i < j - 1)
                    table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                     table[IDX2(i+1, j-1, n)] + match(seq[i], seq[j]));
                else
                    table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                     match(seq[i], seq[j]));
            }
            
            for (int k = i + 1; k < j; k++)
                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                 table[IDX2(i, k, n)] + table[IDX2(k+1, j, n)]);
        }
    }
}

// Wavefront parallelization (anti-diagonal)
static void kernel_nussinov_wavefront(int n, base* seq, int* table) {
    // Process by anti-diagonals (cells on same anti-diagonal are independent)
    for (int diag = 1; diag < n; diag++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n - diag; i++) {
            int j = i + diag;
            
            if (j - 1 >= 0)
                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i, j-1, n)]);
            
            if (i + 1 < n)
                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i+1, j, n)]);
            
            if (j - 1 >= 0 && i + 1 < n) {
                if (i < j - 1)
                    table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                     table[IDX2(i+1, j-1, n)] + match(seq[i], seq[j]));
                else
                    table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                     match(seq[i], seq[j]));
            }
            
            for (int k = i + 1; k < j; k++)
                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                 table[IDX2(i, k, n)] + table[IDX2(k+1, j, n)]);
        }
    }
}

// Tiled wavefront (better cache behavior)
// #define TILE_SIZE 64

// static void kernel_nussinov_tiled(int n, base* seq, int* table) {
//     // Tile the anti-diagonal computation
//     int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
//     // Process tile diagonals
//     for (int tile_diag = 0; tile_diag < 2 * num_tiles - 1; tile_diag++) {
//         #pragma omp parallel for schedule(dynamic)
//         for (int ti = MAX(0, tile_diag - num_tiles + 1); ti <= MIN(tile_diag, num_tiles - 1); ti++) {
//             int tj = tile_diag - ti;
//             if (tj < ti) continue;
            
//             int i_start = ti * TILE_SIZE;
//             int j_start = tj * TILE_SIZE;
//             int i_end = MIN(i_start + TILE_SIZE, n);
//             int j_end = MIN(j_start + TILE_SIZE, n);
            
//             // Process tile (anti-diagonal within tile)
//             for (int i = i_end - 1; i >= i_start; i--) {
//                 for (int j = MAX(j_start, i + 1); j < j_end; j++) {
//                     if (j - 1 >= 0)
//                         table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i, j-1, n)]);
                    
//                     if (i + 1 < n)
//                         table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i+1, j, n)]);
                    
//                     if (j - 1 >= 0 && i + 1 < n) {
//                         if (i < j - 1)
//                             table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
//                                                              table[IDX2(i+1, j-1, n)] + match(seq[i], seq[j]));
//                         else
//                             table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
//                                                              match(seq[i], seq[j]));
//                     }
                    
//                     for (int k = i + 1; k < j; k++)
//                         table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
//                                                          table[IDX2(i, k, n)] + table[IDX2(k+1, j, n)]);
//                 }
//             }
//         }
//     }
// }

/* ======================================================================== */
/* FIX: kernel_nussinov_tiled                                                */
/*                                                                           */
/* Root cause: the original wavefront used tile_diag = T_i + T_j. But       */
/* Nussinov iterates i from n-1 down to 0, so cell (i,j) depends on         */
/* (i+1, j) in the next row DOWN. In tile coordinates, the dependency is on */
/* tile (T_i+1, T_j), whose T_i + T_j is LARGER. The broken code therefore  */
/* processed dependencies AFTER their dependents.                            */
/*                                                                           */
/* The correct wavefront is d_tile = T_j - T_i (tile-stripe distance). All  */
/* inter-tile dependencies fall on strictly smaller d_tile; tiles on the    */
/* same d_tile are independent.                                              */
/* ======================================================================== */
static void kernel_nussinov_tiled(int n, base* seq, int* table) {
    const int TS = NUSSINOV_TILE_SIZE;
    const int num_tiles = (n + TS - 1) / TS;
 
    for (int d_tile = 0; d_tile < num_tiles; d_tile++) {
        /* OpenMP canonical form: loop bound must be var relop invariant-expr.
         * Use a precomputed count instead of  ti + d_tile < num_tiles. */
        const int count = num_tiles - d_tile;
        #pragma omp parallel for schedule(dynamic)
        for (int ti = 0; ti < count; ti++) {
            const int tj = ti + d_tile;
            const int i_start = ti * TS;
            const int j_start = tj * TS;
            const int i_end   = MIN(i_start + TS, n);
            const int j_end   = MIN(j_start + TS, n);
 
            /* Within-tile: sequential Nussinov order. i descends, j ascends. */
            for (int i = i_end - 1; i >= i_start; i--) {
                for (int j = MAX(j_start, i + 1); j < j_end; j++) {
                    int score = table[IDX2(i, j, n)];
 
                    /* Pair case: i pairs with j */
                    if (j - 1 >= 0 && i + 1 < n) {
                        if (i < j - 1)
                            score = max_score(score,
                                              table[IDX2(i + 1, j - 1, n)]
                                              + match(seq[i], seq[j]));
                        else
                            score = max_score(score, match(seq[i], seq[j]));
                    }
                    /* i unpaired */
                    if (i + 1 < n)
                        score = max_score(score, table[IDX2(i + 1, j, n)]);
                    /* j unpaired */
                    if (j - 1 >= 0)
                        score = max_score(score, table[IDX2(i, j - 1, n)]);
                    /* Bifurcation */
                    for (int k = i + 1; k < j; k++)
                        score = max_score(score,
                                          table[IDX2(i, k, n)]
                                          + table[IDX2(k + 1, j, n)]);
 
                    table[IDX2(i, j, n)] = score;
                }
            }
        }
    }
}
// Task-based with dependencies
static void kernel_nussinov_tasks(int n, base* seq, int* table) {
    int chunk = MAX(n / 16, 16);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Process by anti-diagonals with tasks
            for (int diag = 1; diag < n; diag++) {
                // Create tasks for this diagonal
                for (int ii = 0; ii < n - diag; ii += chunk) {
                    #pragma omp task firstprivate(ii, diag)
                    {
                        int i_end = MIN(ii + chunk, n - diag);
                        for (int i = ii; i < i_end; i++) {
                            int j = i + diag;
                            
                            if (j - 1 >= 0)
                                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i, j-1, n)]);
                            
                            if (i + 1 < n)
                                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)], table[IDX2(i+1, j, n)]);
                            
                            if (j - 1 >= 0 && i + 1 < n) {
                                if (i < j - 1)
                                    table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                                     table[IDX2(i+1, j-1, n)] + match(seq[i], seq[j]));
                                else
                                    table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                                     match(seq[i], seq[j]));
                            }
                            
                            for (int k = i + 1; k < j; k++)
                                table[IDX2(i, j, n)] = max_score(table[IDX2(i, j, n)],
                                                                 table[IDX2(i, k, n)] + table[IDX2(k+1, j, n)]);
                        }
                    }
                }
                #pragma omp taskwait  // Barrier between diagonals
            }
        }
    }
}

// SIMD for inner k-loop
static void kernel_nussinov_simd(int n, base* seq, int* table) {
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            int score = 0;
            
            if (j - 1 >= 0)
                score = max_score(score, table[IDX2(i, j-1, n)]);
            
            if (i + 1 < n)
                score = max_score(score, table[IDX2(i+1, j, n)]);
            
            if (j - 1 >= 0 && i + 1 < n) {
                if (i < j - 1)
                    score = max_score(score, table[IDX2(i+1, j-1, n)] + match(seq[i], seq[j]));
                else
                    score = max_score(score, match(seq[i], seq[j]));
            }
            
            // SIMD-friendly k-loop (reduction)
            int max_k = 0;
            #pragma omp simd reduction(max:max_k)
            for (int k = i + 1; k < j; k++) {
                int ks = table[IDX2(i, k, n)] + table[IDX2(k+1, j, n)];
                if (ks > max_k) max_k = ks;
            }
            
            table[IDX2(i, j, n)] = max_score(score, max_k);
        }
    }
}

static int verify_result(int n, const int* table_ref, const int* table) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (table_ref[IDX2(i, j, n)] != table[IDX2(i, j, n)])
                return 0;
    return 1;
}

typedef void (*KernelFunc)(int, base*, int*);
typedef struct { const char* name; KernelFunc func; } Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential", kernel_nussinov_sequential},
    {"wavefront", kernel_nussinov_wavefront},
    {"tiled", kernel_nussinov_tiled},
    {"tasks", kernel_nussinov_tasks},
    {"simd", kernel_nussinov_simd}
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
    double flops = flops_nussinov(n);
    
    printf("Nussinov Benchmark (RNA Folding)\n");
    printf("Dataset: %s (N=%d)\n", DATASET_NAMES[dataset_size], n);
    printf("Threads: %d | FLOPS: %.2e\n", threads, flops);
    printf("NOTE: Dependency-limited benchmark, expect modest speedups\n\n");
    
    base* seq = ALLOC_1D(base, n);
    int* table = ALLOC_2D(int, n, n);
    int* table_ref = ALLOC_2D(int, n, n);
    
    init_array(n, seq, table);
    memcpy(table_ref, table, n * n * sizeof(int));
    kernel_nussinov_sequential(n, seq, table_ref);
    
    MetricsCollector mc;
    metrics_init(&mc, "nussinov", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();
    
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        TimingData timing;
        timing_init(&timing);
        
        for (int w = 0; w < warmup; w++) {
            init_array(n, seq, table);
            STRATEGIES[s].func(n, seq, table);
        }
        
        for (int iter = 0; iter < iterations; iter++) {
            init_array(n, seq, table);
            
            double start = omp_get_wtime();
            STRATEGIES[s].func(n, seq, table);
            double end = omp_get_wtime();
            
            timing_record(&timing, (end - start) * 1000.0);
        }
        
        init_array(n, seq, table);
        STRATEGIES[s].func(n, seq, table);
        int verified = verify_result(n, table_ref, table);
        
        metrics_record(&mc, STRATEGIES[s].name, &timing, flops, verified, verified ? 0.0 : 1.0);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }
    
    if (output_csv) {
        char filename[256], timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));
        snprintf(filename, sizeof(filename), "results/nussinov_%s_%s.csv",
                 DATASET_NAMES[dataset_size], timestamp);
        metrics_export_csv(&mc, filename);
    }
    
    FREE_ARRAY(seq);
    FREE_ARRAY(table);
    FREE_ARRAY(table_ref);
    
    return 0;
}
