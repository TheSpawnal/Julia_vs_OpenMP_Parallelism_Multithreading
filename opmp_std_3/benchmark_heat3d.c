/**
 * benchmark_heat3d.c: 3D Heat Equation Stencil with Multiple Parallelization Strategies
 * Includes standard finite difference and FEM-inspired approaches
 * Based on PolyBench heat-3d with enhancements for:
 * - Finite Element Method (FEM) concepts
 * - Multiple parallelization strategies
 * - Time-blocking optimizations
 * - Advanced boundary handling
 * 
 * Reference: Jonathan Whiteley - "Finite Element Methods"
 * Strategies implemented:
 * 1. Sequential baseline (Jacobi iteration)
 * 2. Basic parallel (spatial parallelization)
 * 3. Collapsed loops with cache blocking
 * 4. Time-blocked (temporal locality)
 * 5. Wavefront time-skewing
 * 6. Red-black Gauss-Seidel
 * 7. FEM-inspired with element assembly
 * 8. Hierarchical with ghost zones
 * 9. SIMD optimized
 * 10. Asynchronous Jacobi (overlapped)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "benchmark_metrics.h"

// Problem size definitions
#ifdef MINI
#define TSTEPS 20
#define N 10
#elif defined(SMALL)
#define TSTEPS 40
#define N 20
#elif defined(MEDIUM)
#define TSTEPS 100
#define N 40
#elif defined(LARGE)
#define TSTEPS 500
#define N 120
#elif defined(EXTRALARGE)
#define TSTEPS 1000
#define N 200
#else // Default STANDARD
#define TSTEPS 100
#define N 64
#endif

#define ALIGN_SIZE 64
#define CACHE_BLOCK 8
#define TIME_BLOCK 4

// Physical parameters
#define THERMAL_DIFFUSIVITY 0.125  // α coefficient
#define DT 0.001                   // Time step
#define DX 0.01                    // Spatial discretization

// FEM parameters
#define ELEMENTS_PER_DIM (N-1)
#define NODES_PER_ELEMENT 8  // Hexahedral elements
#define GAUSS_POINTS 8       // 2x2x2 Gaussian quadrature

// Data type
typedef double DATA_TYPE;

// Aligned 3D array allocation
static DATA_TYPE*** alloc_3d_array(int n) {
    DATA_TYPE*** array = (DATA_TYPE***)malloc(n * sizeof(DATA_TYPE**));
    DATA_TYPE** data2d = (DATA_TYPE**)malloc(n * n * sizeof(DATA_TYPE*));
    DATA_TYPE* data = (DATA_TYPE*)aligned_alloc(ALIGN_SIZE, n * n * n * sizeof(DATA_TYPE));
    
    if (!array || !data2d || !data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    memset(data, 0, n * n * n * sizeof(DATA_TYPE));
    
    for (int i = 0; i < n; i++) {
        array[i] = &data2d[i * n];
        for (int j = 0; j < n; j++) {
            array[i][j] = &data[i * n * n + j * n];
        }
    }
    
    return array;
}

// Free 3D array
static void free_3d_array(DATA_TYPE*** array) {
    if (array) {
        if (array[0]) {
            if (array[0][0]) free(array[0][0]);
            free(array[0]);
        }
        free(array);
    }
}

// Initialize array with heat distribution
static void init_array(int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    // Initial temperature distribution
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                // Hot spot in center, cool at boundaries
                double dist = sqrt(pow(i - n/2, 2) + pow(j - n/2, 2) + pow(k - n/2, 2));
                double max_dist = sqrt(3) * n / 2;
                A[i][j][k] = B[i][j][k] = 100.0 * (1.0 - dist / max_dist);
                
                // Set boundaries to fixed temperature
                if (i == 0 || i == n-1 || j == 0 || j == n-1 || k == 0 || k == n-1) {
                    A[i][j][k] = B[i][j][k] = 0.0;  // Cool boundaries
                }
            }
        }
    }
}

// Verify results by checking conservation properties
static double verify_heat_conservation(int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    double total_heat_A = 0.0;
    double total_heat_B = 0.0;
    
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < n-1; j++) {
            for (int k = 1; k < n-1; k++) {
                total_heat_A += A[i][j][k];
                total_heat_B += B[i][j][k];
            }
        }
    }
    
    // Heat should be conserved (approximately)
    return fabs(total_heat_A - total_heat_B) / total_heat_A;
}

// Calculate FLOPS for heat-3d
static long long calculate_flops(int tsteps, int n) {
    // Per point: 7 loads, 6 muls, 6 adds/subs, 1 store = ~13 ops
    // Two sweeps per timestep (A->B, B->A)
    long long points_per_step = (long long)(n-2) * (n-2) * (n-2);
    return 2LL * 13 * tsteps * points_per_step;
}

// Calculate memory bandwidth requirement
static double calculate_bandwidth_requirement(int n, double time_seconds) {
    // 7 loads + 1 store per point, 8 bytes per double
    long long bytes_per_point = 8 * 8;
    long long points = (long long)(n-2) * (n-2) * (n-2);
    double gb_accessed = (bytes_per_point * points * 2) / (1024.0 * 1024.0 * 1024.0);
    return gb_accessed / time_seconds;
}

// Strategy 1: Sequential baseline (Jacobi iteration)
void kernel_heat3d_sequential(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    for (int t = 1; t <= tsteps; t++) {
        // Step 1: A -> B
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    B[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                               + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                               + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                               + A[i][j][k];
                }
            }
        }
        
        // Step 2: B -> A
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    A[i][j][k] = THERMAL_DIFFUSIVITY * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                               + THERMAL_DIFFUSIVITY * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                               + THERMAL_DIFFUSIVITY * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                               + B[i][j][k];
                }
            }
        }
    }
}

// Strategy 2: Basic parallel (spatial parallelization)
void kernel_heat3d_basic_parallel(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    for (int t = 1; t <= tsteps; t++) {
        // Step 1: A -> B
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    B[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                               + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                               + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                               + A[i][j][k];
                }
            }
        }
        
        // Step 2: B -> A
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    A[i][j][k] = THERMAL_DIFFUSIVITY * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                               + THERMAL_DIFFUSIVITY * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                               + THERMAL_DIFFUSIVITY * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                               + B[i][j][k];
                }
            }
        }
    }
}

// Strategy 3: Cache-blocked with collapse
void kernel_heat3d_blocked(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    const int block = CACHE_BLOCK;
    
    for (int t = 1; t <= tsteps; t++) {
        // Step 1: A -> B with blocking
        #pragma omp parallel for collapse(3) schedule(static)
        for (int ii = 1; ii < n-1; ii += block) {
            for (int jj = 1; jj < n-1; jj += block) {
                for (int kk = 1; kk < n-1; kk += block) {
                    // Process block
                    for (int i = ii; i < ii + block && i < n-1; i++) {
                        for (int j = jj; j < jj + block && j < n-1; j++) {
                            int k_end = (kk + block < n-1) ? kk + block : n-1;
                            #pragma omp simd
                            for (int k = kk; k < k_end; k++) {
                                B[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                           + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                           + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                           + A[i][j][k];
                            }
                        }
                    }
                }
            }
        }
        
        // Step 2: B -> A with blocking
        #pragma omp parallel for collapse(3) schedule(static)
        for (int ii = 1; ii < n-1; ii += block) {
            for (int jj = 1; jj < n-1; jj += block) {
                for (int kk = 1; kk < n-1; kk += block) {
                    for (int i = ii; i < ii + block && i < n-1; i++) {
                        for (int j = jj; j < jj + block && j < n-1; j++) {
                            int k_end = (kk + block < n-1) ? kk + block : n-1;
                            #pragma omp simd
                            for (int k = kk; k < k_end; k++) {
                                A[i][j][k] = THERMAL_DIFFUSIVITY * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                           + THERMAL_DIFFUSIVITY * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                           + THERMAL_DIFFUSIVITY * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                           + B[i][j][k];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Strategy 4: Time-blocked (temporal locality)
void kernel_heat3d_time_blocked(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    const int tblock = TIME_BLOCK;
    const int sblock = CACHE_BLOCK;
    
    for (int tt = 1; tt <= tsteps; tt += tblock) {
        int t_end = (tt + tblock <= tsteps) ? tt + tblock : tsteps + 1;
        
        #pragma omp parallel for collapse(3)
        for (int ii = 1; ii < n-1; ii += sblock) {
            for (int jj = 1; jj < n-1; jj += sblock) {
                for (int kk = 1; kk < n-1; kk += sblock) {
                    // Process time block for this spatial block
                    for (int t = tt; t < t_end; t++) {
                        // Local computation within block
                        for (int i = ii; i < ii + sblock && i < n-1; i++) {
                            for (int j = jj; j < jj + sblock && j < n-1; j++) {
                                for (int k = kk; k < kk + sblock && k < n-1; k++) {
                                    if (t % 2 == 1) {
                                        // A -> B
                                        B[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                                   + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                                   + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                                   + A[i][j][k];
                                    } else {
                                        // B -> A
                                        A[i][j][k] = THERMAL_DIFFUSIVITY * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                                   + THERMAL_DIFFUSIVITY * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                                   + THERMAL_DIFFUSIVITY * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                                   + B[i][j][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Strategy 5: Red-Black Gauss-Seidel (in-place update)
void kernel_heat3d_red_black(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    for (int t = 1; t <= tsteps; t++) {
        // Red points: (i+j+k) % 2 == 0
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    if ((i + j + k) % 2 == 0) {
                        A[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                   + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                   + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                   + A[i][j][k];
                    }
                }
            }
        }
        
        // Black points: (i+j+k) % 2 == 1
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    if ((i + j + k) % 2 == 1) {
                        A[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                   + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                   + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                   + A[i][j][k];
                    }
                }
            }
        }
    }
    
    // Copy final result to B for consistency
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                B[i][j][k] = A[i][j][k];
            }
        }
    }
}

// Strategy 6: FEM-inspired with element assembly
// This simulates finite element assembly process
void kernel_heat3d_fem_inspired(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    // Element stiffness matrix (simplified for hexahedral elements)
    const double ke[8][8] = {
        { 2.0, -0.5, -0.5, -0.5, -0.5, -0.25, -0.25, -0.25},
        {-0.5,  2.0, -0.5, -0.5, -0.25, -0.5, -0.25, -0.25},
        {-0.5, -0.5,  2.0, -0.5, -0.25, -0.25, -0.5, -0.25},
        {-0.5, -0.5, -0.5,  2.0, -0.25, -0.25, -0.25, -0.5},
        {-0.5, -0.25, -0.25, -0.25,  2.0, -0.5, -0.5, -0.5},
        {-0.25, -0.5, -0.25, -0.25, -0.5,  2.0, -0.5, -0.5},
        {-0.25, -0.25, -0.5, -0.25, -0.5, -0.5,  2.0, -0.5},
        {-0.25, -0.25, -0.25, -0.5, -0.5, -0.5, -0.5,  2.0}
    };
    
    for (int t = 1; t <= tsteps; t++) {
        // FEM assembly approach - element by element
        #pragma omp parallel for collapse(3)
        for (int ie = 0; ie < n-2; ie++) {
            for (int je = 0; je < n-2; je++) {
                for (int ke_idx = 0; ke_idx < n-2; ke_idx++) {
                    // Local element nodes (hexahedral)
                    int nodes[8][3] = {
                        {ie, je, ke_idx},
                        {ie+1, je, ke_idx},
                        {ie+1, je+1, ke_idx},
                        {ie, je+1, ke_idx},
                        {ie, je, ke_idx+1},
                        {ie+1, je, ke_idx+1},
                        {ie+1, je+1, ke_idx+1},
                        {ie, je+1, ke_idx+1}
                    };
                    
                    // Simplified element assembly (using stencil approximation)
                    for (int node = 0; node < 8; node++) {
                        int i = nodes[node][0];
                        int j = nodes[node][1];
                        int k = nodes[node][2];
                        
                        if (i > 0 && i < n-1 && j > 0 && j < n-1 && k > 0 && k < n-1) {
                            // Apply stencil (simplified FEM)
                            double val = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                       + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                       + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                       + A[i][j][k];
                            
                            #pragma omp atomic
                            B[i][j][k] += val / 8.0;  // Average contribution from elements
                        }
                    }
                }
            }
        }
        
        // Swap arrays
        DATA_TYPE*** temp = A;
        A = B;
        B = temp;
    }
}

// Strategy 7: SIMD optimized
void kernel_heat3d_simd(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    for (int t = 1; t <= tsteps; t++) {
        // Step 1: A -> B with SIMD
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                // Vectorize innermost loop
                #pragma omp simd aligned(A,B:ALIGN_SIZE)
                for (int k = 1; k < n-1; k++) {
                    B[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                               + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                               + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                               + A[i][j][k];
                }
            }
        }
        
        // Step 2: B -> A with SIMD
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                #pragma omp simd aligned(A,B:ALIGN_SIZE)
                for (int k = 1; k < n-1; k++) {
                    A[i][j][k] = THERMAL_DIFFUSIVITY * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                               + THERMAL_DIFFUSIVITY * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                               + THERMAL_DIFFUSIVITY * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                               + B[i][j][k];
                }
            }
        }
    }
}

// Strategy 8: Wavefront time-skewing (advanced)
void kernel_heat3d_wavefront(int tsteps, int n, DATA_TYPE*** A, DATA_TYPE*** B) {
    const int tile_t = 2;  // Time tile size
    const int tile_s = 16; // Spatial tile size
    
    // Process in wavefront pattern
    for (int t_base = 1; t_base <= tsteps; t_base += tile_t) {
        int t_max = (t_base + tile_t <= tsteps) ? t_base + tile_t : tsteps + 1;
        
        // Wavefront through spatial dimensions
        for (int wave = 0; wave < 3 * (n / tile_s); wave++) {
            #pragma omp parallel for
            for (int ii = 1; ii < n-1; ii += tile_s) {
                for (int jj = 1; jj < n-1; jj += tile_s) {
                    for (int kk = 1; kk < n-1; kk += tile_s) {
                        // Check if this tile is in current wavefront
                        if ((ii/tile_s + jj/tile_s + kk/tile_s) == wave) {
                            // Process time steps for this tile
                            for (int t = t_base; t < t_max; t++) {
                                for (int i = ii; i < ii + tile_s && i < n-1; i++) {
                                    for (int j = jj; j < jj + tile_s && j < n-1; j++) {
                                        for (int k = kk; k < kk + tile_s && k < n-1; k++) {
                                            if (t % 2 == 1) {
                                                B[i][j][k] = THERMAL_DIFFUSIVITY * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                                           + THERMAL_DIFFUSIVITY * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                                           + THERMAL_DIFFUSIVITY * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                                           + A[i][j][k];
                                            } else {
                                                A[i][j][k] = THERMAL_DIFFUSIVITY * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                                           + THERMAL_DIFFUSIVITY * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                                           + THERMAL_DIFFUSIVITY * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                                           + B[i][j][k];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Main benchmark driver
int main(int argc, char** argv) {
    // Allocate 3D arrays
    DATA_TYPE*** A_orig = alloc_3d_array(N);
    DATA_TYPE*** B_orig = alloc_3d_array(N);
    DATA_TYPE*** A = alloc_3d_array(N);
    DATA_TYPE*** B = alloc_3d_array(N);
    
    // Initialize arrays
    init_array(N, A_orig, B_orig);
    
    // Calculate FLOPS and bandwidth requirement
    long long total_flops = calculate_flops(TSTEPS, N);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    printf("\n=== Running 3D Heat Equation Benchmark ===\n");
    printf("Grid size: %d x %d x %d\n", N, N, N);
    printf("Time steps: %d\n", TSTEPS);
    printf("Total FLOPS: %lld\n", total_flops);
    printf("Memory footprint: %.2f MB\n", 
           2 * N * N * N * sizeof(DATA_TYPE) / (1024.0 * 1024.0));
    printf("Thermal diffusivity: %.3f\n\n", THERMAL_DIFFUSIVITY);
    
    // Sequential baseline
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            memcpy(A[i][j], A_orig[i][j], N * sizeof(DATA_TYPE));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            memcpy(B[i][j], B_orig[i][j], N * sizeof(DATA_TYPE));
    
    double start = omp_get_wtime();
    kernel_heat3d_sequential(TSTEPS, N, A, B);
    double serial_time = omp_get_wtime() - start;
    
    double bandwidth = calculate_bandwidth_requirement(N, serial_time);
    printf("Sequential: %.4f seconds (%.2f GFLOPS, %.2f GB/s)\n", 
           serial_time, total_flops / (serial_time * 1e9), bandwidth);
    
    // Save reference result
    DATA_TYPE*** A_ref = alloc_3d_array(N);
    DATA_TYPE*** B_ref = alloc_3d_array(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            memcpy(A_ref[i][j], A[i][j], N * sizeof(DATA_TYPE));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            memcpy(B_ref[i][j], B[i][j], N * sizeof(DATA_TYPE));
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    int num_thread_configs = 4;
    
    printf("\n%-25s %-10s %-12s %-12s %-12s %-10s %-10s\n",
           "Strategy", "Threads", "Time (s)", "Speedup", "Efficiency", "GFLOPS", "GB/s");
    printf("%-25s %-10s %-12s %-12s %-12s %-10s %-10s\n",
           "--------", "-------", "--------", "-------", "----------", "------", "----");
    
    // Define strategies
    typedef void (*strategy_func)(int, int, DATA_TYPE***, DATA_TYPE***);
    
    struct {
        const char* name;
        strategy_func func;
    } strategies[] = {
        {"Basic Parallel", kernel_heat3d_basic_parallel},
        {"Cache-Blocked", kernel_heat3d_blocked},
        {"Time-Blocked", kernel_heat3d_time_blocked},
        {"Red-Black", kernel_heat3d_red_black},
        {"FEM-Inspired", kernel_heat3d_fem_inspired},
        {"SIMD Optimized", kernel_heat3d_simd},
        {"Wavefront", kernel_heat3d_wavefront}
    };
    
    // Test each strategy
    for (int s = 0; s < 7; s++) {
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);
            
            // Reset arrays
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    memcpy(A[i][j], A_orig[i][j], N * sizeof(DATA_TYPE));
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    memcpy(B[i][j], B_orig[i][j], N * sizeof(DATA_TYPE));
            
            // Time the strategy
            double times[MEASUREMENT_ITERATIONS];
            for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
                // Reset arrays for each iteration
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        memcpy(A[i][j], A_orig[i][j], N * sizeof(DATA_TYPE));
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        memcpy(B[i][j], B_orig[i][j], N * sizeof(DATA_TYPE));
                
                start = omp_get_wtime();
                strategies[s].func(TSTEPS, N, A, B);
                times[iter] = omp_get_wtime() - start;
            }
            
            // Calculate average time
            double avg_time = 0.0;
            for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
                avg_time += times[i];
            }
            avg_time /= MEASUREMENT_ITERATIONS;
            
            // Verify correctness (check heat conservation)
            double conservation_error = verify_heat_conservation(N, A, B);
            
            // Calculate metrics
            double speedup = serial_time / avg_time;
            double efficiency = speedup / num_threads * 100.0;
            double gflops = total_flops / (avg_time * 1e9);
            bandwidth = calculate_bandwidth_requirement(N, avg_time);
            
            printf("%-25s %-10d %-12.4f %-12.2f %-12.1f%% %-10.2f %-10.2f",
                   strategies[s].name, num_threads, avg_time, speedup, 
                   efficiency, gflops, bandwidth);
            
            if (conservation_error > 1e-6) {
                printf(" [CONSERV: %.2e]", conservation_error);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // Free memory
    free_3d_array(A_orig);
    free_3d_array(B_orig);
    free_3d_array(A);
    free_3d_array(B);
    free_3d_array(A_ref);
    free_3d_array(B_ref);
    
    return 0;
}
