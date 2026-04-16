module Config
#=
Configuration Module for Julia PolyBench Benchmarks
REFACTORED: Consistent FLOP calculations aligned with PolyBench standard

=============================================================================
FLOP COUNTING METHODOLOGY
=============================================================================

PolyBench uses the following conventions:
- Each fused multiply-add (FMA) counts as 2 FLOPs
- Matrix multiplication C = A * B where A is m x k, B is k x n:
  FLOPs = 2 * m * n * k (multiply + add for each element)
  
- For algorithms with multiple operations, sum all FLOPs
- Division and sqrt count as 1 FLOP each (simplification)

=============================================================================
=#

using LinearAlgebra
using Base.Threads

export configure_blas_threads, print_system_info
export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY
export DATASETS_CORRELATION, DATASETS_JACOBI2D, DATASETS_NUSSINOV
export flops_2mm, flops_3mm, flops_cholesky
export flops_correlation, flops_jacobi2d, flops_nussinov
export memory_bytes_2mm, memory_bytes_3mm

#=============================================================================
 BLAS Configuration
 
 Critical for fair benchmarking:
 - When Julia uses multiple threads for OUR code, set BLAS to 1 thread
   to avoid oversubscription
 - When testing BLAS strategy specifically, we let BLAS use its own threading
=============================================================================#
function configure_blas_threads(;verbose::Bool=false, for_blas_benchmark::Bool=false)
    if for_blas_benchmark
        # Let BLAS use all cores for BLAS-specific benchmarks
        BLAS.set_num_threads(Sys.CPU_THREADS)
        verbose && println("BLAS threads: $(Sys.CPU_THREADS) (BLAS benchmark mode)")
    elseif Threads.nthreads() > 1
        # Disable BLAS threading when Julia is multi-threaded
        BLAS.set_num_threads(1)
        verbose && println("BLAS threads: 1 (Julia using $(Threads.nthreads()) threads)")
    else
        # Single Julia thread - let BLAS use moderate threading
        BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
        verbose && println("BLAS threads: $(min(4, Sys.CPU_THREADS)) (Julia single-threaded)")
    end
end

function print_system_info()
    println("System Configuration:")
    println("  Julia version: $(VERSION)")
    println("  Julia threads: $(Threads.nthreads())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  BLAS vendor: $(BLAS.vendor())")
    println("  CPU cores: $(Sys.CPU_THREADS)")
    println("  Total memory: $(round(Sys.total_memory() / 1024^3, digits=1)) GB")
end

#=============================================================================
 Dataset Sizes - PolyBench Standard
 
 These match the official PolyBench/C 4.2.1 dataset definitions
=============================================================================#

# 2MM: D = alpha*A*B*C + beta*D
# A: ni x nk, B: nk x nj, tmp: ni x nj, C: nj x nl, D: ni x nl
const DATASETS_2MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl), NTuple{4, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=22,   nl=24),
    "SMALL"      => (ni=40,   nj=50,   nk=70,   nl=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=210,  nl=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

# 3MM: G = (A*B)*(C*D)
# A: ni x nk, B: nk x nj, C: nj x nm, D: nm x nl
# E: ni x nj (temp), F: nj x nl (temp), G: ni x nl (result)
const DATASETS_3MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl, :nm), NTuple{5, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=20,   nl=22,   nm=24),
    "SMALL"      => (ni=40,   nj=50,   nk=60,   nl=70,   nm=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=200,  nl=210,  nm=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

# Cholesky decomposition: L*L^T = A
const DATASETS_CHOLESKY = Dict{String, Int}(
    "MINI"       => 40,
    "SMALL"      => 120,
    "MEDIUM"     => 400,
    "LARGE"      => 2000,
    "EXTRALARGE" => 4000
)

# Correlation: Compute correlation matrix from data matrix
const DATASETS_CORRELATION = Dict{String, NamedTuple{(:m, :n), NTuple{2, Int}}}(
    "MINI"       => (m=28,   n=32),
    "SMALL"      => (m=80,   n=100),
    "MEDIUM"     => (m=240,  n=260),
    "LARGE"      => (m=1200, n=1400),
    "EXTRALARGE" => (m=2600, n=3000)
)

# Jacobi 2D stencil
const DATASETS_JACOBI2D = Dict{String, NamedTuple{(:n, :tsteps), NTuple{2, Int}}}(
    "MINI"       => (n=30,   tsteps=20),
    "SMALL"      => (n=90,   tsteps=40),
    "MEDIUM"     => (n=250,  tsteps=100),
    "LARGE"      => (n=1300, tsteps=500),
    "EXTRALARGE" => (n=2800, tsteps=1000)
)

# Nussinov RNA folding
const DATASETS_NUSSINOV = Dict{String, Int}(
    "MINI"       => 60,
    "SMALL"      => 180,
    "MEDIUM"     => 500,
    "LARGE"      => 2500,
    "EXTRALARGE" => 5500
)

#=============================================================================
 FLOP Calculations
 
 Following PolyBench methodology:
 - Multiply-add = 2 FLOPs
 - Division = 1 FLOP
 - Sqrt = 1 FLOP
=============================================================================#

"""
    flops_2mm(ni, nj, nk, nl) -> Float64

FLOPs for 2MM: D = alpha * A * B * C + beta * D
  tmp = alpha * A * B : 2 * ni * nj * nk
  D = tmp * C + beta*D : 2 * ni * nl * nj + ni * nl (scale)
"""
function flops_2mm(ni::Int, nj::Int, nk::Int, nl::Int)::Float64
    # tmp = A * B (ni x nj result from ni x nk and nk x nj)
    flops_tmp = 2.0 * ni * nj * nk
    # D = tmp * C (ni x nl result from ni x nj and nj x nl)
    flops_d = 2.0 * ni * nl * nj
    # beta * D (scaling)
    flops_scale = Float64(ni * nl)
    return flops_tmp + flops_d + flops_scale
end

"""
    flops_3mm(ni, nj, nk, nl, nm) -> Float64

FLOPs for 3MM: G = (A * B) * (C * D)
  E = A * B : 2 * ni * nj * nk
  F = C * D : 2 * nj * nl * nm
  G = E * F : 2 * ni * nl * nj
"""
function flops_3mm(ni::Int, nj::Int, nk::Int, nl::Int, nm::Int)::Float64
    flops_e = 2.0 * ni * nj * nk       # E = A * B
    flops_f = 2.0 * nj * nl * nm       # F = C * D
    flops_g = 2.0 * ni * nl * nj       # G = E * F
    return flops_e + flops_f + flops_g
end

"""
    flops_cholesky(n) -> Float64

FLOPs for Cholesky decomposition: L*L^T = A
Approximately n^3/3 for standard algorithm
"""
function flops_cholesky(n::Int)::Float64
    return Float64(n)^3 / 3.0
end

"""
    flops_correlation(m, n) -> Float64

FLOPs for correlation matrix computation:
  1. Mean computation: m * n
  2. Stddev computation: 2 * m * n + n (sqrt)
  3. Normalization: m * n
  4. Correlation matrix: 2 * n * n * m
"""
function flops_correlation(m::Int, n::Int)::Float64
    flops_mean = Float64(m * n)
    flops_stddev = 2.0 * m * n + n  # includes sqrt
    flops_normalize = Float64(m * n)
    flops_corr = 2.0 * n * n * m
    return flops_mean + flops_stddev + flops_normalize + flops_corr
end

"""
    flops_jacobi2d(n, tsteps) -> Float64

FLOPs for Jacobi 2D stencil:
  - Each interior point: 4 adds + 1 multiply = 5 FLOPs
  - Interior points: (n-2)^2
  - Two updates per timestep (A->B, B->A)
"""
function flops_jacobi2d(n::Int, tsteps::Int)::Float64
    interior_points = (n - 2)^2
    flops_per_point = 5.0  # 4 additions + 1 multiply (by 0.2)
    return 2.0 * tsteps * interior_points * flops_per_point
end

"""
    flops_nussinov(n) -> Float64

FLOPs for Nussinov RNA folding (approximate):
  - O(n^3) for the split search
  - Each cell: comparisons and additions
"""
function flops_nussinov(n::Int)::Float64
    # Approximate: sum over all diagonals
    total = 0.0
    for d in 1:(n-1)
        cells_on_diag = n - d
        # Each cell does O(d) comparisons for split
        for i in 1:cells_on_diag
            j = i + d
            total += 3.0  # base cases
            total += Float64(j - i - 1) * 2.0  # split search
        end
    end
    return total
end

#=============================================================================
 Memory Calculations (bytes)
=============================================================================#

"""Memory footprint for 2MM benchmark in bytes"""
function memory_bytes_2mm(ni::Int, nj::Int, nk::Int, nl::Int)::Int
    # A(ni,nk) + B(nk,nj) + tmp(ni,nj) + C(nj,nl) + D(ni,nl) + D_orig(ni,nl)
    return (ni*nk + nk*nj + ni*nj + nj*nl + 2*ni*nl) * sizeof(Float64)
end

"""Memory footprint for 3MM benchmark in bytes"""
function memory_bytes_3mm(ni::Int, nj::Int, nk::Int, nl::Int, nm::Int)::Int
    # A(ni,nk) + B(nk,nj) + C(nj,nm) + D(nm,nl) + E(ni,nj) + F(nj,nl) + G(ni,nl) + G_ref
    return (ni*nk + nk*nj + nj*nm + nm*nl + ni*nj + nj*nl + 2*ni*nl) * sizeof(Float64)
end

end # module Config
