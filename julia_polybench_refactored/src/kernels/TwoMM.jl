module TwoMM
#=
PolyBench 2MM Kernel - Julia Implementation
REFACTORED VERSION

Computation: D = alpha * A * B * C + beta * D
  Step 1: tmp = alpha * A * B
  Step 2: D = tmp * C + beta * D

=============================================================================
DESIGN PRINCIPLES
=============================================================================
1. Zero-allocation hot paths (verified with @allocated)
2. Column-major optimized loop order (j outer for Julia)
3. Consistent Float64 for fair OpenMP comparison
4. @simd @inbounds on all inner loops
5. Scalar hoisting outside SIMD loops

=============================================================================
STRATEGIES
=============================================================================
1. sequential      - Baseline with SIMD, single-threaded
2. threads_static  - Static scheduling over columns
3. threads_dynamic - Dynamic scheduling for load balance
4. tiled           - Cache-blocked with parallel outer loop
5. blas            - BLAS mul! (reference, uses BLAS internal threading)
6. tasks           - Coarse-grained task parallelism

Author: SpawnAl / Falkor collaboration
=============================================================================
=#

using LinearAlgebra
using Base.Threads

export init_2mm!, reset_2mm!
export kernel_2mm_seq!, kernel_2mm_threads_static!, kernel_2mm_threads_dynamic!
export kernel_2mm_tiled!, kernel_2mm_blas!, kernel_2mm_tasks!
export STRATEGIES_2MM, get_kernel_2mm

#=============================================================================
 Strategy Registry
=============================================================================#
const STRATEGIES_2MM = [
    "sequential",
    "threads_static",
    "threads_dynamic",
    "tiled",
    "blas",
    "tasks"
]

#=============================================================================
 Initialization - PolyBench Compatible
 
 Matrix dimensions:
   A: ni x nk
   B: nk x nj
   tmp: ni x nj (intermediate)
   C: nj x nl
   D: ni x nl (result, also input)
=============================================================================#
function init_2mm!(
    alpha::Ref{Float64}, beta::Ref{Float64},
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    nj = size(B, 2)
    nl = size(C, 2)
    
    alpha[] = 1.5
    beta[] = 1.2
    
    # Column-major initialization (j outer for cache efficiency)
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = ((i - 1) * (j - 1) + 1) % ni / Float64(ni)
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = (i - 1) * j % nj / Float64(nj)
    end
    
    @inbounds for j in 1:nl, i in 1:nj
        C[i, j] = ((i - 1) * (j + 2) + 1) % nl / Float64(nl)
    end
    
    @inbounds for j in 1:nl, i in 1:ni
        D[i, j] = (i - 1) * (j + 1) % nk / Float64(nk)
    end
    
    fill!(tmp, 0.0)
    
    return nothing
end

"""
Reset arrays for re-benchmarking.
Call before each timed iteration.
"""
function reset_2mm!(tmp::Matrix{Float64}, D::Matrix{Float64}, D_orig::Matrix{Float64})
    fill!(tmp, 0.0)
    copyto!(D, D_orig)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential Baseline
 
 - Single-threaded with SIMD vectorization
 - Column-major loop order for Julia
 - Zero allocations in hot path
=============================================================================#
function kernel_2mm_seq!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B
    @inbounds for j in 1:nj
        for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @inbounds for j in 1:nl
        @simd for i in 1:ni
            D[i, j] *= beta
        end
        for k in 1:nj
            c_kj = C[k, j]
            @simd for i in 1:ni
                D[i, j] += tmp[i, k] * c_kj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded with Static Scheduling
 
 - Parallelizes outer (column) loop
 - Static scheduling: columns divided equally among threads
 - Best for uniform workload
=============================================================================#
function kernel_2mm_threads_static!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B
    @threads :static for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @threads :static for j in 1:nl
        @inbounds begin
            @simd for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                @simd for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Threaded with Dynamic Scheduling
 
 - Work-stealing for load balance
 - Better for irregular workloads or NUMA systems
=============================================================================#
function kernel_2mm_threads_dynamic!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B
    @threads :dynamic for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @threads :dynamic for j in 1:nl
        @inbounds begin
            @simd for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                @simd for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Tiled/Blocked with Parallel Outer Loop
 
 - Cache optimization through blocking
 - Tile size tuned for L2 cache (~256KB per core)
 - Outer tile loop parallelized
=============================================================================#
function kernel_2mm_tiled!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64};
    tile_size::Int=64
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    ts = tile_size
    
    # tmp = alpha * A * B (tiled)
    @threads :static for jj in 1:ts:nj
        j_end = min(jj + ts - 1, nj)
        @inbounds for kk in 1:ts:nk
            k_end = min(kk + ts - 1, nk)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        b_kj = alpha * B[k, j]
                        @simd for i in ii:i_end
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    # D = beta * D + tmp * C (tiled)
    @threads :static for jj in 1:ts:nl
        j_end = min(jj + ts - 1, nl)
        @inbounds for j in jj:j_end
            @simd for i in 1:ni
                D[i, j] *= beta
            end
        end
        @inbounds for kk in 1:ts:nj
            k_end = min(kk + ts - 1, nj)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        c_kj = C[k, j]
                        @simd for i in ii:i_end
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: BLAS Reference
 
 - Uses optimized BLAS mul!
 - Reference upper bound for matrix operations
 - BLAS handles its own threading internally
=============================================================================#
function kernel_2mm_blas!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    # tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0)
    
    # D = tmp * C + beta * D
    mul!(D, tmp, C, 1.0, beta)
    
    return nothing
end

#=============================================================================
 Strategy 6: Task-based Parallelism
 
 - Coarse-grained tasks for column blocks
 - Useful for irregular or large workloads
=============================================================================#
function kernel_2mm_tasks!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64};
    num_tasks::Int=nthreads()
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B (task-parallel)
    chunk_j = cld(nj, num_tasks)
    @sync begin
        for t in 1:num_tasks
            j_start = (t - 1) * chunk_j + 1
            j_end = min(t * chunk_j, nj)
            j_start > nj && continue
            
            @spawn begin
                @inbounds for j in j_start:j_end
                    for k in 1:nk
                        b_kj = alpha * B[k, j]
                        @simd for i in 1:ni
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    # D = beta * D + tmp * C (task-parallel)
    chunk_l = cld(nl, num_tasks)
    @sync begin
        for t in 1:num_tasks
            j_start = (t - 1) * chunk_l + 1
            j_end = min(t * chunk_l, nl)
            j_start > nl && continue
            
            @spawn begin
                @inbounds for j in j_start:j_end
                    @simd for i in 1:ni
                        D[i, j] *= beta
                    end
                    for k in 1:nj
                        c_kj = C[k, j]
                        @simd for i in 1:ni
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Kernel Dispatcher
=============================================================================#
function get_kernel_2mm(strategy::AbstractString)
    s = lowercase(String(strategy))
    
    if s == "sequential" || s == "seq"
        return kernel_2mm_seq!
    elseif s == "threads_static" || s == "threads" || s == "static"
        return kernel_2mm_threads_static!
    elseif s == "threads_dynamic" || s == "dynamic"
        return kernel_2mm_threads_dynamic!
    elseif s == "tiled" || s == "blocked"
        return kernel_2mm_tiled!
    elseif s == "blas"
        return kernel_2mm_blas!
    elseif s == "tasks"
        return kernel_2mm_tasks!
    else
        error("Unknown 2MM strategy: $strategy. Available: $(join(STRATEGIES_2MM, ", "))")
    end
end

end # module TwoMM
