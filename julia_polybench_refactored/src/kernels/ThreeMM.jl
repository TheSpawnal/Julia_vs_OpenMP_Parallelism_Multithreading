module ThreeMM
# Stub - implement similar to TwoMM
using LinearAlgebra
using Base.Threads

export STRATEGIES_3MM
const STRATEGIES_3MM = ["sequential", "threads_static", "threads_dynamic", "tiled", "blas", "tasks"]

end
