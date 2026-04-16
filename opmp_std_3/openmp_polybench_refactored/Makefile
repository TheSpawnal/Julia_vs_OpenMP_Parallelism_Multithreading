# OpenMP PolyBench Benchmark Suite - Makefile
# Aligned with Julia PolyBench implementation

CC ?= gcc

# DEFAULT: Portable build (no architecture-specific instructions)
# Use 'make native' for machine-specific optimizations
CFLAGS = -O3 -fopenmp -Wall -Wextra -Wno-stringop-truncation
LDFLAGS = -fopenmp -lm

# Include paths
INCLUDES = -I./include

# Source and object directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

# Common sources
COMMON_SRC = $(SRC_DIR)/metrics.c
COMMON_OBJ = $(OBJ_DIR)/metrics.o

# Benchmark sources
BENCHMARKS = benchmark_2mm benchmark_3mm benchmark_cholesky benchmark_correlation benchmark_nussinov

# Results directory
RESULTS_DIR = results

.PHONY: all clean results benchmarks help test native avx2 sse4 das5 debug

all: dirs $(BENCHMARKS)
	@echo "Built with PORTABLE flags (no SIMD). Use 'make native' for optimizations."

dirs:
	@mkdir -p $(OBJ_DIR) $(RESULTS_DIR)

# Common object file
$(OBJ_DIR)/metrics.o: $(SRC_DIR)/metrics.c include/metrics.h include/benchmark_common.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Individual targets (explicit to avoid pattern rule issues)
benchmark_2mm: $(SRC_DIR)/benchmark_2mm.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_3mm: $(SRC_DIR)/benchmark_3mm.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_cholesky: $(SRC_DIR)/benchmark_cholesky.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_correlation: $(SRC_DIR)/benchmark_correlation.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_nussinov: $(SRC_DIR)/benchmark_nussinov.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

# Debug build
debug: CFLAGS = -O0 -g -fopenmp -Wall -Wextra -DDEBUG
debug: clean all

# NATIVE: Machine-specific optimizations (use on target machine only)
native: CFLAGS = -O3 -march=native -mtune=native -fopenmp -ffast-math -Wall
native: clean all
	@echo "Built with NATIVE optimizations. Binary is machine-specific."

# AVX2: For modern CPUs (Haswell 2013+, Ryzen, etc.)
avx2: CFLAGS = -O3 -march=haswell -mavx2 -mfma -fopenmp -Wall
avx2: clean all
	@echo "Built with AVX2. Requires Haswell+ or Ryzen+ CPU."

# SSE4: For older CPUs (Nehalem 2008+)
sse4: CFLAGS = -O3 -march=nehalem -msse4.2 -fopenmp -Wall
sse4: clean all
	@echo "Built with SSE4.2. Requires Nehalem+ CPU."

# DAS-5 cluster (Intel Xeon E5-2630-v3 = Haswell)
das5: CFLAGS = -O3 -march=haswell -mavx2 -mfma -fopenmp -Wall
das5: clean all
	@echo "Built for DAS-5 cluster (Haswell)."

# Run all benchmarks with LARGE dataset
run: all
	@echo "Running all benchmarks with LARGE dataset..."
	@for bench in $(BENCHMARKS); do \
		echo ""; \
		echo "=== $$bench ==="; \
		./$$bench --dataset LARGE --output csv; \
	done

# Run quick tests with MINI dataset
test: all
	@echo "Running quick tests with MINI dataset..."
	@for bench in $(BENCHMARKS); do \
		echo ""; \
		./$$bench --dataset MINI --iterations 3 --warmup 1; \
	done

# Scaling study (1, 2, 4, 8, 16 threads)
scaling: all
	@echo "Running scaling study..."
	@for bench in $(BENCHMARKS); do \
		echo ""; \
		echo "=== Scaling: $$bench ==="; \
		for t in 1 2 4 8 16; do \
			echo "Threads: $$t"; \
			OMP_NUM_THREADS=$$t ./$$bench --dataset LARGE --threads $$t --iterations 5 --output csv; \
		done; \
	done

# Clean build artifacts
clean:
	rm -f $(BENCHMARKS)
	rm -rf $(OBJ_DIR)
	rm -f *.o

# Deep clean including results
distclean: clean
	rm -rf $(RESULTS_DIR)

# Help
help:
	@echo "OpenMP PolyBench Benchmark Suite"
	@echo ""
	@echo "Build targets:"
	@echo "  all       - Portable build (DEFAULT, works on any x86-64)"
	@echo "  native    - Machine-specific optimizations (compile on target!)"
	@echo "  avx2      - AVX2 build (Haswell 2013+, Ryzen+)"
	@echo "  sse4      - SSE4.2 build (Nehalem 2008+)"
	@echo "  das5      - DAS-5 cluster (Haswell)"
	@echo "  debug     - Debug symbols, no optimization"
	@echo ""
	@echo "Run targets:"
	@echo "  test      - Quick test with MINI dataset"
	@echo "  run       - Run all benchmarks with LARGE dataset"
	@echo "  scaling   - Thread scaling study (1,2,4,8,16 threads)"
	@echo "  clean     - Remove build artifacts"
	@echo "  distclean - Remove all generated files"
	@echo ""
	@echo "Benchmark options:"
	@echo "  --dataset SIZE    MINI, SMALL, MEDIUM, LARGE, EXTRALARGE"
	@echo "  --iterations N    Number of timed iterations"
	@echo "  --warmup N        Number of warmup iterations"
	@echo "  --threads N       Number of OpenMP threads"
	@echo "  --output csv      Export results to CSV"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Portable build"
	@echo "  make native       # Optimized for this machine"
	@echo "  ./benchmark_2mm --dataset LARGE --threads 8 --output csv"
