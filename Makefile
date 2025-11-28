.PHONY: build build-hybrid test test-short test-hybrid clean install \
	bench bench-quick bench-baseline bench-compare bench-profile bench-quality build-bench

# Default build (semantic search only)
build:
	go build -o sgrep ./cmd/sgrep

# Build with hybrid search support (semantic + BM25 via SQLite FTS5)
build-hybrid:
	CGO_CFLAGS="-DSQLITE_ENABLE_FTS5" go build -o sgrep ./cmd/sgrep

# Run all tests
test:
	go test -race ./...

# Run quick tests (skips integration tests requiring llama-server)
test-short:
	go test -short ./...

# Run tests with FTS5 support (for hybrid search tests)
test-hybrid:
	CGO_CFLAGS="-DSQLITE_ENABLE_FTS5" go test -short ./...

# Run tests with coverage
test-cover:
	go test -short -cover ./...

# Run tests with coverage and FTS5 support
test-cover-hybrid:
	CGO_CFLAGS="-DSQLITE_ENABLE_FTS5" go test -short -cover ./...

# Clean build artifacts
clean:
	rm -f sgrep sgrep-local

# Install to GOPATH/bin
install:
	go install ./cmd/sgrep

# Install with hybrid search support
install-hybrid:
	CGO_CFLAGS="-DSQLITE_ENABLE_FTS5" go install ./cmd/sgrep

# ============ Benchmarks ============

# Run performance benchmarks
bench:
	go test ./... -run=^$$ -bench=. -benchmem -tags=sqlite_vec 2>&1 | tee bench.log

# Run quick benchmarks (skip large tests)
bench-quick:
	go test ./... -run=^$$ -bench=. -benchmem -tags=sqlite_vec -short

# Run benchmarks and save as baseline
bench-baseline:
	./scripts/perf_bench.sh --save

# Run benchmarks and compare to baseline
bench-compare:
	./scripts/perf_bench.sh

# Run benchmarks with CPU/memory profiling
bench-profile:
	./scripts/perf_bench.sh --profile

# Run quality evaluation
bench-quality:
	go run ./cmd/sgrep-bench quality -codebase . -dataset bench/quality/dataset.json

# Build benchmark CLI
build-bench:
	go build -o sgrep-bench ./cmd/sgrep-bench
