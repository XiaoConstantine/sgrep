.PHONY: build build-hybrid test test-short test-hybrid clean install lint lint-skills \
	bench bench-quick bench-baseline bench-compare bench-profile bench-quality build-bench

# Keep this in sync with .github/workflows/ci.yml (golangci-lint-action version).
GOLANGCI_LINT_VERSION ?= v2.13.1
GOLANGCI_LINT ?= $(shell go env GOPATH)/bin/golangci-lint

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

# Install the pinned golangci-lint with the module's Go toolchain.
# Homebrew bottles are often built with an older Go and refuse to lint this module.
$(GOLANGCI_LINT):
	go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

# Match CI: same binary version, timeout, and sqlite_vec build tag.
lint: $(GOLANGCI_LINT)
	@current="$$($(GOLANGCI_LINT) version --short 2>/dev/null || true)"; \
	want="$(GOLANGCI_LINT_VERSION:v%=%)"; \
	if [ "$$current" != "$$want" ]; then \
		echo "Installing golangci-lint $(GOLANGCI_LINT_VERSION) with $$(go version)"; \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION); \
	fi
	$(GOLANGCI_LINT) run --timeout=5m --build-tags=sqlite_vec ./...

# Validate the canonical open agent skill
lint-skills:
	go test ./plugins/sgrep/skills/sgrep

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
