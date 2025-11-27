.PHONY: build build-hybrid test test-short test-hybrid clean install

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
