//go:build sqlite_vec
// +build sqlite_vec

package store

import (
	"fmt"
	"os"
	"path/filepath"
)

// OpenDefault opens the default store implementation (BufferedStore with sqlite-vec).
func OpenDefault(path string, quantization QuantizationMode) (Storer, error) {
	return OpenBuffered(path, WithBufferedQuantization(quantization))
}

// OpenForSearch prefers compact TQ-MSE artifacts just like the libSQL build.
func OpenForSearch(path string) (Storer, error) {
	repoDir := filepath.Dir(path)
	backend := os.Getenv("SGREP_VECTOR_BACKEND")
	if backend == "sqlite" || backend == "libsql" {
		legacy, err := OpenBuffered(path)
		if err != nil {
			return nil, err
		}
		if HasTQVectorStore(repoDir) && legacy.VectorCount() == 0 {
			_ = legacy.Close()
			return nil, fmt.Errorf("SGREP_VECTOR_BACKEND=%s requires an index built with --sql-vectors", backend)
		}
		return legacy, nil
	}
	if backend == "tqmse" && !HasTQVectorStore(repoDir) {
		return nil, fmt.Errorf("SGREP_VECTOR_BACKEND=tqmse but %s is missing", TQVectorPath(repoDir))
	}
	if HasTQVectorStore(repoDir) {
		metadata, err := OpenSQLiteMetadata(path)
		if err == nil {
			wrapped, wrapErr := OpenTQSearchStoreIfAvailable(metadata, repoDir)
			if wrapErr == nil {
				if _, ok := wrapped.(*TQSearchStore); ok {
					return wrapped, nil
				}
				_ = wrapped.Close()
			} else {
				_ = metadata.Close()
				if backend == "tqmse" {
					return nil, wrapErr
				}
			}
		}
	}
	return OpenBuffered(path)
}

// OpenForStats opens a store for stats queries.
func OpenForStats(path string) (Storer, error) {
	return OpenInMem(path)
}
