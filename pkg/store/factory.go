//go:build !sqlite_vec
// +build !sqlite_vec

package store

import (
	"fmt"
	"os"
	"path/filepath"
)

// OpenDefault opens the default store implementation (LibSQLStore with DiskANN).
func OpenDefault(path string, quantization QuantizationMode) (Storer, error) {
	return OpenLibSQL(path, WithLibSQLQuantization(quantization))
}

// OpenForSearch opens a store optimized for search operations.
func OpenForSearch(path string) (Storer, error) {
	repoDir := filepath.Dir(path)
	backend := os.Getenv("SGREP_VECTOR_BACKEND")
	if backend == "tqmse" {
		return openTQForSearch(path, repoDir, true)
	}
	if backend != "sqlite" && backend != "libsql" && HasTQVectorStore(repoDir) {
		s, err := openTQForSearch(path, repoDir, false)
		if err != nil {
			return nil, err
		}
		if s != nil {
			return s, nil
		}
	}
	return OpenLibSQL(path)
}

func openTQForSearch(path, repoDir string, forced bool) (Storer, error) {
	s, err := OpenSQLiteMetadata(path)
	if err != nil {
		return nil, err
	}
	wrapped, err := OpenTQSearchStoreIfAvailable(s, repoDir)
	if err != nil {
		return nil, err
	}
	tq, ok := wrapped.(*TQSearchStore)
	if !ok {
		_ = wrapped.Close()
		return nil, nil
	}
	if tq.fileDense != nil {
		return tq, nil
	}
	fileStore, err := OpenLibSQL(path, WithLibSQLReadOnly(true))
	if err != nil {
		_ = tq.Close()
		if forced {
			return nil, fmt.Errorf("open file embedding delegate: %w", err)
		}
		return nil, nil
	}
	tq.fileStore = fileStore
	tq.fileStoreCloser = fileStore
	return tq, nil
}

// OpenForStats opens a store for stats queries.
func OpenForStats(path string) (Storer, error) {
	s, err := OpenSQLiteMetadata(path)
	if err != nil {
		return nil, err
	}
	return OpenTQSearchStoreIfAvailable(s, filepath.Dir(path))
}
