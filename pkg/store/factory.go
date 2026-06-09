//go:build !sqlite_vec
// +build !sqlite_vec

package store

import (
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
	useTQ := backend == "tqmse" || (backend != "sqlite" && backend != "libsql" && HasTQVectorStore(repoDir))
	var (
		s   Storer
		err error
	)
	if useTQ {
		s, err = OpenSQLiteMetadata(path)
	} else {
		s, err = OpenLibSQL(path)
	}
	if err != nil {
		return nil, err
	}
	return OpenTQSearchStoreIfAvailable(s, repoDir)
}

// OpenForStats opens a store for stats queries.
func OpenForStats(path string) (Storer, error) {
	s, err := OpenSQLiteMetadata(path)
	if err != nil {
		return nil, err
	}
	return OpenTQSearchStoreIfAvailable(s, filepath.Dir(path))
}
