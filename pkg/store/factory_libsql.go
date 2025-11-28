//go:build libsql
// +build libsql

package store

// OpenDefault opens the default store implementation (LibSQLStore with DiskANN).
func OpenDefault(path string, quantization QuantizationMode) (Storer, error) {
	return OpenLibSQL(path, WithLibSQLQuantization(quantization))
}

// OpenForSearch opens a store optimized for search operations.
func OpenForSearch(path string) (Storer, error) {
	return OpenLibSQL(path)
}

// OpenForStats opens a store for stats queries.
func OpenForStats(path string) (Storer, error) {
	return OpenLibSQL(path)
}
