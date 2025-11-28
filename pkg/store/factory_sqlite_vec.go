//go:build sqlite_vec
// +build sqlite_vec

package store

// OpenDefault opens the default store implementation (BufferedStore with sqlite-vec).
func OpenDefault(path string, quantization QuantizationMode) (Storer, error) {
	return OpenBuffered(path, WithBufferedQuantization(quantization))
}

// OpenForSearch opens a store optimized for search operations.
func OpenForSearch(path string) (Storer, error) {
	return OpenBuffered(path)
}

// OpenForStats opens a store for stats queries.
func OpenForStats(path string) (Storer, error) {
	return OpenInMem(path)
}
