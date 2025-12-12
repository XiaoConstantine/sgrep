package store

import (
	"math/rand"
	"os"
	"testing"
)

func TestMMapVectorStore(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dims := 768

	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "mmap_vec_test")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	// Create store
	store, err := OpenMMapVectorStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}

	// Create test vectors
	testVectors := map[string][]float32{
		"chunk_001": randomVectorWithRng(rng, dims),
		"chunk_002": randomVectorWithRng(rng, dims),
		"chunk_003": randomVectorWithRng(rng, dims),
	}

	// Write vectors
	store.BeginWrite()
	for id, vec := range testVectors {
		store.WriteVector(id, vec)
	}
	if err := store.CommitWrite(); err != nil {
		t.Fatal(err)
	}

	t.Logf("Wrote %d vectors", store.VectorCount())

	// Read back and verify
	for chunkID, expected := range testVectors {
		got := store.GetVector(chunkID)
		if got == nil {
			t.Errorf("GetVector(%s) returned nil", chunkID)
			continue
		}
		if len(got) != dims {
			t.Errorf("%s: expected %d dims, got %d", chunkID, dims, len(got))
			continue
		}
		// Verify values match
		for i := 0; i < dims; i++ {
			if got[i] != expected[i] {
				t.Errorf("%s[%d]: expected %f, got %f", chunkID, i, expected[i], got[i])
				break
			}
		}
	}

	// Test GetAllVectors
	ids, vecs := store.GetAllVectors()
	if len(ids) != 3 || len(vecs) != 3 {
		t.Errorf("GetAllVectors: expected 3 vectors, got %d ids, %d vecs", len(ids), len(vecs))
	}

	// Close and reopen
	_ = store.Close()

	store2, err := OpenMMapVectorStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store2.Close() }()

	// Verify data persisted
	if store2.VectorCount() != 3 {
		t.Errorf("After reopen: expected 3 vectors, got %d", store2.VectorCount())
	}

	vec := store2.GetVector("chunk_002")
	if vec == nil {
		t.Error("GetVector(chunk_002) returned nil after reopen")
	}

	t.Logf("MMap vector store test passed with %d vectors", store2.VectorCount())
}

func randomVectorWithRng(rng *rand.Rand, dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}
	return vec
}
