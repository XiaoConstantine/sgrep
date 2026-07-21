//go:build sqlite_vec

package store

import (
	"context"
	"path/filepath"
	"testing"
)

func TestSQLiteVecOpenForSearchUsesCompactArtifact(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "")
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "index.db")
	base, err := OpenDefault(path, QuantizeInt8)
	if err != nil {
		t.Fatal(err)
	}
	doc := &Document{ID: "doc", FilePath: "doc.go", Content: "func doc() {}", StartLine: 1, EndLine: 1}
	metadataStore, ok := base.(MetadataBatchStorer)
	if !ok {
		t.Fatalf("default sqlite store %T lacks MetadataBatchStorer", base)
	}
	if err := metadataStore.StoreMetadataBatch(ctx, []*Document{doc}); err != nil {
		t.Fatal(err)
	}
	if err := base.Close(); err != nil {
		t.Fatal(err)
	}
	query := make([]float32, getDims())
	query[0] = 1
	if _, err := BuildTQVectorStore(ctx, dir, []string{doc.ID}, [][]float32{query}, TQVectorBuildOptions{Dims: getDims()}); err != nil {
		t.Fatal(err)
	}

	searchStore, err := OpenForSearch(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = searchStore.Close() }()
	docs, _, err := searchStore.Search(ctx, query, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != doc.ID {
		t.Fatalf("compact sqlite search = %+v", docs)
	}
}

func TestSQLiteVecForcedSQLBackendRejectsCompactIndex(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "sqlite")
	dir := t.TempDir()
	path := filepath.Join(dir, "index.db")
	base, err := OpenDefault(path, QuantizeInt8)
	if err != nil {
		t.Fatal(err)
	}
	if err := base.Close(); err != nil {
		t.Fatal(err)
	}
	query := make([]float32, getDims())
	query[0] = 1
	if _, err := BuildTQVectorStore(context.Background(), dir, []string{"doc"}, [][]float32{query}, TQVectorBuildOptions{Dims: getDims()}); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenForSearch(path); err == nil {
		t.Fatal("forced SQL backend accepted a compact index with no SQL vectors")
	}
}
