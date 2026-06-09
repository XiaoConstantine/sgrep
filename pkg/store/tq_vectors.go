package store

import (
	"container/heap"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

const (
	tqVectorFileName    = "vectors.tqmse"
	tqVectorMagic       = "SGTV"
	tqVectorVersion     = 1
	tqVectorHeaderSize  = 64
	tqVectorDefaultBit  = 4
	tqVectorDefaultSeed = 42
)

// TQVectorStore is a memory-mapped compressed dense-vector artifact.
//
// It stores one TQ-MSE code per chunk and scans those compressed codes with a
// query-side lookup table. SQLite remains responsible for content, metadata,
// and FTS; this artifact owns only first-stage dense vector scoring.
type TQVectorStore struct {
	path string

	mu   sync.RWMutex
	file *os.File
	data []byte

	dims        int
	bits        int
	vectorCount int
	codeSize    int
	codesOffset int

	quantizer *util.TQMSEQuantizer
	chunkIDs  []string
	idToIndex map[string]int
}

// TQVectorBuildOptions configures TQVectorStore export.
type TQVectorBuildOptions struct {
	Dims int
	Bits int
	Seed uint64
}

// TQVectorPath returns the default compact dense vector artifact path.
func TQVectorPath(dir string) string {
	return filepath.Join(dir, tqVectorFileName)
}

// HasTQVectorStore reports whether a compact dense vector artifact exists.
func HasTQVectorStore(dir string) bool {
	info, err := os.Stat(TQVectorPath(dir))
	return err == nil && info.Size() >= tqVectorHeaderSize
}

// RemoveTQVectorStore removes the compact dense vector artifact when present.
func RemoveTQVectorStore(dir string) error {
	if err := os.Remove(TQVectorPath(dir)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// BuildTQVectorStore writes a compact dense vector artifact for chunk vectors.
func BuildTQVectorStore(ctx context.Context, dir string, chunkIDs []string, embeddings [][]float32, opts TQVectorBuildOptions) (int, error) {
	if len(chunkIDs) != len(embeddings) {
		return 0, fmt.Errorf("chunk/vector count mismatch: %d ids, %d vectors", len(chunkIDs), len(embeddings))
	}
	if len(chunkIDs) == 0 {
		return 0, nil
	}
	dims := opts.Dims
	if dims <= 0 {
		dims = getDims()
	}
	bits := opts.Bits
	if bits <= 0 {
		bits = tqVectorDefaultBit
	}
	seed := opts.Seed
	if seed == 0 {
		seed = tqVectorDefaultSeed
	}
	q, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: dims,
		Bits: bits,
		Seed: seed,
	})
	if err != nil {
		return 0, err
	}
	metadata, err := q.Serialize()
	if err != nil {
		return 0, fmt.Errorf("serialize tq vector quantizer: %w", err)
	}

	type item struct {
		id  string
		vec []float32
	}
	items := make([]item, 0, len(chunkIDs))
	for i, id := range chunkIDs {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		vec := embeddings[i]
		if len(vec) != dims {
			return 0, fmt.Errorf("vector %s has %d dims, expected %d", id, len(vec), dims)
		}
		items = append(items, item{id: id, vec: vec})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].id < items[j].id
	})

	indexSize := 0
	for _, item := range items {
		if len(item.id) > int(^uint16(0)) {
			return 0, fmt.Errorf("chunk id too long: %s", item.id)
		}
		indexSize += 2 + len(item.id)
	}
	codeSize := q.CodeSize()
	metadataOffset := tqVectorHeaderSize
	indexOffset := metadataOffset + len(metadata)
	codesOffset := indexOffset + indexSize
	totalSize := codesOffset + len(items)*codeSize

	if err := os.MkdirAll(dir, 0755); err != nil {
		return 0, err
	}
	f, err := os.CreateTemp(dir, tqVectorFileName+".*.tmp")
	if err != nil {
		return 0, err
	}
	tmpPath := f.Name()
	defer func() {
		_ = f.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := f.Truncate(int64(totalSize)); err != nil {
		_ = os.Remove(tmpPath)
		return 0, err
	}

	header := make([]byte, tqVectorHeaderSize)
	copy(header[0:4], tqVectorMagic)
	binary.LittleEndian.PutUint32(header[4:8], tqVectorVersion)
	binary.LittleEndian.PutUint32(header[8:12], uint32(dims))
	binary.LittleEndian.PutUint32(header[12:16], uint32(bits))
	binary.LittleEndian.PutUint32(header[16:20], uint32(len(items)))
	binary.LittleEndian.PutUint32(header[20:24], uint32(codeSize))
	binary.LittleEndian.PutUint32(header[24:28], uint32(metadataOffset))
	binary.LittleEndian.PutUint32(header[28:32], uint32(len(metadata)))
	binary.LittleEndian.PutUint32(header[32:36], uint32(indexOffset))
	binary.LittleEndian.PutUint32(header[36:40], uint32(indexSize))
	binary.LittleEndian.PutUint32(header[40:44], uint32(codesOffset))
	if _, err := f.WriteAt(header, 0); err != nil {
		_ = os.Remove(tmpPath)
		return 0, err
	}
	if _, err := f.WriteAt(metadata, int64(metadataOffset)); err != nil {
		_ = os.Remove(tmpPath)
		return 0, err
	}

	indexBuf := make([]byte, indexSize)
	offset := 0
	for _, item := range items {
		binary.LittleEndian.PutUint16(indexBuf[offset:offset+2], uint16(len(item.id)))
		offset += 2
		copy(indexBuf[offset:], item.id)
		offset += len(item.id)
	}
	if _, err := f.WriteAt(indexBuf, int64(indexOffset)); err != nil {
		_ = os.Remove(tmpPath)
		return 0, err
	}

	encoder := q.NewEncoder()
	code := util.TQMSECode{Codes: make([]byte, codeSize)}
	norm := make([]float32, dims)
	codeOffset := int64(codesOffset)
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			_ = os.Remove(tmpPath)
			return 0, err
		}
		copy(norm, item.vec)
		norm = util.NormalizeVector(norm)
		var err error
		code, err = encoder.EncodeInto(code, norm)
		if err != nil {
			_ = os.Remove(tmpPath)
			return 0, fmt.Errorf("encode vector %s: %w", item.id, err)
		}
		if _, err := f.WriteAt(code.Codes, codeOffset); err != nil {
			_ = os.Remove(tmpPath)
			return 0, err
		}
		codeOffset += int64(codeSize)
	}
	if err := f.Sync(); err != nil {
		_ = os.Remove(tmpPath)
		return 0, err
	}
	if err := f.Close(); err != nil {
		return 0, err
	}
	if err := os.Rename(tmpPath, TQVectorPath(dir)); err != nil {
		return 0, err
	}
	return len(items), nil
}

// OpenTQVectorStore opens an existing compact dense vector artifact.
func OpenTQVectorStore(dir string) (*TQVectorStore, error) {
	s := &TQVectorStore{
		path:      TQVectorPath(dir),
		idToIndex: make(map[string]int),
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *TQVectorStore) load() error {
	f, err := os.OpenFile(s.path, os.O_RDONLY, 0644)
	if err != nil {
		return err
	}
	stat, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return err
	}
	if stat.Size() < tqVectorHeaderSize {
		_ = f.Close()
		return fmt.Errorf("tq vector file too small: %d", stat.Size())
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(stat.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("mmap tq vectors: %w", err)
	}

	if string(data[0:4]) != tqVectorMagic {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("invalid tq vector magic: %q", string(data[0:4]))
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != tqVectorVersion {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("unsupported tq vector version: %d", version)
	}

	dims := int(binary.LittleEndian.Uint32(data[8:12]))
	bits := int(binary.LittleEndian.Uint32(data[12:16]))
	vectorCount := int(binary.LittleEndian.Uint32(data[16:20]))
	codeSize := int(binary.LittleEndian.Uint32(data[20:24]))
	metadataOffset := int(binary.LittleEndian.Uint32(data[24:28]))
	metadataLen := int(binary.LittleEndian.Uint32(data[28:32]))
	indexOffset := int(binary.LittleEndian.Uint32(data[32:36]))
	indexSize := int(binary.LittleEndian.Uint32(data[36:40]))
	codesOffset := int(binary.LittleEndian.Uint32(data[40:44]))

	if metadataOffset < tqVectorHeaderSize || metadataOffset+metadataLen > len(data) {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("invalid tq vector metadata bounds")
	}
	if indexOffset < metadataOffset+metadataLen || indexOffset+indexSize > len(data) {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("invalid tq vector index bounds")
	}
	if codesOffset < indexOffset+indexSize || codesOffset+vectorCount*codeSize > len(data) {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("invalid tq vector code bounds")
	}

	q, err := util.DeserializeTQMSEQuantizer(data[metadataOffset : metadataOffset+metadataLen])
	if err != nil {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("load tq vector quantizer: %w", err)
	}
	if q.Dims() != dims || q.Bits() != bits || q.CodeSize() != codeSize {
		_ = syscall.Munmap(data)
		_ = f.Close()
		return fmt.Errorf("tq vector quantizer/header mismatch")
	}

	chunkIDs := make([]string, vectorCount)
	idToIndex := make(map[string]int, vectorCount)
	offset := indexOffset
	for i := 0; i < vectorCount; i++ {
		if offset+2 > indexOffset+indexSize {
			_ = syscall.Munmap(data)
			_ = f.Close()
			return fmt.Errorf("truncated tq vector index")
		}
		idLen := int(binary.LittleEndian.Uint16(data[offset : offset+2]))
		offset += 2
		if offset+idLen > indexOffset+indexSize {
			_ = syscall.Munmap(data)
			_ = f.Close()
			return fmt.Errorf("truncated tq vector id")
		}
		id := string(data[offset : offset+idLen])
		offset += idLen
		chunkIDs[i] = id
		idToIndex[id] = i
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.file = f
	s.data = data
	s.dims = dims
	s.bits = bits
	s.vectorCount = vectorCount
	s.codeSize = codeSize
	s.codesOffset = codesOffset
	s.quantizer = q
	s.chunkIDs = chunkIDs
	s.idToIndex = idToIndex
	return nil
}

// Close unmaps and closes the vector artifact.
func (s *TQVectorStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var err error
	if s.data != nil {
		err = syscall.Munmap(s.data)
		s.data = nil
	}
	if s.file != nil {
		if closeErr := s.file.Close(); err == nil {
			err = closeErr
		}
		s.file = nil
	}
	return err
}

// VectorCount returns the number of compressed vectors.
func (s *TQVectorStore) VectorCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.vectorCount
}

// Dims returns the vector dimension.
func (s *TQVectorStore) Dims() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.dims
}

// Bits returns the per-coordinate bit width.
func (s *TQVectorStore) Bits() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.bits
}

// Search returns compressed dense-vector nearest neighbors.
func (s *TQVectorStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]DenseSearchResult, error) {
	if limit <= 0 {
		return nil, nil
	}

	s.mu.RLock()
	data := s.data
	q := s.quantizer
	chunkIDs := s.chunkIDs
	vectorCount := s.vectorCount
	codeSize := s.codeSize
	codesOffset := s.codesOffset
	dims := s.dims
	s.mu.RUnlock()

	if len(data) == 0 || q == nil || vectorCount == 0 {
		return nil, nil
	}
	if len(embedding) != dims {
		return nil, fmt.Errorf("query has %d dims, expected %d", len(embedding), dims)
	}

	query := util.NormalizeVectorCopy(embedding)
	prepared, err := q.PrepareQuery(query)
	if err != nil {
		return nil, err
	}

	h := make(tqResultMaxHeap, 0, limit)
	code := util.TQMSECode{}
	for i := 0; i < vectorCount; i++ {
		if i&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		start := codesOffset + i*codeSize
		code.Codes = data[start : start+codeSize]
		dot := q.Dot(prepared, code)
		distance := 1.0 - dot
		if distance > threshold {
			continue
		}
		item := DenseSearchResult{ID: chunkIDs[i], Distance: distance}
		if h.Len() < limit {
			heap.Push(&h, item)
			continue
		}
		if distance < h[0].Distance {
			h[0] = item
			heap.Fix(&h, 0)
		}
	}

	results := make([]DenseSearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(&h).(DenseSearchResult)
	}
	return results, nil
}

type tqResultMaxHeap []DenseSearchResult

func (h tqResultMaxHeap) Len() int { return len(h) }
func (h tqResultMaxHeap) Less(i, j int) bool {
	return h[i].Distance > h[j].Distance
}
func (h tqResultMaxHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *tqResultMaxHeap) Push(x interface{}) {
	*h = append(*h, x.(DenseSearchResult))
}

func (h *tqResultMaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
