package store

import (
	"container/heap"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

const (
	tqVectorFileName     = "vectors.tqmse"
	tqFileVectorFileName = "file_vectors.tqmse"
	tqVectorMagic        = "SGTV"
	tqVectorVersion      = 1
	tqVectorHeaderSize   = 64
	tqVectorDefaultBit   = 4
	tqVectorDefaultSeed  = 42
)

// TQVectorStore is a memory-mapped compressed dense-vector artifact.
//
// It stores one TQ-MSE code per item and scans those compressed codes with a
// query-side lookup table. SQLite remains responsible for content, metadata,
// and FTS; this artifact owns dense vector scoring.
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
	ids       []string
	idToIndex map[string]int
}

// ContentSHA256 identifies the exact mapped sidecar bytes.
func (s *TQVectorStore) ContentSHA256() string {
	if s == nil {
		return ""
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	sum := sha256.Sum256(s.data)
	return hex.EncodeToString(sum[:])
}

// TQVectorBuildOptions configures TQVectorStore export.
type TQVectorBuildOptions struct {
	Dims int
	Bits int
	Seed uint64
}

// TQVectorAccumulator incrementally encodes vectors for a compact dense-vector artifact.
type TQVectorAccumulator struct {
	dims      int
	bits      int
	quantizer *util.TQMSEQuantizer
	encoder   *util.TQMSEEncoder
	scratch   util.TQMSECode
	ids       []string
	codes     [][]byte
}

// NewTQVectorAccumulator creates an incremental TQ-MSE vector accumulator.
func NewTQVectorAccumulator(opts TQVectorBuildOptions) (*TQVectorAccumulator, error) {
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
		return nil, err
	}
	return &TQVectorAccumulator{
		dims:      dims,
		bits:      bits,
		quantizer: q,
		encoder:   q.NewEncoder(),
		scratch:   util.TQMSECode{Codes: make([]byte, q.CodeSize())},
	}, nil
}

// Add normalizes and encodes one vector into the accumulator.
func (a *TQVectorAccumulator) Add(id string, embedding []float32) error {
	if id == "" {
		return nil
	}
	if len(embedding) != a.dims {
		return fmt.Errorf("vector %s has %d dims, expected %d", id, len(embedding), a.dims)
	}
	norm := util.NormalizeVectorCopy(embedding)
	code, err := a.encoder.EncodeInto(a.scratch, norm)
	if err != nil {
		return fmt.Errorf("encode vector %s: %w", id, err)
	}
	a.scratch = code
	a.ids = append(a.ids, id)
	a.codes = append(a.codes, append([]byte(nil), code.Codes...))
	return nil
}

// Count returns the number of encoded vectors.
func (a *TQVectorAccumulator) Count() int {
	if a == nil {
		return 0
	}
	return len(a.ids)
}

// WriteChunkStore writes the accumulator as the default chunk-vector artifact.
func (a *TQVectorAccumulator) WriteChunkStore(ctx context.Context, dir string) (int, error) {
	if a == nil || a.Count() == 0 {
		return 0, RemoveTQVectorStore(dir)
	}
	return buildTQVectorStoreFromCodes(ctx, TQVectorPath(dir), a.ids, a.codes, a.quantizer, a.dims, a.bits)
}

// WriteFileStore writes the accumulator as the default file-vector artifact.
func (a *TQVectorAccumulator) WriteFileStore(ctx context.Context, dir string) (int, error) {
	if a == nil || a.Count() == 0 {
		return 0, RemoveTQFileVectorStore(dir)
	}
	return buildTQVectorStoreFromCodes(ctx, TQFileVectorPath(dir), a.ids, a.codes, a.quantizer, a.dims, a.bits)
}

// TQVectorPath returns the default compact dense vector artifact path.
func TQVectorPath(dir string) string {
	return filepath.Join(dir, tqVectorFileName)
}

// TQFileVectorPath returns the compact file-level vector artifact path.
func TQFileVectorPath(dir string) string {
	return filepath.Join(dir, tqFileVectorFileName)
}

// HasTQVectorStoreAtPath reports whether a compact dense vector artifact exists.
func HasTQVectorStoreAtPath(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Size() >= tqVectorHeaderSize
}

// HasTQVectorStore reports whether a compact dense vector artifact exists.
func HasTQVectorStore(dir string) bool {
	return HasTQVectorStoreAtPath(TQVectorPath(dir))
}

// HasTQFileVectorStore reports whether a compact file-level vector artifact exists.
func HasTQFileVectorStore(dir string) bool {
	return HasTQVectorStoreAtPath(TQFileVectorPath(dir))
}

// RemoveTQVectorStoreAtPath removes a compact dense vector artifact when present.
func RemoveTQVectorStoreAtPath(path string) error {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// RemoveTQVectorStore removes the compact dense vector artifact when present.
func RemoveTQVectorStore(dir string) error {
	return RemoveTQVectorStoreAtPath(TQVectorPath(dir))
}

// RemoveTQFileVectorStore removes the compact file-level vector artifact when present.
func RemoveTQFileVectorStore(dir string) error {
	return RemoveTQVectorStoreAtPath(TQFileVectorPath(dir))
}

// BuildTQVectorStoreAtPath writes a compact dense vector artifact to path.
func BuildTQVectorStoreAtPath(ctx context.Context, path string, ids []string, embeddings [][]float32, opts TQVectorBuildOptions) (int, error) {
	return buildTQVectorStore(ctx, path, ids, embeddings, opts)
}

// BuildTQVectorStore writes a compact dense vector artifact for chunk vectors.
func BuildTQVectorStore(ctx context.Context, dir string, chunkIDs []string, embeddings [][]float32, opts TQVectorBuildOptions) (int, error) {
	return BuildTQVectorStoreAtPath(ctx, TQVectorPath(dir), chunkIDs, embeddings, opts)
}

// BuildTQFileVectorStore writes a compact dense vector artifact for file-level vectors.
func BuildTQFileVectorStore(ctx context.Context, dir string, filePaths []string, embeddings [][]float32, opts TQVectorBuildOptions) (int, error) {
	return BuildTQVectorStoreAtPath(ctx, TQFileVectorPath(dir), filePaths, embeddings, opts)
}

func buildTQVectorStore(ctx context.Context, path string, ids []string, embeddings [][]float32, opts TQVectorBuildOptions) (int, error) {
	if len(ids) != len(embeddings) {
		return 0, fmt.Errorf("id/vector count mismatch: %d ids, %d vectors", len(ids), len(embeddings))
	}
	if len(ids) == 0 {
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

	encoder := q.NewEncoder()
	codeSize := q.CodeSize()
	code := util.TQMSECode{Codes: make([]byte, codeSize)}
	norm := make([]float32, dims)
	codes := make([][]byte, 0, len(ids))
	for i, id := range ids {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		vec := embeddings[i]
		if len(vec) != dims {
			return 0, fmt.Errorf("vector %s has %d dims, expected %d", id, len(vec), dims)
		}
		copy(norm, vec)
		norm = util.NormalizeVector(norm)
		encoded, err := encoder.EncodeInto(code, norm)
		if err != nil {
			return 0, fmt.Errorf("encode vector %s: %w", id, err)
		}
		codes = append(codes, append([]byte(nil), encoded.Codes...))
	}

	return buildTQVectorStoreFromCodes(ctx, path, ids, codes, q, dims, bits)
}

func buildTQVectorStoreFromCodes(ctx context.Context, path string, ids []string, codes [][]byte, q *util.TQMSEQuantizer, dims, bits int) (int, error) {
	if len(ids) != len(codes) {
		return 0, fmt.Errorf("id/code count mismatch: %d ids, %d codes", len(ids), len(codes))
	}
	if len(ids) == 0 {
		return 0, nil
	}
	metadata, err := q.Serialize()
	if err != nil {
		return 0, fmt.Errorf("serialize tq vector quantizer: %w", err)
	}

	type item struct {
		id   string
		code []byte
	}
	items := make([]item, 0, len(ids))
	codeSize := q.CodeSize()
	for i, id := range ids {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		code := codes[i]
		if len(code) != codeSize {
			return 0, fmt.Errorf("vector %s has code size %d, expected %d", id, len(code), codeSize)
		}
		items = append(items, item{id: id, code: code})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].id < items[j].id
	})

	indexSize := 0
	for _, item := range items {
		if len(item.id) > int(^uint16(0)) {
			return 0, fmt.Errorf("vector id too long: %s", item.id)
		}
		indexSize += 2 + len(item.id)
	}
	metadataOffset := tqVectorHeaderSize
	indexOffset := metadataOffset + len(metadata)
	codesOffset := indexOffset + indexSize
	totalSize := codesOffset + len(items)*codeSize

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return 0, err
	}
	f, err := os.CreateTemp(dir, filepath.Base(path)+".*.tmp")
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

	codeOffset := int64(codesOffset)
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			_ = os.Remove(tmpPath)
			return 0, err
		}
		if _, err := f.WriteAt(item.code, codeOffset); err != nil {
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
	if err := os.Rename(tmpPath, path); err != nil {
		return 0, err
	}
	return len(items), nil
}

// OpenTQVectorStore opens an existing compact dense vector artifact.
func OpenTQVectorStore(dir string) (*TQVectorStore, error) {
	return OpenTQVectorStoreAtPath(TQVectorPath(dir))
}

// OpenTQFileVectorStore opens an existing compact file-level vector artifact.
func OpenTQFileVectorStore(dir string) (*TQVectorStore, error) {
	return OpenTQVectorStoreAtPath(TQFileVectorPath(dir))
}

// OpenTQVectorStoreAtPath opens an existing compact dense vector artifact.
func OpenTQVectorStoreAtPath(path string) (*TQVectorStore, error) {
	s := &TQVectorStore{
		path:      path,
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
	maxInt := int64(int(^uint(0) >> 1))
	if stat.Size() > maxInt {
		_ = f.Close()
		return fmt.Errorf("tq vector file too large: %d", stat.Size())
	}

	header := make([]byte, tqVectorHeaderSize)
	if _, err := f.ReadAt(header, 0); err != nil {
		_ = f.Close()
		return fmt.Errorf("read tq vector header: %w", err)
	}
	if string(header[0:4]) != tqVectorMagic {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector magic: %q", string(header[0:4]))
	}
	version := binary.LittleEndian.Uint32(header[4:8])
	if version != tqVectorVersion {
		_ = f.Close()
		return fmt.Errorf("unsupported tq vector version: %d", version)
	}

	dims64 := int64(binary.LittleEndian.Uint32(header[8:12]))
	bits64 := int64(binary.LittleEndian.Uint32(header[12:16]))
	vectorCount64 := int64(binary.LittleEndian.Uint32(header[16:20]))
	codeSize64 := int64(binary.LittleEndian.Uint32(header[20:24]))
	metadataOffset64 := int64(binary.LittleEndian.Uint32(header[24:28]))
	metadataLen64 := int64(binary.LittleEndian.Uint32(header[28:32]))
	indexOffset64 := int64(binary.LittleEndian.Uint32(header[32:36]))
	indexSize64 := int64(binary.LittleEndian.Uint32(header[36:40]))
	codesOffset64 := int64(binary.LittleEndian.Uint32(header[40:44]))
	fileSize64 := stat.Size()

	if dims64 > maxInt || bits64 > maxInt || vectorCount64 > maxInt || codeSize64 > maxInt {
		_ = f.Close()
		return fmt.Errorf("tq vector header value too large")
	}
	if !tqRangeWithin(metadataOffset64, metadataLen64, tqVectorHeaderSize, fileSize64) {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector metadata bounds")
	}
	metadataEnd64 := metadataOffset64 + metadataLen64
	if !tqRangeWithin(indexOffset64, indexSize64, metadataEnd64, fileSize64) {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector index bounds")
	}
	if vectorCount64 > 0 && indexSize64/vectorCount64 < 2 {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector index bounds")
	}
	indexEnd64 := indexOffset64 + indexSize64
	if !tqRangeWithin(codesOffset64, 0, indexEnd64, fileSize64) {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector code bounds")
	}
	remainingCodeBytes := fileSize64 - codesOffset64
	if vectorCount64 > 0 && codeSize64 > remainingCodeBytes/vectorCount64 {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector code bounds")
	}
	codesLen64 := vectorCount64 * codeSize64
	if !tqRangeWithin(codesOffset64, codesLen64, indexEnd64, fileSize64) {
		_ = f.Close()
		return fmt.Errorf("invalid tq vector code bounds")
	}

	dims := int(dims64)
	bits := int(bits64)
	vectorCount := int(vectorCount64)
	codeSize := int(codeSize64)
	metadataOffset := int(metadataOffset64)
	metadataLen := int(metadataLen64)
	indexOffset := int(indexOffset64)
	indexSize := int(indexSize64)
	codesOffset := int(codesOffset64)
	fileSize := int(fileSize64)
	data, err := syscall.Mmap(int(f.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("mmap tq vectors: %w", err)
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

	ids := make([]string, vectorCount)
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
		ids[i] = id
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
	s.ids = ids
	s.idToIndex = idToIndex
	return nil
}

func tqRangeWithin(offset, length, minOffset, fileSize int64) bool {
	return offset >= minOffset && length >= 0 && offset <= fileSize && length <= fileSize-offset
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

// Path returns the compact artifact path.
func (s *TQVectorStore) Path() string { return s.path }

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
	ids := s.ids
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
		item := DenseSearchResult{ID: ids[i], Distance: distance}
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

// ScoreByID returns compressed dense-vector distances for the requested IDs.
func (s *TQVectorStore) ScoreByID(ctx context.Context, embedding []float32, ids []string) (map[string]float64, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	s.mu.RLock()
	data := s.data
	q := s.quantizer
	idToIndex := s.idToIndex
	codeSize := s.codeSize
	codesOffset := s.codesOffset
	dims := s.dims
	s.mu.RUnlock()

	if len(data) == 0 || q == nil || len(idToIndex) == 0 {
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

	distances := make(map[string]float64, len(ids))
	code := util.TQMSECode{}
	for i, id := range ids {
		if i&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		idx, ok := idToIndex[id]
		if !ok {
			continue
		}
		start := codesOffset + idx*codeSize
		code.Codes = data[start : start+codeSize]
		distances[id] = 1.0 - q.Dot(prepared, code)
	}
	return distances, nil
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
