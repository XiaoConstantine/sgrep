package store

import (
	"container/heap"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"
	"unsafe"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// MMapVectorStore provides memory-mapped storage for chunk vector embeddings.
// This enables zero-copy access to vectors for fast search without loading into heap.
//
// File format:
//
//	Header (32 bytes):
//	  - Magic: 4 bytes ("SGVE")
//	  - Version: 4 bytes (1)
//	  - Dims: 4 bytes (768)
//	  - VectorCount: 4 bytes
//	  - DataOffset: 4 bytes (where vector data starts)
//	  - Reserved: 12 bytes
//	Index (variable):
//	  Per vector: 8 bytes hash + variable chunkID
//	Data (contiguous):
//	  Per vector: dims * 4 bytes (float32)
type MMapVectorStore struct {
	path string
	dims int
	data []byte // memory-mapped file
	file *os.File
	mu   sync.RWMutex

	// Index for fast chunk lookup: chunkID -> vector offset
	vectorIndex map[string]int // maps chunkID to index in vectors array
	vectorIDs   []string       // vector index -> chunk ID
	vectorCount int

	// Precomputed offsets for direct access
	dataOffset int // byte offset where vector data starts

	// Write buffer for building new files
	writeBuffer *mmapVectorWriteBuffer
}

type mmapVectorWriteBuffer struct {
	chunkIDs []string
	vectors  [][]float32
}

// MMapVectorPath returns the exact float32 vector artifact path.
func MMapVectorPath(dir string) string { return filepath.Join(dir, "vectors.mmap") }

// HasMMapVectorStore reports whether an exact vector artifact exists.
func HasMMapVectorStore(dir string) bool {
	info, err := os.Stat(MMapVectorPath(dir))
	return err == nil && info.Size() >= vectorHeaderSize
}

const (
	vectorMagic      = "SGVE" // Sgrep Vector Embeddings
	vectorVersion    = 1
	vectorHeaderSize = 32
)

// OpenMMapVectorStore opens or creates a memory-mapped vector store.
func OpenMMapVectorStore(dir string, dims int) (*MMapVectorStore, error) {
	path := MMapVectorPath(dir)

	store := &MMapVectorStore{
		path:        path,
		dims:        dims,
		vectorIndex: make(map[string]int),
	}

	// Check if file exists
	if _, err := os.Stat(path); err == nil {
		if err := store.load(); err != nil {
			return nil, fmt.Errorf("load mmap vectors: %w", err)
		}
	}

	return store, nil
}

// load memory-maps an existing file.
func (s *MMapVectorStore) load() error {
	f, err := os.OpenFile(s.path, os.O_RDONLY, 0644)
	if err != nil {
		return err
	}
	loaded := false
	defer func() {
		if !loaded {
			_ = f.Close()
		}
	}()

	stat, err := f.Stat()
	if err != nil {
		return err
	}

	if stat.Size() < vectorHeaderSize {
		return fmt.Errorf("file too small: %d", stat.Size())
	}

	// Memory map the file (read-only for safety)
	data, err := syscall.Mmap(int(f.Fd()), 0, int(stat.Size()),
		syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("mmap: %w", err)
	}
	mapped := false
	defer func() {
		if !mapped {
			_ = syscall.Munmap(data)
		}
	}()

	// Verify header
	if string(data[0:4]) != vectorMagic {
		return fmt.Errorf("invalid magic: %s", string(data[0:4]))
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != vectorVersion {
		return fmt.Errorf("unsupported version: %d", version)
	}
	dims := int(binary.LittleEndian.Uint32(data[8:12]))
	vectorCount := int(binary.LittleEndian.Uint32(data[12:16]))
	dataOffset := int(binary.LittleEndian.Uint32(data[16:20]))
	if dims <= 0 || vectorCount < 0 || dataOffset < vectorHeaderSize || dataOffset+vectorCount*dims*4 > len(data) {
		return fmt.Errorf("invalid vector mmap dimensions or offsets")
	}

	// Build local index structures and publish only after complete validation.
	offset := vectorHeaderSize
	vectorIDs := make([]string, vectorCount)
	vectorIndex := make(map[string]int, vectorCount)
	for i := 0; i < vectorCount; i++ {
		if offset+2 > dataOffset {
			return fmt.Errorf("truncated vector index at entry %d", i)
		}
		idLen := int(binary.LittleEndian.Uint16(data[offset : offset+2]))
		if offset+2+idLen > dataOffset {
			return fmt.Errorf("truncated vector ID at entry %d", i)
		}
		chunkID := string(data[offset+2 : offset+2+idLen])
		vectorIndex[chunkID] = i
		vectorIDs[i] = chunkID
		offset += 2 + idLen
	}

	s.file = f
	s.data = data
	s.dims = dims
	s.vectorCount = vectorCount
	s.dataOffset = dataOffset
	s.vectorIndex = vectorIndex
	s.vectorIDs = vectorIDs
	loaded = true
	mapped = true
	return nil
}

// Close unmaps and closes the file.
func (s *MMapVectorStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data != nil {
		if err := syscall.Munmap(s.data); err != nil {
			return err
		}
		s.data = nil
	}
	if s.file != nil {
		if err := s.file.Close(); err != nil {
			return err
		}
		s.file = nil
	}
	return nil
}

// GetVector retrieves a vector for a chunk from mmap (zero-copy).
func (s *MMapVectorStore) GetVector(chunkID string) []float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	idx, ok := s.vectorIndex[chunkID]
	if !ok || s.data == nil {
		return nil
	}

	// Calculate offset into data section
	vecOffset := s.dataOffset + idx*s.dims*4
	if vecOffset+s.dims*4 > len(s.data) {
		return nil
	}

	// Zero-copy cast: read float32 values directly from mmap
	vec := make([]float32, s.dims)
	for i := 0; i < s.dims; i++ {
		bits := binary.LittleEndian.Uint32(s.data[vecOffset+i*4 : vecOffset+i*4+4])
		vec[i] = *(*float32)(unsafe.Pointer(&bits))
	}
	return vec
}

// GetVectorBatch retrieves multiple vectors efficiently.
func (s *MMapVectorStore) GetVectorBatch(chunkIDs []string) map[string][]float32 {
	result := make(map[string][]float32)
	for _, id := range chunkIDs {
		if vec := s.GetVector(id); vec != nil {
			result[id] = vec
		}
	}
	return result
}

// GetAllVectors returns all vectors and their chunk IDs.
// This is used for in-memory search when the store is small enough.
func (s *MMapVectorStore) GetAllVectors() ([]string, [][]float32) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.data == nil || s.vectorCount == 0 {
		return nil, nil
	}

	// Build sorted list of chunkIDs to match index order
	chunkIDs := make([]string, s.vectorCount)
	for id, idx := range s.vectorIndex {
		if idx < len(chunkIDs) {
			chunkIDs[idx] = id
		}
	}

	vectors := make([][]float32, s.vectorCount)
	for i := 0; i < s.vectorCount; i++ {
		vecOffset := s.dataOffset + i*s.dims*4
		if vecOffset+s.dims*4 > len(s.data) {
			break
		}

		vec := make([]float32, s.dims)
		for j := 0; j < s.dims; j++ {
			bits := binary.LittleEndian.Uint32(s.data[vecOffset+j*4 : vecOffset+j*4+4])
			vec[j] = *(*float32)(unsafe.Pointer(&bits))
		}
		vectors[i] = vec
	}

	return chunkIDs, vectors
}

// VectorCount returns the number of vectors in the store.
func (s *MMapVectorStore) VectorCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.vectorCount
}

// HasVectors returns true if vectors exist.
func (s *MMapVectorStore) HasVectors() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.vectorCount > 0
}

// Path returns the mmap artifact path.
func (s *MMapVectorStore) Path() string { return s.path }

// Search performs an exact cosine scan over normalized resident vectors.
func (s *MMapVectorStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]DenseSearchResult, error) {
	if limit <= 0 {
		return nil, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(embedding) != s.dims {
		return nil, fmt.Errorf("query has %d dims, expected %d", len(embedding), s.dims)
	}
	query := util.NormalizeVectorCopy(embedding)
	h := make(tqResultMaxHeap, 0, limit)
	for i := 0; i < s.vectorCount; i++ {
		if i&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		vector := s.vectorAtLocked(i)
		if len(vector) != s.dims {
			continue
		}
		distance := util.DotProductDistance(query, vector)
		if distance > threshold {
			continue
		}
		item := DenseSearchResult{ID: s.idAtIndexLocked(i), Distance: distance}
		if h.Len() < limit {
			heap.Push(&h, item)
		} else if distance < h[0].Distance {
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

// ScoreByID computes exact distances for selected lexical candidates.
func (s *MMapVectorStore) ScoreByID(ctx context.Context, embedding []float32, ids []string) (map[string]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(embedding) != s.dims {
		return nil, fmt.Errorf("query has %d dims, expected %d", len(embedding), s.dims)
	}
	query := util.NormalizeVectorCopy(embedding)
	result := make(map[string]float64, len(ids))
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		idx, ok := s.vectorIndex[id]
		if !ok {
			continue
		}
		result[id] = util.DotProductDistance(query, s.vectorAtLocked(idx))
	}
	return result, nil
}

func (s *MMapVectorStore) idAtIndexLocked(index int) string {
	if index < 0 || index >= len(s.vectorIDs) {
		return ""
	}
	return s.vectorIDs[index]
}

func (s *MMapVectorStore) vectorAtLocked(index int) []float32 {
	offset := s.dataOffset + index*s.dims*4
	if offset < 0 || offset+s.dims*4 > len(s.data) {
		return nil
	}
	if offset%4 == 0 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&s.data[offset])), s.dims)
	}
	vector := make([]float32, s.dims)
	for i := range vector {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(s.data[offset+i*4 : offset+i*4+4]))
	}
	return vector
}

// BeginWrite starts a write transaction for building the mmap file.
func (s *MMapVectorStore) BeginWrite() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writeBuffer = &mmapVectorWriteBuffer{
		chunkIDs: make([]string, 0),
		vectors:  make([][]float32, 0),
	}
}

// WriteVector adds a vector to the write buffer.
func (s *MMapVectorStore) WriteVector(chunkID string, vector []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.writeBuffer == nil {
		return
	}
	s.writeBuffer.chunkIDs = append(s.writeBuffer.chunkIDs, chunkID)
	s.writeBuffer.vectors = append(s.writeBuffer.vectors, util.NormalizeVectorCopy(vector))
}

// CommitWrite finalizes and writes the mmap file.
func (s *MMapVectorStore) CommitWrite() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.writeBuffer == nil {
		return fmt.Errorf("no write transaction")
	}

	// Keep the existing mapping alive while the replacement is built. Renaming
	// a complete inode prevents readers from observing a truncated artifact.

	// Sort by chunk ID for deterministic output
	type idxVec struct {
		id  string
		vec []float32
	}
	sorted := make([]idxVec, len(s.writeBuffer.chunkIDs))
	for i := range s.writeBuffer.chunkIDs {
		sorted[i] = idxVec{s.writeBuffer.chunkIDs[i], s.writeBuffer.vectors[i]}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].id < sorted[j].id
	})

	// Calculate sizes
	indexSize := 0
	for _, iv := range sorted {
		if len(iv.id) > math.MaxUint16 {
			return fmt.Errorf("vector ID is too long: %d bytes", len(iv.id))
		}
		if len(iv.vec) != s.dims {
			return fmt.Errorf("vector %q has %d dimensions, want %d", iv.id, len(iv.vec), s.dims)
		}
		indexSize += 2 + len(iv.id) // 2 bytes length + chunkID
	}
	indexPadding := (4 - (vectorHeaderSize+indexSize)%4) % 4
	dataSize := len(sorted) * s.dims * 4
	totalSize := vectorHeaderSize + indexSize + indexPadding + dataSize

	// Build beside the destination so the final rename is atomic.
	f, err := os.CreateTemp(filepath.Dir(s.path), ".vectors-*.mmap.tmp")
	if err != nil {
		return err
	}
	tempPath := f.Name()
	committed := false
	defer func() {
		_ = f.Close()
		if !committed {
			_ = os.Remove(tempPath)
		}
	}()
	if err := f.Chmod(0o644); err != nil {
		return err
	}

	// Preallocate
	if err := f.Truncate(int64(totalSize)); err != nil {
		_ = f.Close()
		return err
	}

	// Write header
	header := make([]byte, vectorHeaderSize)
	copy(header[0:4], vectorMagic)
	binary.LittleEndian.PutUint32(header[4:8], vectorVersion)
	binary.LittleEndian.PutUint32(header[8:12], uint32(s.dims))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(sorted)))
	binary.LittleEndian.PutUint32(header[16:20], uint32(vectorHeaderSize+indexSize+indexPadding))
	if _, err := f.Write(header); err != nil {
		_ = f.Close()
		return err
	}

	// Write index (chunk IDs). Do not publish these structures until the
	// replacement file is fully installed and load validates it.
	for _, iv := range sorted {
		idBytes := []byte(iv.id)
		entry := make([]byte, 2+len(idBytes))
		binary.LittleEndian.PutUint16(entry[0:2], uint16(len(idBytes)))
		copy(entry[2:], idBytes)
		if _, err := f.Write(entry); err != nil {
			_ = f.Close()
			return err
		}
	}

	if indexPadding > 0 {
		if _, err := f.Write(make([]byte, indexPadding)); err != nil {
			_ = f.Close()
			return err
		}
	}

	// Write vector data (contiguous float32 arrays)
	for _, iv := range sorted {
		vecData := make([]byte, s.dims*4)
		for j, v := range iv.vec {
			bits := *(*uint32)(unsafe.Pointer(&v))
			binary.LittleEndian.PutUint32(vecData[j*4:j*4+4], bits)
		}
		if _, err := f.Write(vecData); err != nil {
			_ = f.Close()
			return err
		}
	}

	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, s.path); err != nil {
		return err
	}
	committed = true

	if s.data != nil {
		_ = syscall.Munmap(s.data)
		s.data = nil
	}
	if s.file != nil {
		_ = s.file.Close()
		s.file = nil
	}
	s.writeBuffer = nil

	// Reload and validate the atomically installed artifact.
	return s.load()
}
