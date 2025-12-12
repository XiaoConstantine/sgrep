package store

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"
	"unsafe"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// MMapSegmentStore provides memory-mapped storage for ColBERT segment embeddings.
// This enables zero-copy access to embeddings with OS-level caching.
//
// File format:
//
//	Header (32 bytes):
//	  - Magic: 4 bytes ("SGCS")
//	  - Version: 4 bytes (1)
//	  - Dims: 4 bytes (768)
//	  - ChunkCount: 4 bytes
//	  - SegmentCount: 4 bytes
//	  - DataOffset: 4 bytes
//	  - Reserved: 8 bytes
//	Index (20 bytes per chunk):
//	  - ChunkID hash: 8 bytes
//	  - DataOffset: 4 bytes
//	  - SegmentCount: 2 bytes
//	  - ChunkIDLen: 2 bytes
//	  - ChunkID: variable (padded to 4-byte alignment)
//	Data (variable):
//	  - Per segment: dims bytes (int8) + 4 bytes (scale:f32) + 4 bytes (min:f32)
type MMapSegmentStore struct {
	path string
	dims int
	data []byte // memory-mapped file
	file *os.File
	mu   sync.RWMutex

	// In-memory index for fast chunk lookup
	chunkIndex map[string]chunkLoc

	// Write buffer for building new files
	writeBuffer *mmapWriteBuffer
}

type chunkLoc struct {
	offset int // byte offset into data section
	count  int // number of segments
}

type mmapWriteBuffer struct {
	chunks   map[string][]ColBERTSegment
	totalSegs int
}

const (
	mmapMagic      = "SGCS" // Sgrep ColBERT Segments
	mmapVersion    = 1
	mmapHeaderSize = 32
)

// OpenMMapSegmentStore opens or creates a memory-mapped segment store.
func OpenMMapSegmentStore(dir string, dims int) (*MMapSegmentStore, error) {
	path := filepath.Join(dir, "colbert_segments.mmap")

	store := &MMapSegmentStore{
		path:       path,
		dims:       dims,
		chunkIndex: make(map[string]chunkLoc),
	}

	// Check if file exists
	if _, err := os.Stat(path); err == nil {
		if err := store.load(); err != nil {
			return nil, fmt.Errorf("load mmap: %w", err)
		}
	}

	return store, nil
}

// load memory-maps an existing file.
func (s *MMapSegmentStore) load() error {
	f, err := os.OpenFile(s.path, os.O_RDONLY, 0644)
	if err != nil {
		return err
	}
	s.file = f

	stat, err := f.Stat()
	if err != nil {
		return err
	}

	if stat.Size() < mmapHeaderSize {
		return fmt.Errorf("file too small: %d", stat.Size())
	}

	// Memory map the file (read-only for safety)
	data, err := syscall.Mmap(int(f.Fd()), 0, int(stat.Size()),
		syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("mmap: %w", err)
	}
	s.data = data

	// Verify header
	if string(data[0:4]) != mmapMagic {
		return fmt.Errorf("invalid magic: %s", string(data[0:4]))
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != mmapVersion {
		return fmt.Errorf("unsupported version: %d", version)
	}
	s.dims = int(binary.LittleEndian.Uint32(data[8:12]))
	chunkCount := int(binary.LittleEndian.Uint32(data[12:16]))
	dataOffset := int(binary.LittleEndian.Uint32(data[20:24]))

	// Build chunk index
	offset := mmapHeaderSize
	for i := 0; i < chunkCount; i++ {
		if offset+12 > len(data) {
			break
		}
		segDataOffset := int(binary.LittleEndian.Uint32(data[offset+8 : offset+12]))
		segCount := int(binary.LittleEndian.Uint16(data[offset+12 : offset+14]))
		chunkIDLen := int(binary.LittleEndian.Uint16(data[offset+14 : offset+16]))

		if offset+16+chunkIDLen > len(data) {
			break
		}
		chunkID := string(data[offset+16 : offset+16+chunkIDLen])

		s.chunkIndex[chunkID] = chunkLoc{
			offset: dataOffset + segDataOffset,
			count:  segCount,
		}

		// Move to next entry (aligned to 4 bytes)
		entrySize := 16 + chunkIDLen
		entrySize = (entrySize + 3) & ^3 // Align to 4 bytes
		offset += entrySize
	}

	return nil
}

// Close unmaps and closes the file.
func (s *MMapSegmentStore) Close() error {
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

// GetColBERTSegments retrieves segments for a chunk from mmap.
func (s *MMapSegmentStore) GetColBERTSegments(ctx context.Context, chunkID string) ([]ColBERTSegment, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	loc, ok := s.chunkIndex[chunkID]
	if !ok {
		return nil, nil // Not found
	}

	if s.data == nil {
		return nil, fmt.Errorf("mmap not loaded")
	}

	segSize := s.dims + 8 // int8 embedding + scale(4) + min(4)
	segments := make([]ColBERTSegment, loc.count)

	for i := 0; i < loc.count; i++ {
		segOffset := loc.offset + i*segSize
		if segOffset+segSize > len(s.data) {
			return nil, fmt.Errorf("segment out of bounds")
		}

		// Read int8 embedding
		embInt8 := make([]int8, s.dims)
		for j := 0; j < s.dims; j++ {
			embInt8[j] = int8(s.data[segOffset+j])
		}

		// Read scale and min
		scale := float32frombytes(s.data[segOffset+s.dims : segOffset+s.dims+4])
		min := float32frombytes(s.data[segOffset+s.dims+4 : segOffset+s.dims+8])

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			EmbeddingInt8: embInt8,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	return segments, nil
}

// GetColBERTSegmentsBatch retrieves segments for multiple chunks.
func (s *MMapSegmentStore) GetColBERTSegmentsBatch(ctx context.Context, chunkIDs []string) (map[string][]ColBERTSegment, error) {
	result := make(map[string][]ColBERTSegment)
	for _, id := range chunkIDs {
		segs, err := s.GetColBERTSegments(ctx, id)
		if err != nil {
			return nil, err
		}
		if segs != nil {
			result[id] = segs
		}
	}
	return result, nil
}

// BeginWrite starts a write transaction for building the mmap file.
func (s *MMapSegmentStore) BeginWrite() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writeBuffer = &mmapWriteBuffer{
		chunks: make(map[string][]ColBERTSegment),
	}
}

// WriteSegments adds segments to the write buffer.
func (s *MMapSegmentStore) WriteSegments(chunkID string, segments []ColBERTSegment) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.writeBuffer == nil {
		return
	}
	s.writeBuffer.chunks[chunkID] = segments
	s.writeBuffer.totalSegs += len(segments)
}

// CommitWrite finalizes and writes the mmap file.
func (s *MMapSegmentStore) CommitWrite() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.writeBuffer == nil {
		return fmt.Errorf("no write transaction")
	}

	// Close existing mmap
	if s.data != nil {
		_ = syscall.Munmap(s.data)
		s.data = nil
	}
	if s.file != nil {
		_ = s.file.Close()
		s.file = nil
	}

	// Sort chunk IDs for deterministic output
	chunkIDs := make([]string, 0, len(s.writeBuffer.chunks))
	for id := range s.writeBuffer.chunks {
		chunkIDs = append(chunkIDs, id)
	}
	sort.Strings(chunkIDs)

	// Calculate sizes
	segSize := s.dims + 8
	indexSize := 0
	for _, id := range chunkIDs {
		entrySize := 16 + len(id)
		entrySize = (entrySize + 3) & ^3
		indexSize += entrySize
	}
	dataSize := s.writeBuffer.totalSegs * segSize
	totalSize := mmapHeaderSize + indexSize + dataSize

	// Create file
	f, err := os.Create(s.path)
	if err != nil {
		return err
	}

	// Preallocate
	if err := f.Truncate(int64(totalSize)); err != nil {
		_ = f.Close()
		return err
	}

	// Write header
	header := make([]byte, mmapHeaderSize)
	copy(header[0:4], mmapMagic)
	binary.LittleEndian.PutUint32(header[4:8], mmapVersion)
	binary.LittleEndian.PutUint32(header[8:12], uint32(s.dims))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(chunkIDs)))
	binary.LittleEndian.PutUint32(header[16:20], uint32(s.writeBuffer.totalSegs))
	binary.LittleEndian.PutUint32(header[20:24], uint32(mmapHeaderSize+indexSize))
	if _, err := f.Write(header); err != nil {
		_ = f.Close()
		return err
	}

	// Write index and data
	dataOffset := 0
	s.chunkIndex = make(map[string]chunkLoc)

	for _, chunkID := range chunkIDs {
		segments := s.writeBuffer.chunks[chunkID]

		// Write index entry
		entry := make([]byte, 16+len(chunkID))
		h := fnv.New64a()
		h.Write([]byte(chunkID))
		binary.LittleEndian.PutUint64(entry[0:8], h.Sum64())
		binary.LittleEndian.PutUint32(entry[8:12], uint32(dataOffset))
		binary.LittleEndian.PutUint16(entry[12:14], uint16(len(segments)))
		binary.LittleEndian.PutUint16(entry[14:16], uint16(len(chunkID)))
		copy(entry[16:], chunkID)

		// Pad to 4-byte alignment
		if len(entry)%4 != 0 {
			entry = append(entry, make([]byte, 4-len(entry)%4)...)
		}
		if _, err := f.Write(entry); err != nil {
			_ = f.Close()
			return err
		}

		s.chunkIndex[chunkID] = chunkLoc{
			offset: mmapHeaderSize + indexSize + dataOffset,
			count:  len(segments),
		}
		dataOffset += len(segments) * segSize
	}

	// Write segment data
	for _, chunkID := range chunkIDs {
		segments := s.writeBuffer.chunks[chunkID]
		for _, seg := range segments {
			segData := make([]byte, segSize)

			// Write int8 embedding
			for j, v := range seg.EmbeddingInt8 {
				segData[j] = byte(v)
			}

			// Write scale and min
			copy(segData[s.dims:s.dims+4], float32tobytes(seg.QuantScale))
			copy(segData[s.dims+4:s.dims+8], float32tobytes(seg.QuantMin))

			if _, err := f.Write(segData); err != nil {
				_ = f.Close()
				return err
			}
		}
	}

	_ = f.Close()
	s.writeBuffer = nil

	// Reload mmap
	return s.load()
}

// HasColBERTSegments checks if any segments exist.
func (s *MMapSegmentStore) HasColBERTSegments(ctx context.Context) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunkIndex) > 0, nil
}

// StoreColBERTSegments stores pre-computed segment embeddings for a chunk.
// For MMap store, this buffers segments until CommitWrite is called.
func (s *MMapSegmentStore) StoreColBERTSegments(ctx context.Context, chunkID string, segments []ColBERTSegment) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.writeBuffer == nil {
		s.writeBuffer = &mmapWriteBuffer{
			chunks: make(map[string][]ColBERTSegment),
		}
	}
	s.writeBuffer.chunks[chunkID] = segments
	s.writeBuffer.totalSegs += len(segments)
	return nil
}

// StoreColBERTSegmentsBatch stores segments for multiple chunks efficiently.
func (s *MMapSegmentStore) StoreColBERTSegmentsBatch(ctx context.Context, chunkSegments map[string][]ColBERTSegment) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.writeBuffer == nil {
		s.writeBuffer = &mmapWriteBuffer{
			chunks: make(map[string][]ColBERTSegment),
		}
	}
	for chunkID, segments := range chunkSegments {
		s.writeBuffer.chunks[chunkID] = segments
		s.writeBuffer.totalSegs += len(segments)
	}
	return nil
}

// DeleteColBERTSegments removes segment embeddings for a chunk.
// Note: For MMap store, this requires rebuilding the file (expensive).
// Consider batching deletes or using SQLite for frequent updates.
func (s *MMapSegmentStore) DeleteColBERTSegments(ctx context.Context, chunkID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// If in write mode, just remove from buffer
	if s.writeBuffer != nil {
		if segs, exists := s.writeBuffer.chunks[chunkID]; exists {
			s.writeBuffer.totalSegs -= len(segs)
			delete(s.writeBuffer.chunks, chunkID)
		}
		return nil
	}

	// For existing data, we need to rebuild without this chunk
	// This is expensive but deletion should be rare
	if _, exists := s.chunkIndex[chunkID]; !exists {
		return nil // Already doesn't exist
	}

	// Collect all segments except the deleted chunk
	allSegments := make(map[string][]ColBERTSegment)
	for id := range s.chunkIndex {
		if id == chunkID {
			continue
		}
		segs, err := s.GetColBERTSegments(ctx, id)
		if err != nil {
			return err
		}
		allSegments[id] = segs
	}

	// Rebuild the file
	s.BeginWrite()
	for id, segs := range allSegments {
		s.WriteSegments(id, segs)
	}
	return s.CommitWrite()
}

// Ensure MMapSegmentStore implements ColBERTSegmentStorer
var _ ColBERTSegmentStorer = (*MMapSegmentStore)(nil)

// Helper functions
func float32tobytes(f float32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, *(*uint32)(unsafe.Pointer(&f)))
	return buf
}

func float32frombytes(b []byte) float32 {
	bits := binary.LittleEndian.Uint32(b)
	return *(*float32)(unsafe.Pointer(&bits))
}

// SegmentPooler provides token pooling / segment merging functionality.
// It clusters similar segment embeddings and keeps representative centroids.
type SegmentPooler struct {
	maxSegments int     // Maximum segments to keep per chunk
	minSim      float64 // Minimum similarity to merge (0.95 = very similar)
}

// NewSegmentPooler creates a pooler with given parameters.
func NewSegmentPooler(maxSegments int, minSim float64) *SegmentPooler {
	if maxSegments <= 0 {
		maxSegments = 5
	}
	if minSim <= 0 {
		minSim = 0.90
	}
	return &SegmentPooler{
		maxSegments: maxSegments,
		minSim:      minSim,
	}
}

// Pool reduces segments by merging similar ones and keeping diverse representatives.
// Uses greedy furthest-point sampling to maximize coverage.
func (p *SegmentPooler) Pool(segments []ColBERTSegment) []ColBERTSegment {
	if len(segments) <= p.maxSegments {
		return segments
	}

	// Convert to float32 for similarity computation
	embeddings := make([][]float32, len(segments))
	for i, seg := range segments {
		if seg.EmbeddingInt8 != nil {
			embeddings[i] = util.DequantizeInt8(seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin)
		} else if seg.Embedding != nil {
			embeddings[i] = seg.Embedding
		} else {
			// Skip segments without embeddings
			continue
		}
		// Normalize for cosine similarity
		embeddings[i] = util.NormalizeVector(embeddings[i])
	}

	// Greedy furthest-point sampling (diversity sampling)
	// Start with the first segment, then repeatedly add the most distant one
	selected := make([]int, 0, p.maxSegments)
	selected = append(selected, 0)

	// Track minimum distance to any selected point
	minDists := make([]float64, len(segments))
	for i := range minDists {
		minDists[i] = 2.0 // Max cosine distance
	}

	for len(selected) < p.maxSegments && len(selected) < len(segments) {
		// Update distances to nearest selected point
		lastSelected := selected[len(selected)-1]
		for i := range segments {
			if embeddings[i] == nil {
				minDists[i] = -1 // Mark invalid
				continue
			}
			dist := 1.0 - util.DotProductUnrolled8(embeddings[lastSelected], embeddings[i])
			if dist < minDists[i] {
				minDists[i] = dist
			}
		}

		// Find point furthest from all selected
		maxDist := float64(-1)
		maxIdx := -1
		for i, d := range minDists {
			if d > maxDist {
				// Check not already selected
				alreadySelected := false
				for _, s := range selected {
					if s == i {
						alreadySelected = true
						break
					}
				}
				if !alreadySelected {
					maxDist = d
					maxIdx = i
				}
			}
		}

		if maxIdx < 0 {
			break
		}
		selected = append(selected, maxIdx)
	}

	// Build result
	result := make([]ColBERTSegment, len(selected))
	for i, idx := range selected {
		result[i] = segments[idx]
		result[i].SegmentIdx = i // Renumber
	}

	return result
}

// MergeBySimlarity merges segments that are highly similar.
// Returns merged segments where similar ones are averaged.
func (p *SegmentPooler) MergeBySimilarity(segments []ColBERTSegment) []ColBERTSegment {
	if len(segments) <= 1 {
		return segments
	}

	// Convert to float32
	embeddings := make([][]float32, len(segments))
	for i, seg := range segments {
		if seg.EmbeddingInt8 != nil {
			embeddings[i] = util.DequantizeInt8(seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin)
		} else if seg.Embedding != nil {
			embeddings[i] = make([]float32, len(seg.Embedding))
			copy(embeddings[i], seg.Embedding)
		}
		if embeddings[i] != nil {
			embeddings[i] = util.NormalizeVector(embeddings[i])
		}
	}

	// Union-find for clustering
	parent := make([]int, len(segments))
	for i := range parent {
		parent[i] = i
	}

	var find func(i int) int
	find = func(i int) int {
		if parent[i] != i {
			parent[i] = find(parent[i])
		}
		return parent[i]
	}

	union := func(i, j int) {
		pi, pj := find(i), find(j)
		if pi != pj {
			parent[pi] = pj
		}
	}

	// Merge similar segments
	for i := 0; i < len(segments); i++ {
		if embeddings[i] == nil {
			continue
		}
		for j := i + 1; j < len(segments); j++ {
			if embeddings[j] == nil {
				continue
			}
			sim := util.DotProductUnrolled8(embeddings[i], embeddings[j])
			if sim >= p.minSim {
				union(i, j)
			}
		}
	}

	// Group by cluster
	clusters := make(map[int][]int)
	for i := range segments {
		root := find(i)
		clusters[root] = append(clusters[root], i)
	}

	// Create merged segments
	result := make([]ColBERTSegment, 0, len(clusters))
	dims := len(embeddings[0])

	for _, members := range clusters {
		if len(members) == 1 {
			result = append(result, segments[members[0]])
			continue
		}

		// Average embeddings
		avgEmb := make([]float32, dims)
		var texts []string
		validCount := 0

		for _, idx := range members {
			if embeddings[idx] == nil {
				continue
			}
			validCount++
			for d := 0; d < dims; d++ {
				avgEmb[d] += embeddings[idx][d]
			}
			texts = append(texts, segments[idx].Text)
		}

		if validCount > 0 {
			for d := 0; d < dims; d++ {
				avgEmb[d] /= float32(validCount)
			}
			avgEmb = util.NormalizeVector(avgEmb)

			// Quantize back to int8
			quantized, scale, min := util.QuantizeInt8(avgEmb)

			result = append(result, ColBERTSegment{
				SegmentIdx:    len(result),
				Text:          texts[0], // Keep first text as representative
				EmbeddingInt8: quantized,
				QuantScale:    scale,
				QuantMin:      min,
			})
		}
	}

	return result
}

// PoolAndMerge applies both similarity merging and diversity sampling.
func (p *SegmentPooler) PoolAndMerge(segments []ColBERTSegment) []ColBERTSegment {
	// First merge very similar segments
	merged := p.MergeBySimilarity(segments)
	// Then sample for diversity if still too many
	return p.Pool(merged)
}
