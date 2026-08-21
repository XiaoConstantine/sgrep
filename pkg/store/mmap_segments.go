package store

import (
	"bufio"
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
//	v1 int8:
//	  Header (32 bytes):
//	    - Magic: 4 bytes ("SGCS")
//	    - Version: 4 bytes (1)
//	    - Dims: 4 bytes
//	    - ChunkCount: 4 bytes
//	    - SegmentCount: 4 bytes
//	    - DataOffset: 4 bytes
//	    - Reserved: 8 bytes
//
//	v2 compact codecs:
//	  Header (40 bytes):
//	    - Magic: 4 bytes ("SGCS")
//	    - Version: 4 bytes (2)
//	    - Dims: 4 bytes
//	    - ChunkCount: 4 bytes
//	    - SegmentCount: 4 bytes
//	    - HeaderSize: 4 bytes
//	    - DataOffset: 4 bytes
//	    - Codec: 4 bytes
//	    - MetadataSize: 4 bytes
//	Index (20 bytes per chunk):
//	  - ChunkID hash: 8 bytes
//	  - DataOffset: 4 bytes
//	  - SegmentCount: 2 bytes
//	  - ChunkIDLen: 2 bytes
//	  - ChunkID: variable (padded to 4-byte alignment)
//	Data (variable):
//	  - v1 per segment: dims bytes (int8) + 4 bytes (scale:f32) + 4 bytes (min:f32)
//	  - v2 per segment: codec-specific packed bytes
type MMapSegmentStore struct {
	path  string
	dims  int
	data  []byte // memory-mapped file
	file  *os.File
	mu    sync.RWMutex
	codec ColBERTCodec
	pq    *util.ProductQuantizer
	tq    *util.TQMSEQuantizer

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
	chunks    map[string][]ColBERTSegment
	totalSegs int
}

const (
	mmapMagic        = "SGCS" // Sgrep ColBERT Segments
	mmapVersionV1    = 1
	mmapVersionV2    = 2
	mmapHeaderSizeV1 = 32
	mmapHeaderSizeV2 = 40
	mmapCodecInt8    = 1
	mmapCodecPQ6     = 2
	mmapCodecTQMSE   = 3
)

// OpenMMapSegmentStore opens or creates a memory-mapped segment store.
func OpenMMapSegmentStore(dir string, dims int) (*MMapSegmentStore, error) {
	path := filepath.Join(dir, "colbert_segments.mmap")

	store := &MMapSegmentStore{
		path:       path,
		dims:       dims,
		codec:      ColBERTCodecInt8,
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

	if stat.Size() < mmapHeaderSizeV1 {
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
	headerSize := 0
	indexOffset := 0
	dataOffset := 0
	switch version {
	case mmapVersionV1:
		indexOffset = mmapHeaderSizeV1
		s.codec = ColBERTCodecInt8
		s.pq = nil
		s.tq = nil
		s.dims = int(binary.LittleEndian.Uint32(data[8:12]))
		dataOffset = int(binary.LittleEndian.Uint32(data[20:24]))
	case mmapVersionV2:
		if len(data) < mmapHeaderSizeV2 {
			return fmt.Errorf("file too small for v2 header: %d", len(data))
		}
		s.dims = int(binary.LittleEndian.Uint32(data[8:12]))
		headerSize = int(binary.LittleEndian.Uint32(data[20:24]))
		dataOffset = int(binary.LittleEndian.Uint32(data[24:28]))
		codecID := binary.LittleEndian.Uint32(data[28:32])
		codebookSize := int(binary.LittleEndian.Uint32(data[32:36]))
		if len(data) < headerSize+codebookSize {
			return fmt.Errorf("v2 header out of bounds")
		}
		switch codecID {
		case mmapCodecPQ6:
			s.codec = ColBERTCodecPQ6
		case mmapCodecTQMSE:
			s.codec = ColBERTCodecTQMSE
		default:
			s.codec = ColBERTCodecInt8
		}
		if s.codec == ColBERTCodecPQ6 && codebookSize > 0 {
			pq, err := util.DeserializeCodebook(data[headerSize : headerSize+codebookSize])
			if err != nil {
				return fmt.Errorf("deserialize mmap codebook: %w", err)
			}
			s.pq = pq
			s.tq = nil
		} else if s.codec == ColBERTCodecTQMSE && codebookSize > 0 {
			tq, err := util.DeserializeTQMSEQuantizer(data[headerSize : headerSize+codebookSize])
			if err != nil {
				return fmt.Errorf("deserialize mmap tqmse metadata: %w", err)
			}
			s.tq = tq
			s.pq = nil
		} else {
			s.pq = nil
			s.tq = nil
		}
		indexOffset = headerSize + codebookSize
	default:
		return fmt.Errorf("unsupported version: %d", version)
	}
	chunkCount := int(binary.LittleEndian.Uint32(data[12:16]))

	// Build chunk index
	offset := indexOffset
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

// SetColBERTCodec configures the codec used by the next write transaction.
func (s *MMapSegmentStore) SetColBERTCodec(codec ColBERTCodec, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.codec = ResolveColBERTCodec(codec, ColBERTCodecUnspecified)
	s.pq = pq
	s.tq = tq
}

// ColBERTCodec reports the codec stored in the mmap artifact.
func (s *MMapSegmentStore) ColBERTCodec() ColBERTCodec {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.codec
}

// ProductQuantizer returns the mmap-embedded PQ codebook when present.
func (s *MMapSegmentStore) ProductQuantizer() *util.ProductQuantizer {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.pq
}

// TQMSEQuantizer returns the mmap-embedded TQ-MSE quantizer when present.
func (s *MMapSegmentStore) TQMSEQuantizer() *util.TQMSEQuantizer {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tq
}

func (s *MMapSegmentStore) segmentSize() (int, error) {
	switch s.codec {
	case ColBERTCodecPQ6:
		if s.pq == nil {
			return 0, fmt.Errorf("pq mmap missing codebook")
		}
		return s.pq.CodeSize(), nil
	case ColBERTCodecTQMSE:
		if s.tq == nil {
			return 0, fmt.Errorf("tqmse mmap missing quantizer")
		}
		return s.tq.CodeSize(), nil
	default:
		return s.dims + 8, nil
	}
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
	return s.getColBERTSegmentsLocked(ctx, chunkID)
}

func (s *MMapSegmentStore) getColBERTSegmentsLocked(ctx context.Context, chunkID string) ([]ColBERTSegment, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	loc, ok := s.chunkIndex[chunkID]
	if !ok {
		return nil, nil // Not found
	}

	if s.data == nil {
		return nil, fmt.Errorf("mmap not loaded")
	}

	segSize, err := s.segmentSize()
	if err != nil {
		return nil, err
	}
	segments := make([]ColBERTSegment, loc.count)

	for i := 0; i < loc.count; i++ {
		segOffset := loc.offset + i*segSize
		if segOffset+segSize > len(s.data) {
			return nil, fmt.Errorf("segment out of bounds")
		}
		switch s.codec {
		case ColBERTCodecPQ6:
			codes := append([]byte(nil), s.data[segOffset:segOffset+segSize]...)
			segments[i] = ColBERTSegment{
				SegmentIdx: i,
				PQCodes:    codes,
			}
		case ColBERTCodecTQMSE:
			codes := append([]byte(nil), s.data[segOffset:segOffset+segSize]...)
			segments[i] = ColBERTSegment{
				SegmentIdx: i,
				TQCodes:    codes,
			}
		default:
			embInt8 := make([]int8, s.dims)
			for j := 0; j < s.dims; j++ {
				embInt8[j] = int8(s.data[segOffset+j])
			}
			scale := float32frombytes(s.data[segOffset+s.dims : segOffset+s.dims+4])
			min := float32frombytes(s.data[segOffset+s.dims+4 : segOffset+s.dims+8])
			segments[i] = ColBERTSegment{
				SegmentIdx:    i,
				EmbeddingInt8: embInt8,
				QuantScale:    scale,
				QuantMin:      min,
			}
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
	return s.commitWriteLocked()
}

func (s *MMapSegmentStore) commitWriteLocked() error {
	if s.writeBuffer == nil {
		return fmt.Errorf("no write transaction")
	}

	// Sort chunk IDs for deterministic output
	chunkIDs := make([]string, 0, len(s.writeBuffer.chunks))
	for id := range s.writeBuffer.chunks {
		chunkIDs = append(chunkIDs, id)
	}
	sort.Strings(chunkIDs)

	codec := s.codec
	if codec == ColBERTCodecUnspecified && len(chunkIDs) > 0 {
		first := s.writeBuffer.chunks[chunkIDs[0]]
		if len(first) > 0 && len(first[0].PQCodes) > 0 {
			codec = ColBERTCodecPQ6
		} else if len(first) > 0 && len(first[0].TQCodes) > 0 {
			codec = ColBERTCodecTQMSE
		} else {
			codec = ColBERTCodecInt8
		}
	}
	if codec == ColBERTCodecUnspecified {
		codec = ColBERTCodecInt8
	}
	if codec == ColBERTCodecPQ6 && s.pq == nil {
		return fmt.Errorf("pq mmap write requires a codebook")
	}
	if codec == ColBERTCodecTQMSE && s.tq == nil {
		return fmt.Errorf("tqmse mmap write requires a quantizer")
	}

	// Calculate sizes
	segSize := s.dims + 8
	switch codec {
	case ColBERTCodecPQ6:
		segSize = s.pq.CodeSize()
	case ColBERTCodecTQMSE:
		segSize = s.tq.CodeSize()
	}
	for chunkID, segments := range s.writeBuffer.chunks {
		for i, seg := range segments {
			switch codec {
			case ColBERTCodecPQ6:
				if len(seg.PQCodes) != segSize {
					return fmt.Errorf("pq mmap segment %s[%d] has code length %d (expected %d)", chunkID, i, len(seg.PQCodes), segSize)
				}
			case ColBERTCodecTQMSE:
				if len(seg.TQCodes) != segSize {
					return fmt.Errorf("tqmse mmap segment %s[%d] has code length %d (expected %d)", chunkID, i, len(seg.TQCodes), segSize)
				}
			}
		}
	}
	indexSize := 0
	for _, id := range chunkIDs {
		entrySize := 16 + len(id)
		entrySize = (entrySize + 3) & ^3
		indexSize += entrySize
	}
	headerSize := mmapHeaderSizeV1
	version := uint32(mmapVersionV1)
	var codecMetadata []byte
	codecID := uint32(mmapCodecInt8)
	switch codec {
	case ColBERTCodecPQ6:
		headerSize = mmapHeaderSizeV2
		version = mmapVersionV2
		codecID = mmapCodecPQ6
		var err error
		codecMetadata, err = s.pq.SerializeCodebook()
		if err != nil {
			return fmt.Errorf("serialize mmap codebook: %w", err)
		}
	case ColBERTCodecTQMSE:
		headerSize = mmapHeaderSizeV2
		version = mmapVersionV2
		codecID = mmapCodecTQMSE
		var err error
		codecMetadata, err = s.tq.Serialize()
		if err != nil {
			return fmt.Errorf("serialize mmap tqmse metadata: %w", err)
		}
	}
	dataSize := s.writeBuffer.totalSegs * segSize
	totalSize := headerSize + len(codecMetadata) + indexSize + dataSize

	// Close existing mmap only after all pre-write validation has passed.
	if s.data != nil {
		_ = syscall.Munmap(s.data)
		s.data = nil
	}
	if s.file != nil {
		_ = s.file.Close()
		s.file = nil
	}

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
	writer := bufio.NewWriterSize(f, 1<<20)

	// Write header
	header := make([]byte, headerSize)
	copy(header[0:4], mmapMagic)
	binary.LittleEndian.PutUint32(header[4:8], version)
	binary.LittleEndian.PutUint32(header[8:12], uint32(s.dims))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(chunkIDs)))
	binary.LittleEndian.PutUint32(header[16:20], uint32(s.writeBuffer.totalSegs))
	if version == mmapVersionV1 {
		binary.LittleEndian.PutUint32(header[20:24], uint32(headerSize+indexSize))
	} else {
		binary.LittleEndian.PutUint32(header[20:24], uint32(headerSize))
		binary.LittleEndian.PutUint32(header[24:28], uint32(headerSize+len(codecMetadata)+indexSize))
		binary.LittleEndian.PutUint32(header[28:32], codecID)
		binary.LittleEndian.PutUint32(header[32:36], uint32(len(codecMetadata)))
	}
	if _, err := writer.Write(header); err != nil {
		_ = f.Close()
		return err
	}
	if len(codecMetadata) > 0 {
		if _, err := writer.Write(codecMetadata); err != nil {
			_ = f.Close()
			return err
		}
	}

	// Write index and data
	dataOffset := 0
	s.chunkIndex = make(map[string]chunkLoc)

	for _, chunkID := range chunkIDs {
		segments := s.writeBuffer.chunks[chunkID]

		// Write index entry
		entrySize := (16 + len(chunkID) + 3) & ^3
		entry := make([]byte, entrySize)
		h := fnv.New64a()
		_, _ = h.Write([]byte(chunkID))
		binary.LittleEndian.PutUint64(entry[0:8], h.Sum64())
		binary.LittleEndian.PutUint32(entry[8:12], uint32(dataOffset))
		binary.LittleEndian.PutUint16(entry[12:14], uint16(len(segments)))
		binary.LittleEndian.PutUint16(entry[14:16], uint16(len(chunkID)))
		copy(entry[16:], chunkID)

		if _, err := writer.Write(entry); err != nil {
			_ = f.Close()
			return err
		}

		s.chunkIndex[chunkID] = chunkLoc{
			offset: headerSize + len(codecMetadata) + indexSize + dataOffset,
			count:  len(segments),
		}
		dataOffset += len(segments) * segSize
	}

	// Write segment data
	var int8Data []byte
	for _, chunkID := range chunkIDs {
		segments := s.writeBuffer.chunks[chunkID]
		for _, seg := range segments {
			var data []byte
			switch codec {
			case ColBERTCodecPQ6:
				data = seg.PQCodes
			case ColBERTCodecTQMSE:
				data = seg.TQCodes
			default:
				if cap(int8Data) < segSize {
					int8Data = make([]byte, segSize)
				} else {
					int8Data = int8Data[:segSize]
				}
				clear(int8Data)
				for j, v := range seg.EmbeddingInt8 {
					int8Data[j] = byte(v)
				}
				binary.LittleEndian.PutUint32(int8Data[s.dims:s.dims+4], *(*uint32)(unsafe.Pointer(&seg.QuantScale)))
				binary.LittleEndian.PutUint32(int8Data[s.dims+4:s.dims+8], *(*uint32)(unsafe.Pointer(&seg.QuantMin)))
				data = int8Data
			}
			if _, err := writer.Write(data); err != nil {
				_ = f.Close()
				return err
			}
		}
	}

	if err := writer.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	s.writeBuffer = nil
	s.codec = codec

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
		segs, err := s.getColBERTSegmentsLocked(ctx, id)
		if err != nil {
			return err
		}
		allSegments[id] = segs
	}

	// Rebuild while retaining the write lock. Calling the public transaction
	// methods here would recursively acquire s.mu and deadlock.
	s.writeBuffer = &mmapWriteBuffer{chunks: make(map[string][]ColBERTSegment)}
	for id, segs := range allSegments {
		s.writeBuffer.chunks[id] = segs
		s.writeBuffer.totalSegs += len(segs)
	}
	return s.commitWriteLocked()
}

// GetChunksForColBERT is not supported by MMapSegmentStore.
// MMapSegmentStore only stores pre-computed segment embeddings, not document content.
// Use LibSQLStore for ColBERT preindexing operations.
func (s *MMapSegmentStore) GetChunksForColBERT(ctx context.Context, batchSize int, offset int) ([]ChunkInfo, error) {
	return nil, fmt.Errorf("GetChunksForColBERT not supported by MMapSegmentStore: use LibSQLStore for preindexing")
}

// Ensure MMapSegmentStore implements ColBERTSegmentStorer
var _ ColBERTSegmentStorer = (*MMapSegmentStore)(nil)

// Helper functions
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
			embeddings[i] = append([]float32(nil), seg.Embedding...)
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

			result = append(result, ColBERTSegment{
				SegmentIdx: len(result),
				Text:       texts[0], // Keep first text as representative
				Embedding:  avgEmb,
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
