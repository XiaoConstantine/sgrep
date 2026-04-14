package index

import (
	"bufio"
	"encoding/gob"
	"io"
	"os"

	"github.com/XiaoConstantine/sgrep/pkg/store"
)

type colbertPQScratchRecord struct {
	ChunkID  string
	Segments []store.ColBERTSegment
}

type colbertPQScratchWriter struct {
	path string
	file *os.File
	buf  *bufio.Writer
	enc  *gob.Encoder
}

func newColBERTPQScratchWriter(dir string) (*colbertPQScratchWriter, error) {
	file, err := os.CreateTemp(dir, "colbert-pq-scratch-*.gob")
	if err != nil {
		return nil, err
	}

	buf := bufio.NewWriterSize(file, 1<<20)
	return &colbertPQScratchWriter{
		path: file.Name(),
		file: file,
		buf:  buf,
		enc:  gob.NewEncoder(buf),
	}, nil
}

func (w *colbertPQScratchWriter) Path() string {
	if w == nil {
		return ""
	}
	return w.path
}

func (w *colbertPQScratchWriter) WriteChunk(chunkID string, segments []store.ColBERTSegment) error {
	record := colbertPQScratchRecord{
		ChunkID:  chunkID,
		Segments: cloneColBERTSegments(segments),
	}
	return w.enc.Encode(record)
}

func (w *colbertPQScratchWriter) Close() error {
	if w == nil {
		return nil
	}
	if err := w.buf.Flush(); err != nil {
		_ = w.file.Close()
		return err
	}
	return w.file.Close()
}

type colbertPQScratchReader struct {
	file *os.File
	dec  *gob.Decoder
}

func openColBERTPQScratchReader(path string) (*colbertPQScratchReader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &colbertPQScratchReader{
		file: file,
		dec:  gob.NewDecoder(bufio.NewReaderSize(file, 1<<20)),
	}, nil
}

func (r *colbertPQScratchReader) Next() (*colbertPQScratchRecord, error) {
	var record colbertPQScratchRecord
	if err := r.dec.Decode(&record); err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, err
	}
	return &record, nil
}

func (r *colbertPQScratchReader) Close() error {
	if r == nil {
		return nil
	}
	return r.file.Close()
}

func cloneColBERTSegments(segments []store.ColBERTSegment) []store.ColBERTSegment {
	cloned := make([]store.ColBERTSegment, len(segments))
	for i, seg := range segments {
		cloned[i] = store.ColBERTSegment{
			SegmentIdx:    seg.SegmentIdx,
			Text:          seg.Text,
			QuantScale:    seg.QuantScale,
			QuantMin:      seg.QuantMin,
			Embedding:     append([]float32(nil), seg.Embedding...),
			EmbeddingInt8: append([]int8(nil), seg.EmbeddingInt8...),
			PQCodes:       append([]byte(nil), seg.PQCodes...),
		}
	}
	return cloned
}
