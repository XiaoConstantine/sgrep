package util

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// TQMSEConfig configures the reconstruction-oriented TurboQuant stage.
type TQMSEConfig struct {
	Dims              int
	Bits              int
	Seed              uint64
	RotationBlockSize int
}

// TQMSECode stores bit-packed per-coordinate scalar quantizer indices.
type TQMSECode struct {
	Codes []byte
}

// TQMSEQuery stores query-side lookup tables for repeated dot products.
type TQMSEQuery struct {
	table []float64
}

// Valid reports whether the query has prepared lookup tables.
func (q TQMSEQuery) Valid() bool {
	return len(q.table) > 0
}

// TQMSEEncoder reuses encoding scratch for repeated vector quantization.
//
// A TQMSEEncoder is not safe for concurrent use.
type TQMSEEncoder struct {
	q       *TQMSEQuantizer
	rotated []float32
}

// TQMSEQuantizer applies random sign flips plus block FWHT rotation, then a
// data-oblivious scalar codebook tuned for normalized vectors.
type TQMSEQuantizer struct {
	dims      int
	bits      int
	levels    int
	blockSize int
	seed      uint64

	coordSigns []float32
	centroids  []float32
	boundaries []float32
}

// TQProdConfig configures the inner-product-oriented two-stage quantizer.
type TQProdConfig struct {
	Dims              int
	Bits              int
	QJLDims           int
	Seed              uint64
	RotationBlockSize int
}

// TQProdCode stores a TQProd vector: (b-1)-bit MSE indices, QJL residual
// signs, and the residual norm.
type TQProdCode struct {
	MSECodes     []byte
	QJLSignBits  []byte
	ResidualNorm float32
}

// TQProdQuery stores prepared query data for fast repeated TQProd scoring.
type TQProdQuery struct {
	mse       TQMSEQuery
	qjlTables []float32
}

// TQProdEncoder reuses encoding scratch for repeated TQProd quantization.
//
// A TQProdEncoder is not safe for concurrent use.
type TQProdEncoder struct {
	q          *TQProdQuantizer
	rotated    []float32
	approx     []float32
	projection []float32
}

// TQProdQuantizer combines a TQMSE stage with a structured 1-bit QJL residual
// correction for inner-product estimation.
type TQProdQuantizer struct {
	dims          int
	bits          int
	qjlDims       int
	qjlSize       int
	qjlSignSize   int
	seed          uint64
	residualScale float64

	mse        *TQMSEQuantizer
	qjlSigns   []float32
	qjlIndices []int
}

const (
	tqmseMagic    = "SGTM"
	tqmseVersion  = 1
	tqprodMagic   = "SGTQ"
	tqprodVersion = 1
)

// NewTQMSEQuantizer creates a reconstruction-oriented TQ quantizer.
func NewTQMSEQuantizer(cfg TQMSEConfig) (*TQMSEQuantizer, error) {
	cfg = defaultTQMSEConfig(cfg)
	if err := validateTQMSEConfig(cfg); err != nil {
		return nil, err
	}

	levels := 1 << uint(cfg.Bits)
	centroids := normalLloydMaxCodebook(levels, cfg.Dims, cfg.Seed+17)
	q := &TQMSEQuantizer{
		dims:       cfg.Dims,
		bits:       cfg.Bits,
		levels:     levels,
		blockSize:  cfg.RotationBlockSize,
		seed:       cfg.Seed,
		coordSigns: makeSigns(cfg.Dims, cfg.Seed+101),
		centroids:  centroids,
		boundaries: centroidBoundaries(centroids),
	}
	return q, nil
}

func defaultTQMSEConfig(cfg TQMSEConfig) TQMSEConfig {
	if cfg.Dims <= 0 {
		cfg.Dims = 768
	}
	if cfg.Bits <= 0 {
		cfg.Bits = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}
	if cfg.RotationBlockSize <= 0 {
		cfg.RotationBlockSize = largestPowerOfTwoDivisor(cfg.Dims)
	}
	return cfg
}

func validateTQMSEConfig(cfg TQMSEConfig) error {
	if cfg.Dims <= 0 {
		return fmt.Errorf("dims must be positive, got %d", cfg.Dims)
	}
	if cfg.Bits < 1 || cfg.Bits > 8 {
		return fmt.Errorf("bits must be in [1, 8], got %d", cfg.Bits)
	}
	if cfg.RotationBlockSize <= 0 {
		return fmt.Errorf("rotation block size must be positive, got %d", cfg.RotationBlockSize)
	}
	if cfg.Dims%cfg.RotationBlockSize != 0 {
		return fmt.Errorf("dims (%d) must be divisible by rotation block size (%d)", cfg.Dims, cfg.RotationBlockSize)
	}
	if !isPowerOfTwo(cfg.RotationBlockSize) {
		return fmt.Errorf("rotation block size must be a power of two, got %d", cfg.RotationBlockSize)
	}
	return nil
}

// Dims returns the vector dimension.
func (q *TQMSEQuantizer) Dims() int { return q.dims }

// Bits returns the per-coordinate bit width.
func (q *TQMSEQuantizer) Bits() int { return q.bits }

// CodeSize returns the number of bytes in a packed code.
func (q *TQMSEQuantizer) CodeSize() int {
	return packedBitLen(q.dims, q.bits)
}

// CompressionRatio returns the storage ratio versus float32 vectors.
func (q *TQMSEQuantizer) CompressionRatio() float64 {
	return float64(q.dims*4) / float64(q.CodeSize())
}

// Encode quantizes a normalized vector.
func (q *TQMSEQuantizer) Encode(vec []float32) (TQMSECode, error) {
	return q.NewEncoder().Encode(vec)
}

// NewEncoder creates a reusable encoder for this quantizer.
func (q *TQMSEQuantizer) NewEncoder() *TQMSEEncoder {
	return &TQMSEEncoder{
		q:       q,
		rotated: make([]float32, q.dims),
	}
}

// Encode quantizes a normalized vector.
func (e *TQMSEEncoder) Encode(vec []float32) (TQMSECode, error) {
	if e == nil || e.q == nil {
		return TQMSECode{}, fmt.Errorf("nil TQMSE encoder")
	}
	return e.EncodeInto(TQMSECode{}, vec)
}

// EncodeInto quantizes a normalized vector into dst, reusing dst.Codes when it
// has sufficient capacity.
func (e *TQMSEEncoder) EncodeInto(dst TQMSECode, vec []float32) (TQMSECode, error) {
	if e == nil || e.q == nil {
		return TQMSECode{}, fmt.Errorf("nil TQMSE encoder")
	}
	q := e.q
	if len(vec) != q.dims {
		return TQMSECode{}, fmt.Errorf("vector has wrong dims: %d (expected %d)", len(vec), q.dims)
	}
	dst.Codes = resizeBytes(dst.Codes, q.CodeSize())
	if err := q.encodeInto(dst.Codes, e.rotated, vec); err != nil {
		return TQMSECode{}, err
	}
	return dst, nil
}

// Decode reconstructs the MSE approximation in the original vector space.
func (q *TQMSEQuantizer) Decode(code TQMSECode) ([]float32, error) {
	rotated := make([]float32, q.dims)
	out := make([]float32, q.dims)
	if err := q.decodeInto(out, rotated, code.Codes); err != nil {
		return nil, err
	}
	return out, nil
}

// PrepareQuery builds the query-side lookup table used by Dot.
func (q *TQMSEQuantizer) PrepareQuery(query []float32) (TQMSEQuery, error) {
	if len(query) != q.dims {
		return TQMSEQuery{}, fmt.Errorf("query has wrong dims: %d (expected %d)", len(query), q.dims)
	}
	rotated := make([]float32, q.dims)
	q.RotateInto(rotated, query)

	table := make([]float64, q.dims*q.levels)
	buildTQMSETable(table, rotated, q.centroids, q.levels)
	return TQMSEQuery{table: table}, nil
}

// Dot estimates dot(query, vector) from prepared query tables and an MSE code.
func (q *TQMSEQuantizer) Dot(prepared TQMSEQuery, code TQMSECode) float64 {
	return dotPackedCodes(prepared.table, code.Codes, q.dims, q.bits, q.levels)
}

// Serialize stores all TQ-MSE metadata needed to score persisted codes.
func (q *TQMSEQuantizer) Serialize() ([]byte, error) {
	if q == nil {
		return nil, fmt.Errorf("nil TQMSE quantizer")
	}
	const headerSize = 36
	size := headerSize + len(q.centroids)*4 + len(q.coordSigns)
	buf := make([]byte, size)
	copy(buf[0:4], tqmseMagic)
	binary.LittleEndian.PutUint32(buf[4:8], tqmseVersion)
	binary.LittleEndian.PutUint32(buf[8:12], uint32(q.dims))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(q.bits))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(q.blockSize))
	binary.LittleEndian.PutUint64(buf[20:28], q.seed)
	binary.LittleEndian.PutUint32(buf[28:32], uint32(len(q.centroids)))
	binary.LittleEndian.PutUint32(buf[32:36], uint32(len(q.coordSigns)))

	offset := headerSize
	for _, centroid := range q.centroids {
		binary.LittleEndian.PutUint32(buf[offset:offset+4], math.Float32bits(centroid))
		offset += 4
	}
	for _, sign := range q.coordSigns {
		if sign > 0 {
			buf[offset] = 1
		}
		offset++
	}
	return buf, nil
}

// DeserializeTQMSEQuantizer loads a quantizer serialized by Serialize.
func DeserializeTQMSEQuantizer(data []byte) (*TQMSEQuantizer, error) {
	const headerSize = 36
	if len(data) < headerSize {
		return nil, fmt.Errorf("data too short for TQMSE header")
	}
	if string(data[0:4]) != tqmseMagic {
		return nil, fmt.Errorf("invalid TQMSE magic: %q", string(data[0:4]))
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != tqmseVersion {
		return nil, fmt.Errorf("unsupported TQMSE version: %d", version)
	}
	cfg := TQMSEConfig{
		Dims:              int(binary.LittleEndian.Uint32(data[8:12])),
		Bits:              int(binary.LittleEndian.Uint32(data[12:16])),
		RotationBlockSize: int(binary.LittleEndian.Uint32(data[16:20])),
		Seed:              binary.LittleEndian.Uint64(data[20:28]),
	}
	centroidCount := int(binary.LittleEndian.Uint32(data[28:32]))
	signCount := int(binary.LittleEndian.Uint32(data[32:36]))
	if err := validateTQMSEConfig(cfg); err != nil {
		return nil, err
	}
	if centroidCount != 1<<uint(cfg.Bits) {
		return nil, fmt.Errorf("centroid count mismatch: %d", centroidCount)
	}
	if signCount != cfg.Dims {
		return nil, fmt.Errorf("sign count mismatch: %d (expected %d)", signCount, cfg.Dims)
	}
	expected := headerSize + centroidCount*4 + signCount
	if len(data) < expected {
		return nil, fmt.Errorf("data too short: %d (expected %d)", len(data), expected)
	}

	offset := headerSize
	centroids := make([]float32, centroidCount)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4
	}
	signs := make([]float32, signCount)
	for i := range signs {
		if data[offset] == 1 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
		offset++
	}
	return &TQMSEQuantizer{
		dims:       cfg.Dims,
		bits:       cfg.Bits,
		levels:     1 << uint(cfg.Bits),
		blockSize:  cfg.RotationBlockSize,
		seed:       cfg.Seed,
		coordSigns: signs,
		centroids:  centroids,
		boundaries: centroidBoundaries(centroids),
	}, nil
}

func (q *TQMSEQuantizer) encodeInto(codes []byte, rotated []float32, vec []float32) error {
	if len(codes) != q.CodeSize() {
		return fmt.Errorf("codes have wrong length: %d (expected %d)", len(codes), q.CodeSize())
	}
	if len(rotated) != q.dims {
		return fmt.Errorf("rotation scratch has wrong dims: %d (expected %d)", len(rotated), q.dims)
	}
	if len(vec) != q.dims {
		return fmt.Errorf("vector has wrong dims: %d (expected %d)", len(vec), q.dims)
	}
	clear(codes)
	q.RotateInto(rotated, vec)
	for dim, value := range rotated {
		setPackedCode(codes, dim, q.bits, q.quantize(value))
	}
	return nil
}

func (q *TQMSEQuantizer) decodeInto(out []float32, rotated []float32, codes []byte) error {
	if len(codes) != q.CodeSize() {
		return fmt.Errorf("codes have wrong length: %d (expected %d)", len(codes), q.CodeSize())
	}
	if len(out) != q.dims {
		return fmt.Errorf("output has wrong dims: %d (expected %d)", len(out), q.dims)
	}
	if len(rotated) != q.dims {
		return fmt.Errorf("rotation scratch has wrong dims: %d (expected %d)", len(rotated), q.dims)
	}
	for dim := 0; dim < q.dims; dim++ {
		rotated[dim] = q.centroids[packedCode(codes, dim, q.bits)]
	}
	q.InverseRotateInto(out, rotated)
	return nil
}

// RotateInto applies the quantizer's orthogonal rotation.
func (q *TQMSEQuantizer) RotateInto(dst []float32, vec []float32) {
	mulFloat32Into(dst, vec, q.coordSigns)
	applyBlockFWHT(dst, q.blockSize)
}

// InverseRotateInto applies the inverse orthogonal rotation.
func (q *TQMSEQuantizer) InverseRotateInto(dst []float32, rotated []float32) {
	copy(dst, rotated)
	applyBlockFWHT(dst, q.blockSize)
	mulFloat32Into(dst, dst, q.coordSigns)
}

func (q *TQMSEQuantizer) quantize(v float32) int {
	idx := sort.Search(len(q.boundaries), func(i int) bool {
		return v < q.boundaries[i]
	})
	if idx >= len(q.centroids) {
		return len(q.centroids) - 1
	}
	return idx
}

// NewTQProdQuantizer creates an inner-product-oriented TQProd quantizer.
func NewTQProdQuantizer(cfg TQProdConfig) (*TQProdQuantizer, error) {
	cfg = defaultTQProdConfig(cfg)
	if err := validateTQProdConfig(cfg); err != nil {
		return nil, err
	}
	mse, err := NewTQMSEQuantizer(TQMSEConfig{
		Dims:              cfg.Dims,
		Bits:              cfg.Bits - 1,
		Seed:              cfg.Seed,
		RotationBlockSize: cfg.RotationBlockSize,
	})
	if err != nil {
		return nil, err
	}
	qjlSize := nextPowerOfTwo(cfg.Dims)
	return newTQProdQuantizerFromParts(cfg, mse, makeSigns(qjlSize, cfg.Seed+211), makeProjectionIndices(qjlSize, cfg.QJLDims, cfg.Seed+307))
}

func defaultTQProdConfig(cfg TQProdConfig) TQProdConfig {
	if cfg.Dims <= 0 {
		cfg.Dims = 768
	}
	if cfg.Bits <= 0 {
		cfg.Bits = 4
	}
	if cfg.QJLDims <= 0 {
		cfg.QJLDims = cfg.Dims
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}
	if cfg.RotationBlockSize <= 0 {
		cfg.RotationBlockSize = largestPowerOfTwoDivisor(cfg.Dims)
	}
	return cfg
}

func validateTQProdConfig(cfg TQProdConfig) error {
	qjlSize := nextPowerOfTwo(cfg.Dims)
	if cfg.Dims <= 0 {
		return fmt.Errorf("dims must be positive, got %d", cfg.Dims)
	}
	if cfg.Bits < 2 || cfg.Bits > 8 {
		return fmt.Errorf("bits must be in [2, 8], got %d", cfg.Bits)
	}
	if cfg.QJLDims <= 0 || cfg.QJLDims > qjlSize {
		return fmt.Errorf("qjl dims must be in [1, %d], got %d", qjlSize, cfg.QJLDims)
	}
	return validateTQMSEConfig(TQMSEConfig{
		Dims:              cfg.Dims,
		Bits:              cfg.Bits - 1,
		Seed:              cfg.Seed,
		RotationBlockSize: cfg.RotationBlockSize,
	})
}

func newTQProdQuantizerFromParts(cfg TQProdConfig, mse *TQMSEQuantizer, qjlSigns []float32, qjlIndices []int) (*TQProdQuantizer, error) {
	qjlSize := nextPowerOfTwo(cfg.Dims)
	if len(qjlSigns) != qjlSize {
		return nil, fmt.Errorf("qjl signs have wrong length: %d (expected %d)", len(qjlSigns), qjlSize)
	}
	if len(qjlIndices) != cfg.QJLDims {
		return nil, fmt.Errorf("qjl indices have wrong length: %d (expected %d)", len(qjlIndices), cfg.QJLDims)
	}
	for i, idx := range qjlIndices {
		if idx < 0 || idx >= qjlSize {
			return nil, fmt.Errorf("qjl index %d out of range: %d", i, idx)
		}
	}
	return &TQProdQuantizer{
		dims:          cfg.Dims,
		bits:          cfg.Bits,
		qjlDims:       cfg.QJLDims,
		qjlSize:       qjlSize,
		qjlSignSize:   packedBitLen(cfg.QJLDims, 1),
		seed:          cfg.Seed,
		residualScale: math.Sqrt(math.Pi*float64(qjlSize)/2.0) / float64(cfg.QJLDims),
		mse:           mse,
		qjlSigns:      append([]float32(nil), qjlSigns...),
		qjlIndices:    append([]int(nil), qjlIndices...),
	}, nil
}

// Dims returns the vector dimension.
func (q *TQProdQuantizer) Dims() int { return q.dims }

// Bits returns the total per-coordinate bit width.
func (q *TQProdQuantizer) Bits() int { return q.bits }

// QJLDims returns the number of residual projection signs stored per vector.
func (q *TQProdQuantizer) QJLDims() int { return q.qjlDims }

// MSEStage returns the stage-1 quantizer.
func (q *TQProdQuantizer) MSEStage() *TQMSEQuantizer { return q.mse }

// CodeSize returns the bytes needed for one TQProd code.
func (q *TQProdQuantizer) CodeSize() int {
	return q.mse.CodeSize() + q.qjlSignSize + 4
}

// CompressionRatio returns the storage ratio versus float32 vectors.
func (q *TQProdQuantizer) CompressionRatio() float64 {
	return float64(q.dims*4) / float64(q.CodeSize())
}

// Encode quantizes a normalized vector with MSE plus QJL residual correction.
// For repeated encoding, create a TQProdEncoder with NewEncoder and call
// EncodeInto to reuse scratch buffers.
func (q *TQProdQuantizer) Encode(vec []float32) (TQProdCode, error) {
	return q.NewEncoder().Encode(vec)
}

// NewEncoder creates a reusable encoder for this quantizer.
func (q *TQProdQuantizer) NewEncoder() *TQProdEncoder {
	scratch := make([]float32, 2*q.dims+q.qjlSize)
	return &TQProdEncoder{
		q:          q,
		rotated:    scratch[:q.dims],
		approx:     scratch[q.dims : 2*q.dims],
		projection: scratch[2*q.dims:],
	}
}

// Encode quantizes a normalized vector with MSE plus QJL residual correction.
func (e *TQProdEncoder) Encode(vec []float32) (TQProdCode, error) {
	if e == nil || e.q == nil {
		return TQProdCode{}, fmt.Errorf("nil TQProd encoder")
	}
	mseSize := e.q.mse.CodeSize()
	data := make([]byte, mseSize+e.q.qjlSignSize)
	return e.EncodeInto(TQProdCode{
		MSECodes:    data[:mseSize],
		QJLSignBits: data[mseSize:],
	}, vec)
}

// EncodeInto quantizes a normalized vector into dst, reusing dst storage when
// it has sufficient capacity.
func (e *TQProdEncoder) EncodeInto(dst TQProdCode, vec []float32) (TQProdCode, error) {
	if e == nil || e.q == nil {
		return TQProdCode{}, fmt.Errorf("nil TQProd encoder")
	}
	q := e.q
	if len(vec) != q.dims {
		return TQProdCode{}, fmt.Errorf("vector has wrong dims: %d (expected %d)", len(vec), q.dims)
	}
	dst.MSECodes = resizeBytes(dst.MSECodes, q.mse.CodeSize())
	dst.QJLSignBits = resizeAndClearBytes(dst.QJLSignBits, q.qjlSignSize)
	if err := q.mse.encodeInto(dst.MSECodes, e.rotated, vec); err != nil {
		return TQProdCode{}, err
	}
	if err := q.mse.decodeInto(e.approx, e.rotated, dst.MSECodes); err != nil {
		return TQProdCode{}, err
	}

	dst.ResidualNorm = q.projectResidualQJLInto(e.projection, vec, e.approx)
	for i, idx := range q.qjlIndices {
		if e.projection[idx] >= 0 {
			setBit(dst.QJLSignBits, i)
		}
	}

	return dst, nil
}

// PrepareQuery builds the query-side MSE and QJL data used by Dot.
func (q *TQProdQuantizer) PrepareQuery(query []float32) (TQProdQuery, error) {
	if len(query) != q.dims {
		return TQProdQuery{}, fmt.Errorf("query has wrong dims: %d (expected %d)", len(query), q.dims)
	}
	mseQuery, err := q.mse.PrepareQuery(query)
	if err != nil {
		return TQProdQuery{}, err
	}
	fullProjection := make([]float32, q.qjlSize)
	q.projectQJLInto(fullProjection, query)
	return TQProdQuery{mse: mseQuery, qjlTables: makeResidualSignTables(fullProjection, q.qjlIndices, q.qjlDims, float32(q.residualScale))}, nil
}

// Dot estimates dot(query, vector) from prepared query data and a TQProd code.
func (q *TQProdQuantizer) Dot(prepared TQProdQuery, code TQProdCode) float64 {
	dot := q.mse.Dot(prepared.mse, TQMSECode{Codes: code.MSECodes})
	if code.ResidualNorm == 0 || len(code.QJLSignBits) == 0 {
		return dot
	}
	correction := dotResidualSignTables(prepared.qjlTables, code.QJLSignBits)
	return dot + float64(code.ResidualNorm)*float64(correction)
}

func (q *TQProdQuantizer) projectQJLInto(dst []float32, vec []float32) {
	for i, value := range vec {
		dst[i] = value * q.qjlSigns[i]
	}
	clear(dst[len(vec):])
	fwhtInPlace(dst)
	scale := float32(1.0 / math.Sqrt(float64(q.qjlSize)))
	for i := range dst {
		dst[i] *= scale
	}
}

func (q *TQProdQuantizer) projectResidualQJLInto(dst []float32, vec []float32, approx []float32) float32 {
	var residualNormSq float64
	for i, value := range vec {
		r := value - approx[i]
		dst[i] = r * q.qjlSigns[i]
		residualNormSq += float64(r * r)
	}
	clear(dst[len(vec):])
	fwhtInPlace(dst)
	scale := float32(1.0 / math.Sqrt(float64(q.qjlSize)))
	for i := range dst {
		dst[i] *= scale
	}
	return float32(math.Sqrt(residualNormSq))
}

// Serialize stores all quantizer metadata needed for stable scoring.
func (q *TQProdQuantizer) Serialize() ([]byte, error) {
	if q == nil || q.mse == nil {
		return nil, fmt.Errorf("nil TQProd quantizer")
	}
	headerSize := 48
	size := headerSize + len(q.mse.centroids)*4 + q.dims + q.qjlDims*4 + q.qjlSize
	buf := make([]byte, size)
	copy(buf[0:4], tqprodMagic)
	binary.LittleEndian.PutUint32(buf[4:8], tqprodVersion)
	binary.LittleEndian.PutUint32(buf[8:12], uint32(q.dims))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(q.bits))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(q.qjlDims))
	binary.LittleEndian.PutUint32(buf[20:24], uint32(q.qjlSize))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(q.mse.blockSize))
	binary.LittleEndian.PutUint64(buf[28:36], q.seed)
	binary.LittleEndian.PutUint32(buf[36:40], uint32(len(q.mse.centroids)))
	binary.LittleEndian.PutUint32(buf[40:44], uint32(len(q.mse.coordSigns)))
	binary.LittleEndian.PutUint32(buf[44:48], uint32(len(q.qjlSigns)))

	offset := headerSize
	for _, centroid := range q.mse.centroids {
		binary.LittleEndian.PutUint32(buf[offset:offset+4], math.Float32bits(centroid))
		offset += 4
	}
	for _, sign := range q.mse.coordSigns {
		if sign > 0 {
			buf[offset] = 1
		}
		offset++
	}
	for _, idx := range q.qjlIndices {
		binary.LittleEndian.PutUint32(buf[offset:offset+4], uint32(idx))
		offset += 4
	}
	for _, sign := range q.qjlSigns {
		if sign > 0 {
			buf[offset] = 1
		}
		offset++
	}
	return buf, nil
}

// DeserializeTQProdQuantizer loads a quantizer serialized by Serialize.
func DeserializeTQProdQuantizer(data []byte) (*TQProdQuantizer, error) {
	if len(data) < 48 {
		return nil, fmt.Errorf("data too short for TQProd header")
	}
	if string(data[0:4]) != tqprodMagic {
		return nil, fmt.Errorf("invalid TQProd magic: %q", string(data[0:4]))
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != tqprodVersion {
		return nil, fmt.Errorf("unsupported TQProd version: %d", version)
	}
	cfg := TQProdConfig{
		Dims:              int(binary.LittleEndian.Uint32(data[8:12])),
		Bits:              int(binary.LittleEndian.Uint32(data[12:16])),
		QJLDims:           int(binary.LittleEndian.Uint32(data[16:20])),
		RotationBlockSize: int(binary.LittleEndian.Uint32(data[24:28])),
		Seed:              binary.LittleEndian.Uint64(data[28:36]),
	}
	qjlSize := int(binary.LittleEndian.Uint32(data[20:24]))
	centroidCount := int(binary.LittleEndian.Uint32(data[36:40]))
	mseSignCount := int(binary.LittleEndian.Uint32(data[40:44]))
	qjlSignCount := int(binary.LittleEndian.Uint32(data[44:48]))
	if err := validateTQProdConfig(cfg); err != nil {
		return nil, err
	}
	if qjlSize != nextPowerOfTwo(cfg.Dims) {
		return nil, fmt.Errorf("qjl size mismatch: %d", qjlSize)
	}
	if centroidCount != 1<<(uint(cfg.Bits)-1) {
		return nil, fmt.Errorf("centroid count mismatch: %d", centroidCount)
	}
	if mseSignCount != cfg.Dims || qjlSignCount != qjlSize {
		return nil, fmt.Errorf("sign count mismatch: mse=%d qjl=%d dims=%d qjlSize=%d", mseSignCount, qjlSignCount, cfg.Dims, qjlSize)
	}

	expected := 48 + centroidCount*4 + mseSignCount + cfg.QJLDims*4 + qjlSignCount
	if len(data) < expected {
		return nil, fmt.Errorf("data too short: %d (expected %d)", len(data), expected)
	}

	offset := 48
	centroids := make([]float32, centroidCount)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4
	}
	mseSigns := make([]float32, mseSignCount)
	for i := range mseSigns {
		if data[offset] == 0 {
			mseSigns[i] = -1
		} else {
			mseSigns[i] = 1
		}
		offset++
	}
	qjlIndices := make([]int, cfg.QJLDims)
	for i := range qjlIndices {
		qjlIndices[i] = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		if qjlIndices[i] < 0 || qjlIndices[i] >= qjlSize {
			return nil, fmt.Errorf("qjl index %d out of range: %d", i, qjlIndices[i])
		}
		offset += 4
	}
	qjlSigns := make([]float32, qjlSignCount)
	for i := range qjlSigns {
		if data[offset] == 0 {
			qjlSigns[i] = -1
		} else {
			qjlSigns[i] = 1
		}
		offset++
	}

	mse := &TQMSEQuantizer{
		dims:       cfg.Dims,
		bits:       cfg.Bits - 1,
		levels:     1 << uint(cfg.Bits-1),
		blockSize:  cfg.RotationBlockSize,
		seed:       cfg.Seed,
		coordSigns: mseSigns,
		centroids:  centroids,
		boundaries: centroidBoundaries(centroids),
	}
	return newTQProdQuantizerFromParts(cfg, mse, qjlSigns, qjlIndices)
}

func normalLloydMaxCodebook(levels int, dims int, seed uint64) []float32 {
	if levels <= 1 {
		return []float32{0}
	}
	sampleCount := levels * 4096
	if sampleCount < 32768 {
		sampleCount = 32768
	}
	if sampleCount > 131072 {
		sampleCount = 131072
	}

	sigma := 1.0 / math.Sqrt(float64(dims))
	rng := rand.New(rand.NewSource(int64(seed)))
	samples := make([]float64, sampleCount)
	for i := range samples {
		samples[i] = rng.NormFloat64() * sigma
	}
	sort.Float64s(samples)

	centroids := make([]float64, levels)
	for i := range centroids {
		idx := int((float64(i) + 0.5) * float64(sampleCount) / float64(levels))
		if idx >= sampleCount {
			idx = sampleCount - 1
		}
		centroids[i] = samples[idx]
	}

	const iterations = 30
	sums := make([]float64, levels)
	counts := make([]int, levels)
	for iter := 0; iter < iterations; iter++ {
		clear(sums)
		clear(counts)
		bucket := 0
		for _, sample := range samples {
			for bucket < levels-1 && sample >= (centroids[bucket]+centroids[bucket+1])/2 {
				bucket++
			}
			sums[bucket] += sample
			counts[bucket]++
		}
		for i := range centroids {
			if counts[i] > 0 {
				centroids[i] = sums[i] / float64(counts[i])
			}
		}
	}

	out := make([]float32, levels)
	for i := range centroids {
		out[i] = float32(centroids[i])
	}
	return out
}

func centroidBoundaries(centroids []float32) []float32 {
	if len(centroids) <= 1 {
		return nil
	}
	boundaries := make([]float32, len(centroids)-1)
	for i := range boundaries {
		boundaries[i] = (centroids[i] + centroids[i+1]) / 2
	}
	return boundaries
}

func makeSigns(dims int, seed uint64) []float32 {
	signs := make([]float32, dims)
	for i := range signs {
		if splitmix64(seed+uint64(i)*0x9e3779b97f4a7c15)&1 == 0 {
			signs[i] = -1
		} else {
			signs[i] = 1
		}
	}
	return signs
}

func makeProjectionIndices(dims int, qjlDims int, seed uint64) []int {
	indices := make([]int, dims)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(int64(seed)))
	for i := len(indices) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}
	return indices[:qjlDims]
}

func largestPowerOfTwoDivisor(n int) int {
	if n <= 0 {
		return 1
	}
	size := 1
	for next := 2; next <= n && n%next == 0; next *= 2 {
		size = next
	}
	return size
}

func nextPowerOfTwo(n int) int {
	if n <= 1 {
		return 1
	}
	size := 1
	for size < n {
		size *= 2
	}
	return size
}

func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

func applyBlockFWHT(values []float32, blockSize int) {
	for start := 0; start < len(values); start += blockSize {
		fwhtInPlace(values[start : start+blockSize])
	}
	scale := float32(1.0 / math.Sqrt(float64(blockSize)))
	scaleFloat32(values, scale)
}

func fwhtInPlace(values []float32) {
	step := 1
	for ; step < len(values) && step < 4; step *= 2 {
		fwhtStepFloat32Scalar(values, step)
	}
	for ; step < len(values); step *= 2 {
		fwhtStepFloat32(values, step)
	}
}

func packedBitLen(items int, bits int) int {
	return (items*bits + 7) / 8
}

func resizeBytes(data []byte, size int) []byte {
	if cap(data) < size {
		return make([]byte, size)
	}
	return data[:size]
}

func resizeAndClearBytes(data []byte, size int) []byte {
	data = resizeBytes(data, size)
	clear(data)
	return data
}

func packedCode(data []byte, idx int, bits int) int {
	bitOffset := idx * bits
	byteIdx := bitOffset >> 3
	shift := uint(bitOffset & 7)
	word := uint32(data[byteIdx])
	if byteIdx+1 < len(data) {
		word |= uint32(data[byteIdx+1]) << 8
	}
	if byteIdx+2 < len(data) {
		word |= uint32(data[byteIdx+2]) << 16
	}
	mask := uint32(1<<uint(bits)) - 1
	return int((word >> shift) & mask)
}

func setPackedCode(data []byte, idx int, bits int, value int) {
	bitOffset := idx * bits
	for b := 0; b < bits; b++ {
		pos := bitOffset + b
		mask := byte(1 << uint(pos%8))
		if value&(1<<uint(b)) != 0 {
			data[pos/8] |= mask
		} else {
			data[pos/8] &^= mask
		}
	}
}

func setBit(data []byte, idx int) {
	data[idx/8] |= 1 << uint(idx%8)
}

func dotPackedCodes(table []float64, data []byte, dims int, bits int, levels int) float64 {
	switch bits {
	case 1:
		return dotPackedCodes1(table, data, dims)
	case 2:
		return dotPackedCodes2(table, data, dims)
	case 3:
		return dotPackedCodes3(table, data, dims)
	case 4:
		return dotPackedCodes4(table, data, dims)
	case 8:
		return dotPackedCodes8(table, data, dims)
	}
	var dot float64
	for dim := 0; dim < dims; dim++ {
		code := packedCode(data, dim, bits)
		dot += table[dim*levels+code]
	}
	return dot
}

func dotPackedCodes1(table []float64, data []byte, dims int) float64 {
	var dot float64
	dim := 0
	for _, word := range data {
		for shift := uint(0); shift < 8 && dim < dims; shift++ {
			code := int((word >> shift) & 1)
			dot += table[dim*2+code]
			dim++
		}
	}
	return dot
}

func dotPackedCodes2(table []float64, data []byte, dims int) float64 {
	var dot float64
	dim := 0
	for _, word := range data {
		for shift := uint(0); shift < 8 && dim < dims; shift += 2 {
			code := int((word >> shift) & 3)
			dot += table[dim*4+code]
			dim++
		}
	}
	return dot
}

func dotPackedCodes3(table []float64, data []byte, dims int) float64 {
	var dot float64
	var word uint32
	var wordBits uint
	byteIdx := 0
	for dim := 0; dim < dims; dim++ {
		for wordBits < 3 {
			word |= uint32(data[byteIdx]) << wordBits
			wordBits += 8
			byteIdx++
		}
		code := int(word & 7)
		dot += table[dim*8+code]
		word >>= 3
		wordBits -= 3
	}
	return dot
}

func dotPackedCodes4(table []float64, data []byte, dims int) float64 {
	var dot float64
	dim := 0
	for _, word := range data {
		low := int(word & 15)
		dot += table[dim*16+low]
		dim++
		if dim >= dims {
			break
		}
		high := int(word >> 4)
		dot += table[dim*16+high]
		dim++
	}
	return dot
}

func dotPackedCodes8(table []float64, data []byte, dims int) float64 {
	var dot float64
	for dim := 0; dim < dims; dim++ {
		dot += table[dim*256+int(data[dim])]
	}
	return dot
}

func dotResidualSigns(qjl []float64, data []byte, dims int) float64 {
	var dot float64
	dim := 0
	for _, word := range data {
		for shift := uint(0); shift < 8 && dim < dims; shift++ {
			if word&(1<<shift) != 0 {
				dot += qjl[dim]
			} else {
				dot -= qjl[dim]
			}
			dim++
		}
	}
	return dot
}

func makeResidualSignTables(projection []float32, indices []int, dims int, scale float32) []float32 {
	tableCount := packedBitLen(dims, 1)
	tables := make([]float32, tableCount*256)
	for tableIdx := 0; tableIdx < tableCount; tableIdx++ {
		baseDim := tableIdx * 8
		table := tables[tableIdx*256 : (tableIdx+1)*256]
		for pattern := range table {
			var sum float32
			for bit := 0; bit < 8; bit++ {
				dim := baseDim + bit
				if dim >= dims {
					break
				}
				qjl := projection[indices[dim]] * scale
				if pattern&(1<<uint(bit)) != 0 {
					sum += qjl
				} else {
					sum -= qjl
				}
			}
			table[pattern] = sum
		}
	}
	return tables
}

func dotResidualSignTables(tables []float32, data []byte) float32 {
	var dot float32
	for i := 0; i < len(data); i++ {
		dot += tables[i<<8+int(data[i])]
	}
	return dot
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
