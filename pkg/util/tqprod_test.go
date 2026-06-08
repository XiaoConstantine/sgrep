package util

import (
	"math"
	"math/rand"
	"testing"
)

func TestTQMSEEncodeDecodeAndDot(t *testing.T) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 64, Bits: 4, Seed: 11})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(64, 1)
	query := randomUnitVectorForTQTest(64, 2)

	code, err := q.Encode(doc)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	decoded, err := q.Decode(code)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	prepared, err := q.PrepareQuery(query)
	if err != nil {
		t.Fatalf("PrepareQuery: %v", err)
	}

	got := q.Dot(prepared, code)
	want := DotProductUnrolled8(query, decoded)
	if diff := math.Abs(got - want); diff > 1e-6 {
		t.Fatalf("Dot = %.8f, decoded dot = %.8f, diff %.8f", got, want, diff)
	}
}

func TestTQMSESerializationRoundTrip(t *testing.T) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 64, Bits: 4, Seed: 11})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(64, 12)
	query := randomUnitVectorForTQTest(64, 13)
	code, err := q.Encode(doc)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	prepared, err := q.PrepareQuery(query)
	if err != nil {
		t.Fatalf("PrepareQuery: %v", err)
	}
	want := q.Dot(prepared, code)

	data, err := q.Serialize()
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	loaded, err := DeserializeTQMSEQuantizer(data)
	if err != nil {
		t.Fatalf("DeserializeTQMSEQuantizer: %v", err)
	}
	loadedPrepared, err := loaded.PrepareQuery(query)
	if err != nil {
		t.Fatalf("loaded PrepareQuery: %v", err)
	}
	got := loaded.Dot(loadedPrepared, code)
	if diff := math.Abs(got - want); diff > 1e-9 {
		t.Fatalf("loaded Dot = %.12f, want %.12f, diff %.12f", got, want, diff)
	}
}

func TestTQMSERotationIsInvertible(t *testing.T) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 96, Bits: 3, Seed: 3, RotationBlockSize: 32})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	vec := randomUnitVectorForTQTest(96, 4)
	rotated := make([]float32, len(vec))
	q.RotateInto(rotated, vec)
	got := make([]float32, len(vec))
	q.InverseRotateInto(got, rotated)

	for i := range vec {
		if diff := math.Abs(float64(got[i] - vec[i])); diff > 1e-5 {
			t.Fatalf("inverse rotation mismatch at %d: got %.8f want %.8f diff %.8f", i, got[i], vec[i], diff)
		}
	}
}

func TestTQProdSerializationRoundTrip(t *testing.T) {
	q, err := NewTQProdQuantizer(TQProdConfig{Dims: 64, Bits: 4, QJLDims: 32, Seed: 9})
	if err != nil {
		t.Fatalf("NewTQProdQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(64, 10)
	query := randomUnitVectorForTQTest(64, 11)
	code, err := q.Encode(doc)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	prepared, err := q.PrepareQuery(query)
	if err != nil {
		t.Fatalf("PrepareQuery: %v", err)
	}
	want := q.Dot(prepared, code)

	data, err := q.Serialize()
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	q2, err := DeserializeTQProdQuantizer(data)
	if err != nil {
		t.Fatalf("DeserializeTQProdQuantizer: %v", err)
	}
	prepared2, err := q2.PrepareQuery(query)
	if err != nil {
		t.Fatalf("PrepareQuery roundtrip: %v", err)
	}
	got := q2.Dot(prepared2, code)
	if diff := math.Abs(got - want); diff > 1e-12 {
		t.Fatalf("roundtrip dot = %.12f, want %.12f, diff %.12f", got, want, diff)
	}
}

func TestTQProdImprovesOverMSEStageOnAverage(t *testing.T) {
	const dims = 256
	q, err := NewTQProdQuantizer(TQProdConfig{Dims: dims, Bits: 4, QJLDims: dims, Seed: 21})
	if err != nil {
		t.Fatalf("NewTQProdQuantizer: %v", err)
	}

	var mseErr float64
	var prodErr float64
	const trials = 80
	for i := 0; i < trials; i++ {
		doc := randomUnitVectorForTQTest(dims, int64(100+i))
		query := randomUnitVectorForTQTest(dims, int64(1000+i))
		exact := DotProductUnrolled8(query, doc)

		mseCode, err := q.MSEStage().Encode(doc)
		if err != nil {
			t.Fatalf("MSE Encode(%d): %v", i, err)
		}
		msePrepared, err := q.MSEStage().PrepareQuery(query)
		if err != nil {
			t.Fatalf("MSE PrepareQuery(%d): %v", i, err)
		}
		mseErr += math.Abs(exact - q.MSEStage().Dot(msePrepared, mseCode))

		prodCode, err := q.Encode(doc)
		if err != nil {
			t.Fatalf("TQProd Encode(%d): %v", i, err)
		}
		prodPrepared, err := q.PrepareQuery(query)
		if err != nil {
			t.Fatalf("TQProd PrepareQuery(%d): %v", i, err)
		}
		prodErr += math.Abs(exact - q.Dot(prodPrepared, prodCode))
	}

	if prodErr >= mseErr {
		t.Fatalf("TQProd average error should improve MSE stage: prod=%.8f mse=%.8f", prodErr/trials, mseErr/trials)
	}
}

func TestTQProdRejectsInvalidConfig(t *testing.T) {
	if _, err := NewTQProdQuantizer(TQProdConfig{Dims: 64, Bits: 1}); err == nil {
		t.Fatal("NewTQProdQuantizer accepted bits=1")
	}
	if _, err := NewTQProdQuantizer(TQProdConfig{Dims: 64, Bits: 4, QJLDims: 65}); err == nil {
		t.Fatal("NewTQProdQuantizer accepted qjlDims > dims")
	}
	if _, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 63, Bits: 4, RotationBlockSize: 32}); err == nil {
		t.Fatal("NewTQMSEQuantizer accepted non-divisible rotation block size")
	}
}

func TestTQProdPreparedScoreNoAllocs(t *testing.T) {
	q, err := NewTQProdQuantizer(TQProdConfig{Dims: 128, Bits: 4, QJLDims: 128, Seed: 31})
	if err != nil {
		t.Fatalf("NewTQProdQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(128, 32)
	query := randomUnitVectorForTQTest(128, 33)
	code, err := q.Encode(doc)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	prepared, err := q.PrepareQuery(query)
	if err != nil {
		t.Fatalf("PrepareQuery: %v", err)
	}

	allocs := testing.AllocsPerRun(100, func() {
		_ = q.Dot(prepared, code)
	})
	if allocs != 0 {
		t.Fatalf("Dot allocated %.2f times per run, want 0", allocs)
	}
}

func TestTQProdEncoderEncodeIntoNoAllocs(t *testing.T) {
	q, err := NewTQProdQuantizer(TQProdConfig{Dims: 128, Bits: 4, QJLDims: 128, Seed: 34})
	if err != nil {
		t.Fatalf("NewTQProdQuantizer: %v", err)
	}
	encoder := q.NewEncoder()
	doc := randomUnitVectorForTQTest(128, 35)
	code := TQProdCode{
		MSECodes:    make([]byte, q.MSEStage().CodeSize()),
		QJLSignBits: make([]byte, packedBitLen(q.QJLDims(), 1)),
	}
	if _, err := encoder.EncodeInto(code, doc); err != nil {
		t.Fatalf("EncodeInto warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(100, func() {
		if _, err := encoder.EncodeInto(code, doc); err != nil {
			t.Fatalf("EncodeInto: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("EncodeInto allocated %.2f times per run, want 0", allocs)
	}
}

func TestPackedCodeRoundTrip(t *testing.T) {
	for bits := 1; bits <= 8; bits++ {
		levels := 1 << bits
		data := make([]byte, packedBitLen(17, bits))
		for i := 0; i < 17; i++ {
			setPackedCode(data, i, bits, i%levels)
		}
		for i := 0; i < 17; i++ {
			if got, want := packedCode(data, i, bits), i%levels; got != want {
				t.Fatalf("bits=%d idx=%d code=%d, want %d", bits, i, got, want)
			}
		}
	}
}

func TestTQFastPackedDotMatchesGeneric(t *testing.T) {
	dims := 19
	for bits := 1; bits <= 8; bits++ {
		levels := 1 << bits
		table := make([]float64, dims*levels)
		for i := range table {
			table[i] = math.Sin(float64(i) * 0.37)
		}

		codes := make([]byte, packedBitLen(dims, bits))
		for dim := 0; dim < dims; dim++ {
			setPackedCode(codes, dim, bits, (dim*7+3)%levels)
		}

		var want float64
		for dim := 0; dim < dims; dim++ {
			code := packedCode(codes, dim, bits)
			want += table[dim*levels+code]
		}

		got := dotPackedCodes(table, codes, dims, bits, levels)
		if diff := got - want; diff < -1e-12 || diff > 1e-12 {
			t.Fatalf("bits=%d fast dot = %.12f, generic = %.12f, diff %.12f", bits, got, want, diff)
		}
	}
}

func TestResidualSignTablesMatchGeneric(t *testing.T) {
	dims := 23
	qjl32 := make([]float32, dims)
	qjl64 := make([]float64, dims)
	indices := make([]int, dims)
	bits := make([]byte, packedBitLen(dims, 1))
	for i := range qjl32 {
		indices[i] = i
		value := math.Cos(float64(i) * 0.19)
		qjl32[i] = float32(value)
		qjl64[i] = float64(qjl32[i])
		if i%3 == 1 {
			setBit(bits, i)
		}
	}

	tables := makeResidualSignTables(qjl32, indices, dims, 1)
	got := float64(dotResidualSignTables(tables, bits))
	want := dotResidualSigns(qjl64, bits, dims)
	if diff := got - want; diff < -1e-6 || diff > 1e-6 {
		t.Fatalf("residual table dot = %.12f, generic = %.12f, diff %.12f", got, want, diff)
	}
}

func BenchmarkTQMSERotate768(b *testing.B) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 768, Bits: 4, Seed: 41})
	if err != nil {
		b.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(768, 42)

	b.Run("accelerated", func(b *testing.B) {
		rotated := make([]float32, q.Dims())
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q.RotateInto(rotated, doc)
		}
		benchmarkTQMSEFloat32Sink = rotated[0]
	})

	b.Run("scalar-reference", func(b *testing.B) {
		rotated := make([]float32, q.Dims())
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			mulFloat32Scalar(rotated, doc, q.coordSigns)
			applyBlockFWHTScalar(rotated, q.blockSize)
		}
		benchmarkTQMSEFloat32Sink = rotated[0]
	})
}

func BenchmarkTQMSELUT768(b *testing.B) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 768, Bits: 4, Seed: 43})
	if err != nil {
		b.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	query := randomUnitVectorForTQTest(768, 44)
	rotated := make([]float32, q.Dims())
	q.RotateInto(rotated, query)

	b.Run("specialized16", func(b *testing.B) {
		table := make([]float64, q.Dims()*q.levels)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buildTQMSETable(table, rotated, q.centroids, q.levels)
		}
		benchmarkTQMSEFloat64Sink = table[0]
	})

	b.Run("generic", func(b *testing.B) {
		table := make([]float64, q.Dims()*q.levels)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buildTQMSETableScalar(table, rotated, q.centroids, q.levels)
		}
		benchmarkTQMSEFloat64Sink = table[0]
	})
}

func BenchmarkTQMSEPrepareQuery768(b *testing.B) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 768, Bits: 4, Seed: 45})
	if err != nil {
		b.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	query := randomUnitVectorForTQTest(768, 46)

	b.ReportAllocs()
	var prepared TQMSEQuery
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prepared, err = q.PrepareQuery(query)
		if err != nil {
			b.Fatalf("PrepareQuery: %v", err)
		}
	}
	benchmarkTQMSEQuerySink = prepared
}

func BenchmarkTQMSEEncode768(b *testing.B) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 768, Bits: 4, Seed: 47})
	if err != nil {
		b.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(768, 48)

	b.Run("one-shot", func(b *testing.B) {
		b.ReportAllocs()
		var code TQMSECode
		for i := 0; i < b.N; i++ {
			code, err = q.Encode(doc)
			if err != nil {
				b.Fatalf("Encode: %v", err)
			}
		}
		benchmarkTQMSECodeSink = code
	})

	b.Run("encode-into", func(b *testing.B) {
		encoder := q.NewEncoder()
		code := TQMSECode{Codes: make([]byte, q.CodeSize())}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			code, err = encoder.EncodeInto(code, doc)
			if err != nil {
				b.Fatalf("EncodeInto: %v", err)
			}
		}
		benchmarkTQMSECodeSink = code
	})
}

func BenchmarkTQProdDot768(b *testing.B) {
	q, err := NewTQProdQuantizer(TQProdConfig{Dims: 768, Bits: 4, QJLDims: 768, Seed: 41})
	if err != nil {
		b.Fatalf("NewTQProdQuantizer: %v", err)
	}
	query := randomUnitVectorForTQTest(768, 42)
	prepared, err := q.PrepareQuery(query)
	if err != nil {
		b.Fatalf("PrepareQuery: %v", err)
	}
	codes := make([]TQProdCode, 256)
	for i := range codes {
		doc := randomUnitVectorForTQTest(768, int64(100+i))
		codes[i], err = q.Encode(doc)
		if err != nil {
			b.Fatalf("Encode(%d): %v", i, err)
		}
	}

	b.ReportAllocs()
	var sink float64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, code := range codes {
			sink += q.Dot(prepared, code)
		}
	}
	benchmarkTQProdSink = sink
}

func BenchmarkTQProdEncode768(b *testing.B) {
	q, err := NewTQProdQuantizer(TQProdConfig{Dims: 768, Bits: 4, QJLDims: 768, Seed: 51})
	if err != nil {
		b.Fatalf("NewTQProdQuantizer: %v", err)
	}
	doc := randomUnitVectorForTQTest(768, 52)

	b.Run("one-shot", func(b *testing.B) {
		b.ReportAllocs()
		var code TQProdCode
		for i := 0; i < b.N; i++ {
			code, err = q.Encode(doc)
			if err != nil {
				b.Fatalf("Encode: %v", err)
			}
		}
		benchmarkTQProdCodeSink = code
	})

	b.Run("encode-into", func(b *testing.B) {
		encoder := q.NewEncoder()
		code := TQProdCode{
			MSECodes:    make([]byte, q.MSEStage().CodeSize()),
			QJLSignBits: make([]byte, packedBitLen(q.QJLDims(), 1)),
		}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			code, err = encoder.EncodeInto(code, doc)
			if err != nil {
				b.Fatalf("EncodeInto: %v", err)
			}
		}
		benchmarkTQProdCodeSink = code
	})
}

var benchmarkTQProdSink float64
var benchmarkTQProdCodeSink TQProdCode
var benchmarkTQMSEFloat32Sink float32
var benchmarkTQMSEFloat64Sink float64
var benchmarkTQMSEQuerySink TQMSEQuery
var benchmarkTQMSECodeSink TQMSECode

func randomUnitVectorForTQTest(dims int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}
	return NormalizeVector(vec)
}
