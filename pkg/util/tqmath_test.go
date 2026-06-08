package util

import (
	"math"
	"testing"
)

func TestTQMathKernelsMatchScalar(t *testing.T) {
	a := []float32{1, -2, 3.5, -4.25, 5.75, -6.5, 7.25}
	b := []float32{-1, -1, 1, -1, 1, 1, -1}

	gotMul := make([]float32, len(a))
	wantMul := make([]float32, len(a))
	mulFloat32Into(gotMul, a, b)
	mulFloat32Scalar(wantMul, a, b)
	compareFloat32Slices(t, "mulFloat32Into", gotMul, wantMul, 0)

	gotScale := append([]float32(nil), a...)
	wantScale := append([]float32(nil), a...)
	scaleFloat32(gotScale, 0.25)
	scaleFloat32Scalar(wantScale, 0.25)
	compareFloat32Slices(t, "scaleFloat32", gotScale, wantScale, 0)

	for _, step := range []int{1, 2, 4, 8, 16} {
		gotFWHT := make([]float32, 32)
		wantFWHT := make([]float32, 32)
		for i := range gotFWHT {
			value := float32(math.Sin(float64(i) * 0.25))
			gotFWHT[i] = value
			wantFWHT[i] = value
		}
		fwhtStepFloat32(gotFWHT, step)
		fwhtStepFloat32Scalar(wantFWHT, step)
		compareFloat32Slices(t, "fwhtStepFloat32", gotFWHT, wantFWHT, 0)
	}
}

func TestTQMSERotateMatchesScalarReference(t *testing.T) {
	q, err := NewTQMSEQuantizer(TQMSEConfig{Dims: 768, Bits: 4, Seed: 29})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer: %v", err)
	}
	vec := randomUnitVectorForTQTest(768, 30)
	got := make([]float32, len(vec))
	want := make([]float32, len(vec))

	q.RotateInto(got, vec)
	mulFloat32Scalar(want, vec, q.coordSigns)
	applyBlockFWHTScalar(want, q.blockSize)

	compareFloat32Slices(t, "RotateInto", got, want, 0)
}

func applyBlockFWHTScalar(values []float32, blockSize int) {
	for start := 0; start < len(values); start += blockSize {
		fwhtInPlaceScalar(values[start : start+blockSize])
	}
	scale := float32(1.0 / math.Sqrt(float64(blockSize)))
	scaleFloat32Scalar(values, scale)
}

func fwhtInPlaceScalar(values []float32) {
	for step := 1; step < len(values); step *= 2 {
		fwhtStepFloat32Scalar(values, step)
	}
}

func compareFloat32Slices(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tolerance {
			t.Fatalf("%s mismatch at %d: got %.8g want %.8g diff %.8g", name, i, got[i], want[i], diff)
		}
	}
}
