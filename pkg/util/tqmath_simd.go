//go:build goexperiment.simd && !arm64

package util

import "simd"

func dotProductFloat32(a, b []float32) float64 {
	// Portable SIMD has no float32-to-float64 widening operation. Accumulating
	// in Float32s changes score semantics, and MulAdd requires FMA even when Go
	// 1.27 selects hardware SIMD on an AVX-only CPU. Preserve the scalar result.
	return dotProductUnrolled8Scalar(a, b)
}

func mulFloat32Into(dst, a, b []float32) {
	lanes := simd.VectorBitSize() / 32
	if simd.Emulated() || lanes <= 0 {
		mulFloat32Scalar(dst, a, b)
		return
	}

	i := 0
	for ; i+lanes <= len(a); i += lanes {
		simd.LoadFloat32s(a[i:]).Mul(simd.LoadFloat32s(b[i:])).Store(dst[i:])
	}
	mulFloat32Scalar(dst[i:], a[i:], b[i:])
}

func scaleFloat32(values []float32, scale float32) {
	// Go 1.27 lowers BroadcastFloat32s to an AVX2-only register broadcast,
	// although its 128-bit SIMD dispatcher only guarantees AVX. Keep scaling
	// scalar so AVX-only hosts cannot reach an unsupported instruction.
	scaleFloat32Scalar(values, scale)
}

func fwhtStepFloat32(values []float32, step int) {
	lanes := simd.VectorBitSize() / 32
	if simd.Emulated() || lanes <= 0 || step < lanes {
		fwhtStepFloat32Scalar(values, step)
		return
	}

	for start := 0; start < len(values); start += 2 * step {
		i := 0
		for ; i+lanes <= step; i += lanes {
			left := simd.LoadFloat32s(values[start+i:])
			right := simd.LoadFloat32s(values[start+i+step:])
			left.Add(right).Store(values[start+i:])
			left.Sub(right).Store(values[start+i+step:])
		}
		for ; i < step; i++ {
			left := values[start+i]
			right := values[start+i+step]
			values[start+i] = left + right
			values[start+i+step] = left - right
		}
	}
}

func buildTQMSETable(table []float64, rotated, centroids []float32, levels int) {
	if levels == 16 {
		buildTQMSETable16Scalar(table, rotated, centroids)
		return
	}
	buildTQMSETableScalar(table, rotated, centroids, levels)
}
