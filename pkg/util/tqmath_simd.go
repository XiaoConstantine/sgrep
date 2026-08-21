//go:build goexperiment.simd && !arm64

package util

import "simd"

const maxSIMDFloat32Lanes = 64

func dotProductFloat32(a, b []float32) float64 {
	lanes := simd.VectorBitSize() / 32
	if len(a) < 32 || simd.Emulated() || lanes <= 0 || lanes > maxSIMDFloat32Lanes {
		return dotProductUnrolled8Scalar(a, b)
	}

	var sums simd.Float32s
	i := 0
	for ; i+lanes <= len(a); i += lanes {
		sums = simd.LoadFloat32s(a[i:]).MulAdd(simd.LoadFloat32s(b[i:]), sums)
	}

	var partial [maxSIMDFloat32Lanes]float32
	sums.Store(partial[:lanes])
	var dot float64
	for _, value := range partial[:lanes] {
		dot += float64(value)
	}
	for ; i < len(a); i++ {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
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
	lanes := simd.VectorBitSize() / 32
	if simd.Emulated() || lanes <= 0 {
		scaleFloat32Scalar(values, scale)
		return
	}

	factor := simd.BroadcastFloat32s(scale)
	i := 0
	for ; i+lanes <= len(values); i += lanes {
		simd.LoadFloat32s(values[i:]).Mul(factor).Store(values[i:])
	}
	scaleFloat32Scalar(values[i:], scale)
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
